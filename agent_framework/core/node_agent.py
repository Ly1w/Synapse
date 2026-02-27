from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..communication.channel import Channel
from ..communication.message import Message, MessageType
from ..communication.router import Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient
from ..skills.reader import SkillsReader
from ..tools.executor import ToolExecutor
from ..tools.ptc import PTCExecutor
from ..tools.registry import ToolRegistry
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

NODE_SYSTEM_PROMPT = """\
You are a Node Agent — the task executor in a hierarchical multi-agent system.

Your role: {role}
Your task: {task}

Rules:
1. Focus ONLY on your assigned task. Do not do work outside your scope.
2. Use the available tools to accomplish your task.
3. When you discover important information outside your scope, use ESCALATE \
to suggest your Head Agent create a new Node for it.
4. When you need to coordinate with a sibling Node, use SIBLING_MSG.
5. When your task is complete, use REPORT to notify your Head Agent.
6. You may use Programmatic Tool Calling (PTC) for complex multi-tool workflows.

Output format for actions:
- To call a tool: use the standard function calling interface
- To use PTC: output a code block tagged with ```ptc ... ```
- To escalate: output ESCALATE: {{description of discovered work}}
- To message a sibling: output SIBLING(agent_id): {{message}}
- To report completion: output REPORT: {{summary of results}}

{ptc_instructions}
"""


class NodeAgent(BaseAgent):
    """
    Task executor agent. Calls tools, generates PTC code, communicates
    with siblings, and reports to its Head Agent.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        task: str,
        head_id: str,
        llm_client: LLMClient,
        router: Router,
        channel: Channel,
        tool_registry: ToolRegistry,
        skills_reader: SkillsReader | None = None,
        context_manager: ContextManager | None = None,
        max_turns: int = 30,
    ):
        self.task = task
        self.head_id = head_id
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)
        self.ptc_executor = PTCExecutor(tool_registry)
        self.skills_reader = skills_reader
        self._completed = False
        self._result: str = ""

        ptc_instructions = self.ptc_executor.get_ptc_prompt_instructions()
        system_prompt = NODE_SYSTEM_PROMPT.format(
            role=role, task=task, ptc_instructions=ptc_instructions
        )

        super().__init__(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            llm_client=llm_client,
            router=router,
            channel=channel,
            context_manager=context_manager,
            max_turns=max_turns,
        )

    async def run(self) -> dict[str, Any]:
        """
        Main execution loop: reason, call tools, handle PTC, and communicate.
        Returns the final result dict.
        """
        self._running = True
        logger.info("NodeAgent %s starting task: %s", self.id, self.task[:100])

        # Load relevant tools via skills reader
        tools = await self._load_tools()

        turn = 0
        current_input = f"Execute this task:\n{self.task}"

        while self._running and turn < self.max_turns and not self._completed:
            turn += 1

            # Check for incoming messages (non-blocking)
            await self._process_incoming_messages()

            if tools:
                response = await self.think_with_tools(current_input, tools)
                msg = response.choices[0].message

                # Handle tool calls
                if msg.tool_calls:
                    tool_results = await self.tool_executor.execute_tool_calls([
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ])
                    for tr in tool_results:
                        self.add_tool_result(tr["tool_call_id"], tr["content"])
                    current_input = "Continue based on the tool results."
                    continue

                text = msg.content or ""
            else:
                text = await self.think(current_input)

            # Parse agent actions from the response text
            action = self._parse_action(text)

            if action["type"] == "ptc":
                ptc_result = await self.ptc_executor.execute_code(action["code"])
                ptc_summary = json.dumps(ptc_result, indent=2)[:2000]
                current_input = f"PTC execution result:\n{ptc_summary}\n\nContinue or report."

            elif action["type"] == "escalate":
                await self.send_message(
                    self.head_id,
                    MessageType.ESCALATION,
                    {"text": action["content"], "source_agent": self.id},
                )
                current_input = "Escalation sent. Continue with your own task."

            elif action["type"] == "sibling":
                await self.send_message(
                    action["target"],
                    MessageType.PEER_MSG,
                    {"text": action["content"], "from": self.id},
                )
                current_input = "Message sent to sibling. Continue with your task."

            elif action["type"] == "report":
                self._completed = True
                self._result = action["content"]
                await self.send_message(
                    self.head_id,
                    MessageType.REPORT,
                    {
                        "text": action["content"],
                        "agent_id": self.id,
                        "role": self.role,
                        "status": "completed",
                    },
                )
            else:
                current_input = "Continue working on your task. Remember to REPORT when done."

        if not self._completed:
            # Force report if max turns reached
            summary = self._result or "Task incomplete: max turns reached."
            await self.send_message(
                self.head_id,
                MessageType.REPORT,
                {
                    "text": summary,
                    "agent_id": self.id,
                    "role": self.role,
                    "status": "incomplete",
                },
            )

        logger.info("NodeAgent %s finished (completed=%s)", self.id, self._completed)
        return {"completed": self._completed, "result": self._result}

    async def _load_tools(self) -> list[dict[str, Any]]:
        """Load relevant tools via skills reader or fall back to all tools."""
        if self.skills_reader:
            try:
                return await self.skills_reader.find_tools(self.task)
            except Exception as e:
                logger.warning("Skills reader failed: %s, using all tools", e)
        return self.tool_registry.get_openai_schemas()

    async def _process_incoming_messages(self) -> None:
        """Process any pending messages from siblings or head."""
        messages = self.drain_messages()
        for msg in messages:
            if msg.msg_type == MessageType.PEER_MSG:
                info = msg.content.get("text", str(msg.content))
                self.context.add_message({
                    "role": "user",
                    "content": f"[Sibling {msg.sender_id}]: {info}",
                })
            elif msg.msg_type == MessageType.CLARIFICATION:
                info = msg.content.get("text", str(msg.content))
                self.context.add_message({
                    "role": "user",
                    "content": f"[Head Agent]: {info}",
                })

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any]:
        """Parse the agent's response for structured actions."""
        # PTC code block
        ptc_match = re.search(r"```ptc\s*\n(.*?)```", text, re.DOTALL)
        if ptc_match:
            return {"type": "ptc", "code": ptc_match.group(1).strip()}

        # Escalation
        esc_match = re.search(r"ESCALATE:\s*(.+)", text, re.DOTALL)
        if esc_match:
            return {"type": "escalate", "content": esc_match.group(1).strip()}

        # Sibling message
        sib_match = re.search(r"SIBLING\((\S+)\):\s*(.+)", text, re.DOTALL)
        if sib_match:
            return {
                "type": "sibling",
                "target": sib_match.group(1),
                "content": sib_match.group(2).strip(),
            }

        # Report
        rep_match = re.search(r"REPORT:\s*(.+)", text, re.DOTALL)
        if rep_match:
            return {"type": "report", "content": rep_match.group(1).strip()}

        return {"type": "continue", "content": text}
