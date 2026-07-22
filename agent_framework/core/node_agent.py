from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from ..communication.channel import Channel
from ..communication.message import MessageType
from ..communication.router import Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient
from ..skills.reader import SkillsReader
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from .base_agent import BaseAgent
from .contracts import AgentBudget, AgentOutcome, OutcomeStatus, TaskContract
from .parsing import extract_json_value

logger = logging.getLogger(__name__)


NODE_SYSTEM_PROMPT = """\
You are a bounded Node Agent in a hierarchical multi-agent system.

Your task contract:
{contract}

Sibling Nodes you may contact:
{sibling_roster}

Rules:
1. Work only inside the contract. You cannot create agents or change the plan.
2. Use business tools when they materially advance the task.
3. A newly noticed issue is a DISCOVERY, not a request to create a new Head.
4. Ask a sibling only a concrete question that unblocks your contract. Never wait idly
   for a reply; continue with the best available information.
5. Peer communication and discoveries have hard budgets enforced by the runtime.
6. When you have a result, stop calling tools and return one JSON object:
   {{"status":"completed|partial|blocked", "summary":"...", \
"evidence":["..."], "unresolved":["..."]}}
"""


def _control_tool_schemas(has_siblings: bool) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "framework_report_discovery",
                "description": (
                    "Report an important out-of-scope discovery to the Head. "
                    "This does not request or create another agent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "impact": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["description", "impact"],
                },
            },
        }
    ]
    if has_siblings:
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "framework_ask_sibling",
                    "description": "Ask one sibling Node a concrete, non-blocking question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_agent_id": {"type": "string"},
                            "question": {"type": "string"},
                        },
                        "required": ["target_agent_id", "question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "framework_reply_sibling",
                    "description": "Reply to a sibling request already received in context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "correlation_id": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["correlation_id", "answer"],
                    },
                },
            },
        ])
    return tools


class NodeAgent(BaseAgent):
    """Bounded executor. It may use tools and limited peer requests, but not reorganize work."""

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
        max_turns: int = 12,
        contract: TaskContract | None = None,
        budget: AgentBudget | None = None,
        sibling_roster: dict[str, str] | None = None,
        run_id: str = "",
        journal: Any | None = None,
    ):
        self.contract = contract or TaskContract(role=role, goal=task, scope=task)
        self.task = self.contract.goal
        self.head_id = head_id
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)
        self.skills_reader = skills_reader
        self.sibling_roster = sibling_roster or {}
        self._pending_peer_requests: dict[str, str] = {}
        self._cancel_requested = False
        self._tool_calls_used = 0
        self._last_text = ""

        actual_budget = budget or AgentBudget(
            max_turns=max_turns,
            max_tool_calls=12,
            max_peer_messages=2,
            max_discoveries=1,
            max_children=0,
            max_revision_rounds=0,
            timeout_seconds=300,
        )
        system_prompt = NODE_SYSTEM_PROMPT.format(
            contract=self.contract.to_prompt(),
            sibling_roster=self._format_sibling_roster(),
        )
        super().__init__(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            llm_client=llm_client,
            router=router,
            channel=channel,
            context_manager=context_manager,
            budget=actual_budget,
            task_id=self.contract.task_id,
            run_id=run_id,
            journal=journal,
        )

    def set_sibling_roster(self, roster: dict[str, str]) -> None:
        self.sibling_roster = dict(roster)
        self.system_prompt = NODE_SYSTEM_PROMPT.format(
            contract=self.contract.to_prompt(),
            sibling_roster=self._format_sibling_roster(),
        )
        self.context.set_system_prompt(self.system_prompt)

    def _format_sibling_roster(self) -> str:
        if not self.sibling_roster:
            return "No siblings."
        return "\n".join(
            f"- {agent_id}: {description}"
            for agent_id, description in self.sibling_roster.items()
        )

    async def run(self) -> AgentOutcome:
        self._running = True
        self.start_clock()
        logger.info("NodeAgent %s starting task: %s", self.id, self.task[:100])
        outcome: AgentOutcome | None = None
        await self.record_event("agent_phase_started", {"contract": self.contract.model_dump(mode="json")})

        try:
            business_tools = await self._load_tools()
            control_tools = _control_tool_schemas(bool(self.sibling_roster))
            tools = business_tools + control_tools
            current_input = f"Execute this contract.\n\n{self.contract.to_prompt()}"

            for turn in range(1, self.max_turns + 1):
                if self._cancel_requested or self.remaining_seconds <= 0:
                    break

                await self._process_incoming_messages()
                response = await self.think_with_tools(current_input, tools)
                msg = response.choices[0].message
                self._last_text = msg.content or self._last_text

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        result = await self._execute_tool_call(tool_call)
                        self.add_tool_result(tool_call.id, result)
                    current_input = (
                        "Continue from the tool results. Respect the remaining budgets and "
                        "return the final JSON result as soon as the contract is satisfied."
                    )
                    continue

                if msg.content:
                    outcome = self._outcome_from_text(msg.content)
                    break
                current_input = "Return the best available result now as the required JSON object."

            if outcome is None:
                if self._cancel_requested:
                    status = OutcomeStatus.CANCELLED
                    summary = self._last_text or "Cancelled by parent."
                else:
                    status = OutcomeStatus.PARTIAL
                    summary = self._last_text or "Budget exhausted before a final result was produced."
                outcome = AgentOutcome(
                    agent_id=self.id,
                    task_id=self.contract.task_id,
                    status=status,
                    summary=summary,
                    unresolved=["Node stopped before satisfying the full contract."],
                    metadata=self._usage_metadata(),
                )
        except Exception as exc:
            logger.exception("NodeAgent %s failed", self.id)
            outcome = AgentOutcome(
                agent_id=self.id,
                task_id=self.contract.task_id,
                status=OutcomeStatus.FAILED,
                summary=f"Node failed: {exc}",
                unresolved=[self.contract.goal],
                metadata=self._usage_metadata(),
            )

        self._running = False
        outcome.metadata.update(self._usage_metadata())
        await self.send_message(
            self.head_id,
            MessageType.REPORT,
            outcome.to_message_content(),
        )
        await self.record_event("agent_phase_finished", outcome.to_message_content())
        if self.journal:
            await self.journal.snapshot_agent(
                self,
                {"contract": self.contract.model_dump(mode="json"),
                 "outcome": outcome.model_dump(mode="json")},
            )
        logger.info("NodeAgent %s finished (%s)", self.id, outcome.status.value)
        return outcome

    async def resume(self, guidance: str) -> AgentOutcome:
        """Start a new bounded phase while retaining this Node's prior context."""
        self.context.add_message({
            "role": "user",
            "content": f"[User-approved follow-up through Head] {guidance}",
        })
        self.contract.context[f"follow_up_{uuid.uuid4().hex[:6]}"] = guidance
        self._sent_counts.clear()
        self._tool_calls_used = 0
        self._cancel_requested = False
        await self.record_event("agent_resumed", {"guidance": guidance})
        return await self.run()

    async def _execute_tool_call(self, tool_call: Any) -> str:
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments"})

        if name.startswith("framework_"):
            result = await self._execute_control_tool(name, arguments)
            await self.record_event("control_action", {
                "name": name,
                "arguments": arguments,
                "result": result,
            })
            return json.dumps(result, ensure_ascii=False)

        if self._tool_calls_used >= self.budget.max_tool_calls:
            return json.dumps({"error": "Business tool-call budget exhausted"})
        self._tool_calls_used += 1
        await self.record_event("tool_call", {"name": name, "arguments": arguments})
        result = await self.tool_executor.execute(name, arguments)
        await self.record_event("tool_result", {
            "name": name,
            "result": str(result)[:4000],
        })
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    async def _execute_control_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "framework_report_discovery":
            if self.sent_count(MessageType.DISCOVERY) >= self.budget.max_discoveries:
                return {"error": "Discovery budget exhausted", "action": "continue_current_task"}
            await self.send_message(
                self.head_id,
                MessageType.DISCOVERY,
                {
                    "description": str(arguments.get("description", "")),
                    "impact": str(arguments.get("impact", "")),
                    "evidence": str(arguments.get("evidence", "")),
                    "source_agent": self.id,
                },
            )
            return {"status": "recorded", "action": "continue_current_task"}

        peer_count = (
            self.sent_count(MessageType.PEER_REQUEST)
            + self.sent_count(MessageType.PEER_RESPONSE)
        )
        if peer_count >= self.budget.max_peer_messages:
            return {"error": "Peer-message budget exhausted", "action": "continue_without_peer"}

        if name == "framework_ask_sibling":
            target = str(arguments.get("target_agent_id", ""))
            if target not in self.sibling_roster or target not in self.router.get_peers(self.id):
                return {"error": "Target is not an available sibling"}
            correlation_id = uuid.uuid4().hex[:12]
            sent = await self.send_message(
                target,
                MessageType.PEER_REQUEST,
                {"question": str(arguments.get("question", "")), "from": self.id},
                correlation_id=correlation_id,
            )
            return {
                "status": "sent" if sent else "blocked",
                "correlation_id": correlation_id,
                "action": "continue_without_waiting",
            }

        if name == "framework_reply_sibling":
            correlation_id = str(arguments.get("correlation_id", ""))
            target = self._pending_peer_requests.pop(correlation_id, "")
            if not target:
                return {"error": "Unknown or already answered correlation_id"}
            sent = await self.send_message(
                target,
                MessageType.PEER_RESPONSE,
                {"answer": str(arguments.get("answer", "")), "from": self.id},
                correlation_id=correlation_id,
            )
            return {"status": "sent" if sent else "blocked"}

        return {"error": f"Unknown framework tool: {name}"}

    async def _load_tools(self) -> list[dict[str, Any]]:
        if self.skills_reader:
            try:
                return await self.skills_reader.find_tools(self.contract.goal)
            except Exception as exc:
                logger.warning("Skills reader failed: %s, using all tools", exc)
        return self.tool_registry.get_openai_schemas()

    async def _process_incoming_messages(self) -> None:
        for msg in self.drain_messages():
            if msg.msg_type == MessageType.PEER_REQUEST:
                self._pending_peer_requests[msg.correlation_id] = msg.sender_id
                self.context.add_message({
                    "role": "user",
                    "content": (
                        f"[Sibling request from {msg.sender_id}; correlation_id="
                        f"{msg.correlation_id}] {msg.content.get('question', '')}"
                    ),
                })
            elif msg.msg_type == MessageType.PEER_RESPONSE:
                self.context.add_message({
                    "role": "user",
                    "content": (
                        f"[Sibling response from {msg.sender_id}; correlation_id="
                        f"{msg.correlation_id}] {msg.content.get('answer', '')}"
                    ),
                })
            elif msg.msg_type in (MessageType.GUIDANCE, MessageType.CLARIFICATION):
                self.context.add_message({
                    "role": "user",
                    "content": f"[Head guidance] {msg.content.get('text', msg.content)}",
                })
            elif msg.msg_type == MessageType.CANCEL:
                self._cancel_requested = True

    def _outcome_from_text(self, text: str) -> AgentOutcome:
        data = extract_json_value(text, dict) or {}
        status_text = str(data.get("status", "completed")).lower()
        try:
            status = OutcomeStatus(status_text)
        except ValueError:
            status = OutcomeStatus.COMPLETED
        summary = str(data.get("summary") or text).strip()
        return AgentOutcome(
            agent_id=self.id,
            task_id=self.contract.task_id,
            status=status,
            summary=summary,
            evidence=[str(item) for item in data.get("evidence", [])],
            unresolved=[str(item) for item in data.get("unresolved", [])],
            artifacts=data.get("artifacts", {}) if isinstance(data.get("artifacts", {}), dict) else {},
            metadata=self._usage_metadata(),
        )

    def _usage_metadata(self) -> dict[str, Any]:
        return {
            "tool_calls": self._tool_calls_used,
            "peer_messages": (
                self.sent_count(MessageType.PEER_REQUEST)
                + self.sent_count(MessageType.PEER_RESPONSE)
            ),
            "discoveries": self.sent_count(MessageType.DISCOVERY),
        }

    def export_state(self) -> dict[str, Any]:
        state = super().export_state()
        state.update({
            "contract": self.contract.model_dump(mode="json"),
            "usage": self._usage_metadata(),
            "sibling_roster": self.sibling_roster,
            "pending_peer_requests": self._pending_peer_requests,
            "last_text": self._last_text,
        })
        return state
