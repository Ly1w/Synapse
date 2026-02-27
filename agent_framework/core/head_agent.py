from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..communication.channel import Channel
from ..communication.message import Message, MessageType
from ..communication.router import AgentRole, Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient
from ..memory.memory_store import MemoryStore
from ..skills.reader import SkillsReader
from ..tools.registry import ToolRegistry
from .base_agent import BaseAgent
from .node_agent import NodeAgent
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

HEAD_SYSTEM_PROMPT = """\
You are a Head Agent (team lead) in a hierarchical multi-agent system.

Your role: {role}
Your sub-task: {task}

Peer Heads and their responsibilities:
{peer_roster}

Rules:
1. Create Node Agents to execute specific parts of your sub-task.
2. Coordinate your Nodes and aggregate their results.
3. You can communicate with peer Head Agents for coordination.
4. If a Node discovers work outside your scope AND outside any peer Head's scope, \
escalate to Master to suggest creating a new Head.
5. If a Node discovers work that belongs to a peer Head, send a PEER_MSG to that Head.
6. When all Nodes complete, aggregate results and REPORT to Master.

Output format for actions:
- To create a node: CREATE_NODE: {{"role": "...", "task": "..."}}
- To message a peer head: PEER(agent_id): {{message}}
- To escalate to master: ESCALATE: {{description}}
- To report to master: REPORT: {{aggregated summary}}

{memory_context}
"""

NODE_DECOMPOSITION_PROMPT = """\
You need to break down this sub-task into specific work items for Node Agents.

Sub-task: {task}

Return a JSON array of node assignments:
[
    {{"role": "specific_role", "task": "detailed task description"}},
    ...
]

Keep each node's task focused and specific. 2-5 nodes is typical.
"""


class HeadAgent(BaseAgent):
    """
    Mid-level coordinator. Creates Node agents, aggregates results,
    coordinates with peer Heads, and reports to Master.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        task: str,
        master_id: str,
        llm_client: LLMClient,
        router: Router,
        channel: Channel,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        skills_content: str = "",
        peer_roster: str = "",
        memory_store: MemoryStore | None = None,
        context_manager: ContextManager | None = None,
        max_turns: int = 40,
    ):
        self.task = task
        self.master_id = master_id
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.skills_content = skills_content
        self.peer_roster_text = peer_roster
        self.memory_store = memory_store
        self._node_tasks: dict[str, asyncio.Task] = {}
        self._node_results: dict[str, dict[str, Any]] = {}
        self._all_nodes_done = asyncio.Event()

        memory_context = ""
        if memory_store:
            memory_context = "(Memory will be loaded at runtime)"

        system_prompt = HEAD_SYSTEM_PROMPT.format(
            role=role,
            task=task,
            peer_roster=peer_roster or "None assigned yet.",
            memory_context=memory_context,
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
        Main Head Agent loop:
        1. Load memory
        2. Decompose sub-task into node assignments
        3. Create and launch Node agents in parallel
        4. Monitor messages (reports, escalations, peer comms)
        5. Aggregate results and report to Master
        """
        self._running = True
        logger.info("HeadAgent %s starting: %s", self.id, self.task[:100])

        # Load memory if available
        if self.memory_store:
            await self.memory_store.initialize()
            mem = await self.memory_store.read()
            if mem:
                self.context.add_pinned(
                    self.memory_store.get_injection_prompt(mem)
                )

        # Decompose into node tasks
        node_specs = await self._decompose_into_nodes()
        logger.info("HeadAgent %s creating %d nodes", self.id, len(node_specs))

        # Create and launch nodes
        skills_reader = None
        if self.skills_content:
            skills_reader = SkillsReader(
                self.skills_content, self.tool_registry, self.llm_client
            )

        for spec in node_specs:
            await self._create_and_launch_node(
                spec["role"], spec["task"], skills_reader
            )

        # Monitor loop: wait for all nodes to finish while handling messages
        expected = len(self._node_tasks)
        completed = 0

        while self._running and completed < expected:
            msg = await self.receive_message(timeout=2.0)
            if msg is None:
                continue

            if msg.msg_type == MessageType.REPORT:
                agent_id = msg.content.get("agent_id", msg.sender_id)
                self._node_results[agent_id] = msg.content
                completed += 1
                logger.info(
                    "HeadAgent %s: node %s reported (%d/%d)",
                    self.id, agent_id, completed, expected,
                )

            elif msg.msg_type == MessageType.ESCALATION:
                await self._handle_escalation(msg)

            elif msg.msg_type == MessageType.PEER_MSG:
                await self._handle_peer_message(msg)

            elif msg.msg_type == MessageType.CLARIFICATION:
                info = msg.content.get("text", str(msg.content))
                self.context.add_message({
                    "role": "user",
                    "content": f"[Master]: {info}",
                })

        # Wait for async tasks to fully finish
        if self._node_tasks:
            await asyncio.gather(*self._node_tasks.values(), return_exceptions=True)

        # Aggregate results
        aggregated = await self._aggregate_results()

        # Report to Master
        await self.send_message(
            self.master_id,
            MessageType.REPORT,
            {
                "text": aggregated,
                "agent_id": self.id,
                "role": self.role,
                "status": "completed",
                "node_count": len(self._node_results),
            },
        )

        # Update memory
        if self.memory_store:
            await self.memory_store.append(
                f"## Task: {self.task[:100]}\nResult: {aggregated[:300]}\n"
            )

        logger.info("HeadAgent %s completed", self.id)
        return {"aggregated": aggregated, "node_results": self._node_results}

    async def _decompose_into_nodes(self) -> list[dict[str, str]]:
        """Use LLM to decompose the sub-task into node assignments."""
        prompt = NODE_DECOMPOSITION_PROMPT.format(task=self.task)
        raw = await self.think(prompt)

        import re
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                specs = json.loads(match.group())
                if isinstance(specs, list) and all(
                    isinstance(s, dict) and "role" in s and "task" in s
                    for s in specs
                ):
                    return specs
            except json.JSONDecodeError:
                pass

        logger.warning("HeadAgent %s: node decomposition parse failed, using single node", self.id)
        return [{"role": "executor", "task": self.task}]

    async def _create_and_launch_node(
        self,
        role: str,
        task: str,
        skills_reader: SkillsReader | None,
    ) -> NodeAgent:
        """Create a NodeAgent, register it, and launch it as an async task."""
        node_id = AgentRegistry.generate_id("node")
        channel = self.router.register(node_id, AgentRole.NODE, parent_id=self.id)

        node = NodeAgent(
            agent_id=node_id,
            role=role,
            task=task,
            head_id=self.id,
            llm_client=self.llm_client,
            router=self.router,
            channel=channel,
            tool_registry=self.tool_registry,
            skills_reader=skills_reader,
        )
        self.agent_registry.register(node, {"role": role, "task": task})
        self._node_tasks[node_id] = asyncio.create_task(node.run())
        logger.info("HeadAgent %s: created node %s (role=%s)", self.id, node_id, role)
        return node

    async def _handle_escalation(self, msg: Message) -> None:
        """Handle escalation from a Node: create new Node or escalate to Master."""
        desc = msg.content.get("text", "")
        logger.info("HeadAgent %s: escalation from %s: %s", self.id, msg.sender_id, desc[:100])

        # Ask LLM whether this is in-scope or needs a new Head
        decision = await self.think(
            f"A Node reported new work: '{desc}'. "
            f"My scope is: '{self.task}'. "
            f"Peer heads handle: {self.peer_roster_text}. "
            f"Should I: (a) create a new Node for it, (b) forward to a peer Head, "
            f"or (c) escalate to Master to suggest a new Head? "
            f"Reply with exactly one of: NEW_NODE, PEER(agent_id), or ESCALATE_MASTER."
        )

        if "NEW_NODE" in decision:
            skills_reader = None
            if self.skills_content:
                skills_reader = SkillsReader(
                    self.skills_content, self.tool_registry, self.llm_client
                )
            await self._create_and_launch_node("escalated_task", desc, skills_reader)
        elif "PEER(" in decision:
            import re
            peer_match = re.search(r"PEER\((\S+)\)", decision)
            if peer_match:
                peer_id = peer_match.group(1)
                await self.send_message(
                    peer_id, MessageType.PEER_MSG,
                    {"text": f"Forwarded work from my Node: {desc}", "from": self.id},
                )
        else:
            await self.send_message(
                self.master_id, MessageType.ESCALATION,
                {
                    "text": desc,
                    "source_head": self.id,
                    "suggestion": "Create a new Head Agent for this area.",
                },
            )

    async def _handle_peer_message(self, msg: Message) -> None:
        """Handle messages from peer Head agents."""
        info = msg.content.get("text", str(msg.content))
        self.context.add_message({
            "role": "user",
            "content": f"[Peer Head {msg.sender_id}]: {info}",
        })
        logger.info("HeadAgent %s: peer message from %s", self.id, msg.sender_id)

    async def _aggregate_results(self) -> str:
        """Use LLM to aggregate all node results into a summary."""
        if not self._node_results:
            return "No nodes completed."

        results_text = "\n\n".join(
            f"Node {aid} ({res.get('role', '?')}): {res.get('text', str(res))}"
            for aid, res in self._node_results.items()
        )

        summary = await self.think(
            f"Aggregate these node results into a coherent summary for the Master Agent.\n\n"
            f"My sub-task was: {self.task}\n\n"
            f"Node results:\n{results_text}\n\n"
            f"Provide a concise but complete summary of what was accomplished."
        )
        return summary
