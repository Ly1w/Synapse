from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..communication.message import Message, MessageType
from ..communication.router import AgentRole, Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient
from ..llm.config import FrameworkLLMConfig, LLMConfig
from ..memory.memory_store import MemoryStore
from ..memory.plan_cache import PlanCache
from ..planning.decomposer import TaskDecomposer
from ..planning.progress import ProgressTracker, TaskProgress, TaskStatus
from ..skills.generator import SkillsGenerator
from ..skills.reader import SkillsReader
from ..tools.registry import ToolRegistry
from .base_agent import BaseAgent
from .head_agent import HeadAgent
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

MASTER_SYSTEM_PROMPT = """\
You are the Master Agent in a hierarchical multi-agent system. \
Users interact only with you.

Your responsibilities:
1. Decompose user requests into sub-tasks and assign Head Agents.
2. Receive and aggregate Head Agent reports.
3. Generate progress reports for the user.
4. Handle escalations from Heads (e.g., create new Heads when needed).

You coordinate but do NOT execute tasks directly. \
Actual work is done by Node Agents under each Head.

{memory_context}
"""


class MasterAgent(BaseAgent):
    """
    Top-level user-facing agent. Decomposes tasks, manages Head agents,
    tracks progress, and aggregates results.

    Master does not execute tasks or call business tools directly, so it
    does not need the skills index in its own context. The skills index is
    generated once and passed down to Node agents via HeadAgents.
    """

    def __init__(
        self,
        llm_config: LLMConfig | FrameworkLLMConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_root: str | None = None,
        cache_dir: str | None = None,
        max_context_tokens: int = 128000,
    ):
        config = llm_config or LLMConfig()
        if isinstance(config, FrameworkLLMConfig):
            self._framework_config = config
            planner_config = config.planner
        else:
            self._framework_config = FrameworkLLMConfig(planner=config)
            planner_config = config

        llm_client = LLMClient(planner_config)
        self._lightweight_client = LLMClient(self._framework_config.get_lightweight())
        self._executor_client = LLMClient(self._framework_config.get_executor())

        self.tool_registry = tool_registry or ToolRegistry()
        self.agent_registry = AgentRegistry()
        self.router = Router()

        master_id = AgentRegistry.generate_id("master")
        channel = self.router.register(master_id, AgentRole.MASTER)

        self.memory_store = MemoryStore("master", memory_root)
        self.plan_cache = PlanCache(self._lightweight_client, cache_dir)
        self.skills_generator = SkillsGenerator(self._lightweight_client)
        self.task_decomposer = TaskDecomposer(llm_client)
        self.progress_tracker = ProgressTracker()

        # Skills index and reader are built after tools are registered.
        self._skills_index: str = ""
        self._skills_reader: SkillsReader | None = None

        self._head_tasks: dict[str, asyncio.Task] = {}
        self._head_results: dict[str, dict[str, Any]] = {}
        self._head_roles: dict[str, str] = {}

        memory_context = "(Memory loaded at runtime)"
        system_prompt = MASTER_SYSTEM_PROMPT.format(memory_context=memory_context)

        super().__init__(
            agent_id=master_id,
            role="master",
            system_prompt=system_prompt,
            llm_client=llm_client,
            router=self.router,
            channel=channel,
            context_manager=ContextManager(max_tokens=max_context_tokens),
        )
        self.agent_registry.register(self, {"role": "master"})

    async def handle_user_request(
        self,
        request: str,
        tools: list[dict[str, Any]] | None = None,
        tool_callables: dict[str, Any] | None = None,
    ) -> str:
        """
        Main entry point. Process a user request end-to-end:
        1. Register tools & generate skills index for Node agents
        2. Check plan cache
        3. Decompose into Head agents
        4. Run all Heads in parallel
        5. Aggregate results
        6. Update cache & memory
        7. Return final result
        """
        logger.info("Master handling request: %s", request[:200])

        # Initialize memory and plan cache
        await self.memory_store.initialize()
        await self.plan_cache.initialize()

        mem = await self.memory_store.read()
        if mem:
            self.context.add_pinned(self.memory_store.get_injection_prompt(mem))

        # 1. Register tools and generate skills index.
        #    The index is NOT injected into Master's context — only Node agents need it.
        if tools:
            self.tool_registry.register_batch(tools, tool_callables)
        self._skills_index = await self._generate_skills_index()
        self._skills_reader = SkillsReader(self._skills_index, self.tool_registry)

        # 2. Check plan cache
        keyword, cached_entry = await self.plan_cache.lookup(request)

        # 3. Decompose — pass tool categories as a hint (no need to inject full index)
        tool_categories = list(self.tool_registry.get_categories().keys())
        if cached_entry:
            plan = await self.task_decomposer.decompose_from_template(
                request,
                cached_entry.template,
                tool_categories=tool_categories,
            )
        else:
            plan = await self.task_decomposer.decompose(
                request,
                tool_categories=tool_categories,
            )

        # 4. Initialize progress tracking
        root_progress = self.progress_tracker.create_root("root", request)

        # 5. Create and launch Head agents
        sub_tasks = plan.get("sub_tasks", [])
        for spec in sub_tasks:
            await self._create_and_launch_head(spec, root_progress)

        # 6. Monitor heads: collect reports and handle escalations
        expected = len(self._head_tasks)
        completed = 0

        while completed < expected:
            msg = await self.receive_message(timeout=3.0)
            if msg is None:
                continue

            if msg.msg_type == MessageType.REPORT:
                agent_id = msg.content.get("agent_id", msg.sender_id)
                self._head_results[agent_id] = msg.content
                completed += 1
                self.progress_tracker.update_status(
                    agent_id, TaskStatus.COMPLETED,
                    msg.content.get("text", "")[:300],
                )
                logger.info("Master: head %s reported (%d/%d)",
                            agent_id, completed, expected)

            elif msg.msg_type == MessageType.ESCALATION:
                new_head = await self._handle_escalation(msg, root_progress)
                if new_head:
                    expected += 1

            elif msg.msg_type == MessageType.PROGRESS:
                self.progress_tracker.update_status(
                    msg.sender_id, TaskStatus.IN_PROGRESS,
                    msg.content.get("text", ""),
                )

        # Wait for all head tasks
        if self._head_tasks:
            await asyncio.gather(*self._head_tasks.values(), return_exceptions=True)

        # 7. Aggregate final results
        final_result = await self._aggregate_final_results(request)

        # 8. Update plan cache and memory
        if keyword:
            execution_log = self.progress_tracker.to_execution_log()
            await self.plan_cache.store(keyword, execution_log)

        await self.memory_store.append(
            f"## Request: {request[:100]}\n"
            f"Strategy: {plan.get('overall_strategy', 'N/A')}\n"
            f"Heads: {len(sub_tasks)}, Result: {final_result[:200]}\n"
        )

        logger.info("Master completed request")
        return final_result

    async def get_progress_report(self) -> str:
        """Generate a formatted progress report for the user."""
        return self.progress_tracker.generate_report()

    async def _generate_skills_index(self) -> str:
        """Generate the lightweight skills index from registered tools."""
        if not self.tool_registry.get_all():
            return ""
        return await self.skills_generator.generate_index(self.tool_registry)

    async def _create_and_launch_head(
        self,
        spec: dict[str, Any],
        parent_progress: TaskProgress,
    ) -> HeadAgent:
        """Create a HeadAgent, register it, and launch as async task."""
        role = spec.get("role", "general")
        task = spec.get("description", spec.get("task", ""))
        head_id = AgentRegistry.generate_id("head")

        peer_roster = self._build_peer_roster(head_id)
        channel = self.router.register(head_id, AgentRole.HEAD, parent_id=self.id)

        head = HeadAgent(
            agent_id=head_id,
            role=role,
            task=task,
            master_id=self.id,
            llm_client=self._executor_client,
            router=self.router,
            channel=channel,
            tool_registry=self.tool_registry,
            agent_registry=self.agent_registry,
            skills_reader=self._skills_reader,
            peer_roster=peer_roster,
            memory_store=MemoryStore(f"head-{role}"),
        )
        self.agent_registry.register(head, {"role": role, "task": task})
        self._head_roles[head_id] = f"{role}: {task}"

        await self._update_all_peer_rosters()

        task_progress = TaskProgress(
            task_id=head_id,
            description=task,
            status=TaskStatus.IN_PROGRESS,
            assigned_to=head_id,
        )
        parent_progress.add_sub_task(task_progress)

        self._head_tasks[head_id] = asyncio.create_task(head.run())
        logger.info("Master: created head %s (role=%s)", head_id, role)
        return head

    def _build_peer_roster(self, exclude_id: str) -> str:
        """Build a text roster of all active Head agents."""
        lines: list[str] = []
        for hid, role_desc in self._head_roles.items():
            if hid != exclude_id:
                lines.append(f"- {hid}: {role_desc}")
        return "\n".join(lines) if lines else "No peers yet."

    async def _update_all_peer_rosters(self) -> None:
        """Update peer rosters for all existing heads (called when a new head is added)."""
        for hid in self._head_roles:
            head = self.agent_registry.get(hid)
            if head and isinstance(head, HeadAgent):
                head.peer_roster_text = self._build_peer_roster(hid)

    async def _handle_escalation(
        self,
        msg: Message,
        parent_progress: TaskProgress,
    ) -> HeadAgent | None:
        """Handle escalation from a Head: potentially create a new Head."""
        desc = msg.content.get("text", "")
        source = msg.content.get("source_head", msg.sender_id)
        logger.info("Master: escalation from %s: %s", source, desc[:100])

        decision = await self.think(
            f"A Head Agent ({source}) escalated: '{desc}'. "
            f"Current heads: {list(self._head_roles.values())}. "
            f"Should I create a new Head Agent for this? "
            f"Reply YES with a role and task, or NO if an existing head can handle it."
        )

        if "YES" in decision.upper():
            spec = {"role": "escalated", "description": desc}
            return await self._create_and_launch_head(spec, parent_progress)
        return None

    async def _aggregate_final_results(self, original_request: str) -> str:
        """Aggregate all Head reports into a final response."""
        if not self._head_results:
            return "No results available."

        results_text = "\n\n".join(
            f"Head {hid} ({self._head_roles.get(hid, '?')}): "
            f"{res.get('text', str(res))}"
            for hid, res in self._head_results.items()
        )

        final = await self.think(
            f"Aggregate these Head Agent reports into a final response "
            f"for the user.\n\n"
            f"Original request: {original_request}\n\n"
            f"Head reports:\n{results_text}\n\n"
            f"Progress:\n{self.progress_tracker.generate_report()}\n\n"
            f"Provide a clear, comprehensive response to the user."
        )
        return final
