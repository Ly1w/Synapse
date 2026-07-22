from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Coroutine

from ..communication.message import Message, MessageType
from ..communication.router import AgentRole, Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient
from ..llm.config import FrameworkLLMConfig, LLMConfig
from ..memory.memory_store import MemoryStore
from ..memory.plan_cache import PlanCache
from ..planning.decomposer import TaskDecomposer
from ..planning.progress import ProgressTracker, TaskProgress, TaskStatus
from ..runtime.run import RunJournal, RunStatus, redact_state
from ..system_prompts import build_master_system_prompt
from ..tools.builtin import register_builtin_tools
from ..tools.executor import ToolExecutor
from ..tools.permissions import ApprovalContext, PermissionManager, PermissionMode
from ..tools.registry import ToolRegistry
from ..tools.mcp_provider import MCPServerConfig, MCPToolProvider
from .base_agent import BaseAgent
from .contracts import AgentBudget, AgentOutcome, OutcomeStatus, TaskContract
from .head_agent import HeadAgent
from .parsing import extract_json_value
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


def _master_finish_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "framework_commit_response",
            "description": (
                "Commit the complete user-facing answer and end this Master execution "
                "phase. Do not call this for plans, progress notes, or text describing "
                "work that remains to be done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Complete final response addressed to the user.",
                    },
                },
                "required": ["response"],
            },
        },
    }


class MasterAgent(BaseAgent):
    """Global owner for interruptible, retained hierarchical runs."""

    def __init__(
        self,
        llm_config: LLMConfig | FrameworkLLMConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_root: str | None = None,
        cache_dir: str | None = None,
        run_root: str | None = None,
        max_context_tokens: int = 128000,
        budget: AgentBudget | None = None,
        head_budget: AgentBudget | None = None,
        node_budget: AgentBudget | None = None,
        retained_run_limit: int | None = 8,
        workspace_root: str | None = None,
        permission_mode: PermissionMode | str = PermissionMode.AUTO,
    ):
        config = llm_config or LLMConfig()
        if isinstance(config, FrameworkLLMConfig):
            self._framework_config = config
        else:
            self._framework_config = FrameworkLLMConfig(planner=config)

        planner_client = LLMClient(self._framework_config.planner)
        self._lightweight_client = LLMClient(self._framework_config.get_lightweight())
        self._executor_client = LLMClient(self._framework_config.get_executor())
        self.tool_registry = tool_registry or ToolRegistry()
        self.workspace_root = Path(workspace_root or Path.cwd()).expanduser().resolve()
        self.permission_manager = PermissionManager(self.workspace_root, permission_mode)
        self._builtin_toolset = register_builtin_tools(
            self.tool_registry,
            self.workspace_root,
        )
        self.tool_executor = ToolExecutor(self.tool_registry, self.permission_manager)
        self.agent_registry = AgentRegistry()
        self.router = Router()

        master_id = AgentRegistry.generate_id("master")
        channel = self.router.register(master_id, AgentRole.MASTER)
        self._memory_root = memory_root
        self._run_root = run_root
        self.retained_run_limit = retained_run_limit
        self.memory_store = MemoryStore("master", memory_root)
        self.plan_cache = PlanCache(self._lightweight_client, cache_dir)
        self.head_budget = head_budget or AgentBudget(
            max_turns=20,
            max_peer_messages=4,
            max_discoveries=2,
            max_children=4,
            max_revision_rounds=1,
            timeout_seconds=600,
        )
        self.node_budget = node_budget or AgentBudget(
            max_turns=10,
            max_peer_messages=2,
            max_discoveries=1,
            max_children=0,
            max_revision_rounds=0,
            timeout_seconds=240,
        )
        master_budget = budget or AgentBudget(
            max_turns=30,
            max_peer_messages=0,
            max_discoveries=2,
            max_children=6,
            # User steering is never consumed as an internal synthesis-repair round.
            max_revision_rounds=0,
            timeout_seconds=1800,
        )
        master_system_prompt = build_master_system_prompt(
            master_budget,
            self.head_budget,
            self.node_budget,
            str(self.workspace_root),
            self.permission_manager.mode.value,
            [item.name for item in self.tool_registry.get_all() if item.source == "builtin"],
            self._builtin_toolset.skill_catalog.model_inventory(),
        )
        self.task_decomposer = TaskDecomposer(
            planner_client,
            system_prompt=master_system_prompt,
        )

        super().__init__(
            agent_id=master_id,
            role="master",
            system_prompt=master_system_prompt,
            llm_client=planner_client,
            router=self.router,
            channel=channel,
            context_manager=ContextManager(max_tokens=max_context_tokens),
            budget=master_budget,
        )
        self.agent_registry.register(self, {"role": "master"})

        self._run_journals: dict[str, RunJournal] = {}
        self._run_tasks: dict[str, asyncio.Task[str]] = {}
        self._run_updates: dict[str, asyncio.Queue[tuple[int, str]]] = {}
        self._run_locks: dict[str, asyncio.Lock] = {}
        self._run_heads: dict[str, dict[str, HeadAgent]] = defaultdict(dict)
        self._run_head_roles: dict[str, dict[str, str]] = defaultdict(dict)
        self._run_head_outcomes: dict[str, dict[str, list[AgentOutcome]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._run_active_head_tasks: dict[
            str, dict[str, asyncio.Task[AgentOutcome]]
        ] = defaultdict(dict)
        self._run_progress: dict[str, ProgressTracker] = {}
        self._run_discoveries: dict[str, set[str]] = defaultdict(set)
        self._run_required_revisions: dict[str, dict[str, dict[int, str]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._run_revision_attempts: dict[str, dict[tuple[str, int], int]] = defaultdict(dict)
        self._master_context_states: dict[str, dict[str, Any]] = {}
        self._run_deferred_contracts: dict[str, list[TaskContract]] = defaultdict(list)
        self._run_master_tool_call_counts: dict[str, int] = defaultdict(int)
        self._mcp_providers: list[MCPToolProvider] = []
        self._cold_run_ids: set[str] = set()
        self._load_persisted_runs()

    async def handle_user_request(
        self,
        request: str,
        tools: list[dict[str, Any]] | None = None,
        tool_callables: dict[str, Any] | None = None,
    ) -> str:
        """Compatibility wrapper: start an interruptible run and await its checkpoint."""
        run_id = await self.start_request(request, tools, tool_callables)
        return await self.wait_for_run(run_id)

    async def start_request(
        self,
        request: str,
        tools: list[dict[str, Any]] | None = None,
        tool_callables: dict[str, Any] | None = None,
    ) -> str:
        """Start work in the background and return a run id that can be steered."""
        if any(not task.done() for task in self._run_tasks.values()):
            raise RuntimeError("This Master already has an active run; steer or await it first.")
        # Skill metadata is cheap to rediscover and may have changed since the last Run.
        self._refresh_master_system_prompt()
        await self._enforce_retention_limit()
        journal = RunJournal(request, root_dir=self._run_root)
        await journal.initialize()
        await journal.set_status(RunStatus.RUNNING)
        run_id = journal.run_id
        self._run_journals[run_id] = journal
        self._run_updates[run_id] = asyncio.Queue()
        self._run_locks[run_id] = asyncio.Lock()
        self._run_tasks[run_id] = asyncio.create_task(
            self._execute_new_run(run_id, request, tools, tool_callables)
        )
        return run_id

    async def wait_for_run(self, run_id: str) -> str:
        task = self._run_tasks.get(run_id)
        if not task:
            raise KeyError(f"Unknown run: {run_id}")
        return await asyncio.shield(task)

    async def steer(self, run_id: str, user_update: str) -> int:
        """Apply a user requirement to a running or completed-retained run."""
        journal = self._run_journals.get(run_id)
        if not journal:
            raise KeyError(f"Unknown run: {run_id}")
        if run_id in self._cold_run_ids:
            raise RuntimeError(
                "This Run was loaded as a retained snapshot after restart and is "
                "inspectable but not resumable. Start a new Run to continue its work."
            )
        lock = self._run_locks[run_id]
        async with lock:
            status = journal.manifest.status
            if status == RunStatus.ARCHIVED:
                raise RuntimeError("Archived runs cannot be steered.")
            revision = await journal.add_requirement(user_update)
            await self._run_updates[run_id].put((revision, user_update))
            if status in {
                RunStatus.COMPLETED_RETAINED,
                RunStatus.FAILED_RETAINED,
                RunStatus.CANCELLED_RETAINED,
            }:
                if any(not task.done() for key, task in self._run_tasks.items() if key != run_id):
                    raise RuntimeError("Another run is active; wait before resuming this run.")
                await journal.set_status(RunStatus.RUNNING)
                prior_task = self._run_tasks.get(run_id)
                if prior_task and not prior_task.done():
                    self._run_tasks[run_id] = asyncio.create_task(
                        self._resume_after_prior(run_id, prior_task)
                    )
                else:
                    self._run_tasks[run_id] = asyncio.create_task(self._resume_run(run_id))
            return revision

    async def cancel_run(self, run_id: str) -> None:
        journal = self._run_journals.get(run_id)
        task = self._run_tasks.get(run_id)
        if not journal or not task:
            raise KeyError(f"Unknown run: {run_id}")
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        active = self._run_active_head_tasks.get(run_id, {})
        items = list(active.items())
        for _, head_task in items:
            if not head_task.done():
                head_task.cancel()
        results = await asyncio.gather(
            *(head_task for _, head_task in items),
            return_exceptions=True,
        )
        for (head_id, _), result in zip(items, results):
            if isinstance(result, AgentOutcome):
                self._run_head_outcomes[run_id][head_id].append(result)
        active.clear()
        await self._retain_master_state(run_id, "Run cancelled by user.")
        await journal.set_status(RunStatus.CANCELLED_RETAINED)

    async def archive_run(self, run_id: str) -> None:
        """Explicitly release live agents after their snapshots are safely on disk."""
        journal = self._run_journals.get(run_id)
        if not journal:
            raise KeyError(f"Unknown run: {run_id}")
        task = self._run_tasks.get(run_id)
        if task and not task.done():
            raise RuntimeError("Cancel or await the run before archiving it.")
        for head_id in list(self._run_heads.get(run_id, {})):
            self.router.unregister_subtree(head_id)
        for agent_id in list(self.agent_registry.get_all_ids()):
            if self.agent_registry.get_metadata(agent_id).get("run_id") == run_id:
                self.agent_registry.unregister(agent_id)
        self._run_heads.pop(run_id, None)
        await journal.set_status(RunStatus.ARCHIVED)

    async def get_run_state(self, run_id: str) -> dict[str, Any]:
        journal = self._run_journals.get(run_id)
        if not journal:
            raise KeyError(f"Unknown run: {run_id}")
        state = await journal.read_state()
        interrupted = run_id in self._cold_run_ids and state["status"] == RunStatus.RUNNING.value
        if interrupted:
            state["status"] = RunStatus.FAILED_RETAINED.value
        state["storage_path"] = str(journal.run_dir)
        state["live_agents"] = [
            agent_id for agent_id in journal.manifest.agent_ids
            if self.agent_registry.get(agent_id) is not None
        ]
        state["progress"] = await self.get_progress_report(run_id)
        state["topology"] = await self._build_run_topology(run_id)
        state["topology"]["master"]["status"] = state["status"]
        state["resumable"] = (
            run_id not in self._cold_run_ids
            and state["status"] != RunStatus.ARCHIVED.value
        )
        state["retention"] = "snapshot" if run_id in self._cold_run_ids else "live"
        state["interrupted"] = interrupted
        return state

    async def list_run_states(self) -> list[dict[str, Any]]:
        states = [await self.get_run_state(run_id) for run_id in self._run_journals]
        states.sort(key=lambda item: float(item.get("created_at", 0)), reverse=True)
        return states

    def get_runtime_info(self) -> dict[str, str]:
        """Return non-secret model routing metadata for control-plane clients."""
        return {
            "planner_model": self._framework_config.planner.model,
            "executor_model": self._framework_config.get_executor().model,
            "lightweight_model": self._framework_config.get_lightweight().model,
            "base_url": self._framework_config.planner.base_url,
            "workspace_root": str(self.workspace_root),
        }

    def get_tool_info(self) -> dict[str, Any]:
        tools = self.tool_registry.get_all()
        return {
            "builtin": [item.name for item in tools if item.source == "builtin"],
            "mcp": [item.name for item in tools if item.source.startswith("mcp:")],
            "custom": [
                item.name for item in tools
                if item.source != "builtin" and not item.source.startswith("mcp:")
            ],
        }

    def get_permission_state(self) -> dict[str, Any]:
        return self.permission_manager.get_state()

    def runtime_state(self) -> dict[str, Any]:
        state = super().runtime_state()
        run_id = self.run_id
        journal = self._run_journals.get(run_id) if run_id else None
        heads = self._run_heads.get(run_id, {}) if run_id else {}
        active = self._run_active_head_tasks.get(run_id, {}) if run_id else {}
        outcomes = self._run_head_outcomes.get(run_id, {}) if run_id else {}
        manifest = journal.manifest if journal else None
        tool_items = self.tool_registry.get_all()
        state.update({
            "role_position": {
                "level": "Master",
                "responsibility": (
                    "Own the cumulative user request, choose direct or hierarchical "
                    "execution, integrate evidence, and commit user-facing checkpoints"
                ),
                "parent": "user",
                "may_create": "Head only",
                "must_not": [
                    "become a passive dispatcher",
                    "create Nodes directly",
                    "commit progress narration as a final response",
                ],
            },
            "current_run": {
                "status": manifest.status.value if manifest else None,
                "revision": manifest.revision if manifest else 0,
                "cumulative_requirement_count": len(manifest.requirements) if manifest else 0,
                "checkpoint_count": len(manifest.checkpoints) if manifest else 0,
                "queued_user_revisions": (
                    self._run_updates[run_id].qsize()
                    if run_id in self._run_updates else 0
                ),
            },
            "current_topology": {
                "head_ids": list(heads),
                "active_head_ids": list(active),
                "settled_head_phases": sum(len(items) for items in outcomes.values()),
                "head_slots_used": len(heads),
                "head_slots_max": self.budget.max_children,
                "head_slots_remaining": max(0, self.budget.max_children - len(heads)),
                "inbox_pending": self.channel.pending,
            },
            "remaining_coordination": {
                "head_discoveries_seen": len(
                    self._run_discoveries.get(run_id, set())
                ) if run_id else 0,
                "head_discoveries_max": self.budget.max_discoveries,
                "head_discoveries_remaining": max(
                    0,
                    self.budget.max_discoveries
                    - len(self._run_discoveries.get(run_id, set())),
                ) if run_id else self.budget.max_discoveries,
                "peer_messages_max": 0,
                "peer_messages_remaining": 0,
                "user_steering_limit": None,
            },
            "current_execution": {
                "master_business_tool_calls_observed_for_run": (
                    self._run_master_tool_call_counts.get(run_id, 0) if run_id else 0
                ),
                "business_tool_call_limit": None,
                "business_tool_stop_condition": (
                    "phase timeout, per-tool timeout, cancellation, or convergence"
                ),
                "deferred_head_contracts": len(
                    self._run_deferred_contracts.get(run_id, [])
                ) if run_id else 0,
            },
            "environment": {
                "workspace_root": str(self.workspace_root),
                "permission": self.permission_manager.get_state(),
                "builtin_tool_count": sum(
                    1 for item in tool_items if item.source == "builtin"
                ),
                "user_extension_tool_count": sum(
                    1 for item in tool_items if item.source != "builtin"
                ),
            },
        })
        return state

    def set_permission_mode(self, mode: PermissionMode | str) -> dict[str, Any]:
        self.permission_manager.set_mode(mode)
        self._refresh_master_system_prompt()
        for heads in self._run_heads.values():
            for head in heads.values():
                head.refresh_system_prompt()
        return self.permission_manager.get_state()

    def _refresh_master_system_prompt(self) -> None:
        prompt = build_master_system_prompt(
            self.budget,
            self.head_budget,
            self.node_budget,
            str(self.workspace_root),
            self.permission_manager.mode.value,
            [item.name for item in self.tool_registry.get_all() if item.source == "builtin"],
            self._builtin_toolset.skill_catalog.model_inventory(),
        )
        self.system_prompt = prompt
        self.context.set_system_prompt(prompt)
        self.task_decomposer.system_prompt = prompt

    def list_pending_approvals(self, run_id: str | None = None) -> list[dict[str, Any]]:
        return self.permission_manager.list_pending(run_id)

    async def resolve_approval(self, approval_id: str, allow: bool) -> dict[str, Any]:
        return await self.permission_manager.resolve(approval_id, allow)

    def get_mcp_status(self) -> list[dict[str, Any]]:
        return [provider.status() for provider in self._mcp_providers]

    async def connect_mcp_server(self, config: MCPServerConfig) -> list[str]:
        """Connect one user-requested MCP server under an isolated tool namespace."""
        if any(provider.config.name == config.name for provider in self._mcp_providers):
            raise ValueError(f"MCP server is already connected: {config.name}")
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", config.name).strip("_") or "server"
        config = replace(config, tool_prefix=f"mcp__{safe_name[:24]}__")
        provider = MCPToolProvider(config)
        names = await provider.connect(self.tool_registry)
        self._mcp_providers.append(provider)
        return names

    async def disconnect_mcp_server(self, name: str) -> None:
        provider = next(
            (item for item in self._mcp_providers if item.config.name == name),
            None,
        )
        if provider is None:
            raise KeyError(f"Unknown MCP server: {name}")
        self._mcp_providers.remove(provider)
        await provider.close()

    async def close_mcp_servers(self) -> None:
        providers = list(reversed(self._mcp_providers))
        self._mcp_providers.clear()
        await asyncio.gather(
            *(provider.close() for provider in providers),
            return_exceptions=True,
        )

    async def get_run_events(
        self,
        run_id: str,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        journal = self._run_journals.get(run_id)
        if not journal:
            raise KeyError(f"Unknown run: {run_id}")
        return await journal.read_events(after_sequence, limit)

    async def get_agent_snapshot(
        self,
        run_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        journal = self._run_journals.get(run_id)
        if not journal:
            raise KeyError(f"Unknown run: {run_id}")
        agent = self.agent_registry.get(agent_id)
        is_current_master = (
            agent_id == self.id
            and self.run_id == run_id
            and journal.manifest.status != RunStatus.ARCHIVED
        )
        is_run_agent = self.agent_registry.get_metadata(agent_id).get("run_id") == run_id
        if agent is not None and (is_current_master or is_run_agent):
            return redact_state(agent.export_state())
        return await journal.read_agent_snapshot(agent_id)

    async def get_progress_report(self, run_id: str | None = None) -> str:
        if run_id is None:
            if not self._run_progress:
                return "No tasks tracked."
            run_id = next(reversed(self._run_progress))
        tracker = self._run_progress.get(run_id)
        return tracker.generate_report() if tracker else "No tasks tracked."

    async def _execute_new_run(
        self,
        run_id: str,
        request: str,
        tools: list[dict[str, Any]] | None,
        tool_callables: dict[str, Any] | None,
    ) -> str:
        journal = self._run_journals[run_id]
        try:
            await self._activate_master_context(run_id, fresh=True)
            await journal.register_agent(self.id, "master")
            await self.record_event("master_step_started", {
                "step": "planning",
                "message": "Master is deciding whether this request needs delegation.",
            })
            await self.memory_store.initialize()
            memory = await self.memory_store.read()
            if memory:
                self.context.add_pinned(self.memory_store.get_injection_prompt(memory))

            if tools:
                self.tool_registry.register_batch(tools, tool_callables)

            tool_categories = list(self.tool_registry.get_categories())
            plan = await self.task_decomposer.decompose(
                request,
                tool_categories,
                runtime_awareness=self.begin_llm_turn_awareness,
            )

            pending = self._contracts_from_plan(plan, request)
            await self.record_event("master_step_finished", {
                "step": "planning",
                "head_count": len(pending),
                "direct": not pending,
            })
            if not pending:
                direct_response = str(plan.get("direct_response") or "").strip()
                if direct_response:
                    return await self._commit_direct_response(
                        run_id,
                        request,
                        direct_response,
                        reason="master_direct",
                    )
                direct_response = await self._execute_master_direct(
                    run_id,
                    request,
                    str(plan.get("direct_instruction") or ""),
                )
                return await self._commit_direct_response(
                    run_id,
                    request,
                    direct_response,
                    reason="master_direct_execution",
                )

            # Plan templates and tool retrieval are useful only after Master has
            # actually chosen delegation. Simple direct work pays none of this cost.
            await self.plan_cache.initialize()
            keyword, cached_entry = await self.plan_cache.lookup(request)
            if cached_entry:
                adapted = await self.task_decomposer.decompose_from_template(
                    request,
                    cached_entry.template,
                    tool_categories,
                    runtime_awareness=self.begin_llm_turn_awareness,
                )
                adapted_pending = self._contracts_from_plan(adapted, request)
                if adapted.get("mode") == "hierarchical" and adapted_pending:
                    plan = adapted
                    pending = adapted_pending

            tracker = ProgressTracker()
            root_progress = tracker.create_root(run_id, request)
            root_progress.mark_in_progress()
            self._run_progress[run_id] = tracker
            active: dict[str, asyncio.Task[AgentOutcome]] = {}
            self._run_active_head_tasks[run_id] = active

            final = await self._drive_and_commit(
                run_id, request, pending, active, root_progress
            )
            outcomes = self._latest_outcomes(run_id)
            if outcomes and all(item.successful for item in outcomes):
                await self._post_run_learning(
                    keyword,
                    tracker.to_execution_log(),
                    request,
                    str(plan.get("overall_strategy", "N/A")),
                    final,
                )
            return final
        except asyncio.CancelledError:
            await journal.set_status(RunStatus.CANCELLED_RETAINED)
            raise
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            result = f"Run failed, but its state was retained: {exc}"
            await self._retain_master_state(run_id, result)
            await journal.set_status(RunStatus.FAILED_RETAINED, error=result)
            await journal.record(
                "run_failed",
                source=self.id,
                task_id=run_id,
                payload={"error": result, "revision": journal.manifest.revision},
            )
            return result

    async def _resume_run(self, run_id: str) -> str:
        self._refresh_master_system_prompt()
        journal = self._run_journals[run_id]
        try:
            await self._activate_master_context(run_id, fresh=False)
            if not self._run_heads.get(run_id):
                return await self._resume_master_only_run(run_id)
            tracker = self._run_progress.get(run_id) or ProgressTracker()
            if tracker.root is None:
                tracker.create_root(run_id, journal.manifest.user_request).mark_in_progress()
            self._run_progress[run_id] = tracker
            return await self._drive_and_commit(
                run_id,
                journal.manifest.user_request,
                pending={},
                active=self._new_active_task_map(run_id),
                root_progress=tracker.root,
            )
        except asyncio.CancelledError:
            await journal.set_status(RunStatus.CANCELLED_RETAINED)
            raise
        except Exception as exc:
            logger.exception("Retained run %s failed to resume", run_id)
            result = f"Run revision failed, but prior state remains retained: {exc}"
            await self._retain_master_state(run_id, result)
            await journal.set_status(RunStatus.FAILED_RETAINED, error=result)
            await journal.record(
                "run_failed",
                source=self.id,
                task_id=run_id,
                payload={"error": result, "revision": journal.manifest.revision},
            )
            return result

    async def _resume_after_prior(
        self,
        run_id: str,
        prior_task: asyncio.Task[str],
    ) -> str:
        await asyncio.shield(prior_task)
        return await self._resume_run(run_id)

    async def _drive_and_commit(
        self,
        run_id: str,
        original_request: str,
        pending: dict[str, TaskContract],
        active: dict[str, asyncio.Task[AgentOutcome]],
        root_progress: TaskProgress,
    ) -> str:
        journal = self._run_journals[run_id]
        while True:
            await self._drive_scheduler(run_id, pending, active, root_progress)
            candidate = await self._aggregate_final_results(run_id, original_request)

            # Committing and accepting a new steering event are serialized.  An update
            # either changes this checkpoint or starts a retained revision afterwards.
            lock = self._run_locks[run_id]
            async with lock:
                if not self._run_updates[run_id].empty():
                    continue
                root_progress.mark_completed(candidate[:300])
                await self._retain_master_state(run_id, candidate)
                checkpoint = await journal.commit_checkpoint(
                    candidate,
                    reason="hierarchical_synthesis",
                )
                await journal.record(
                    "run_checkpoint_committed",
                    source=self.id,
                    task_id=run_id,
                    payload={
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "revision": checkpoint.revision,
                        "response_chars": len(candidate),
                    },
                )
                return candidate

    async def _drive_scheduler(
        self,
        run_id: str,
        pending: dict[str, TaskContract],
        active: dict[str, asyncio.Task[AgentOutcome]],
        root_progress: TaskProgress,
    ) -> None:
        self._phase = "coordinating_heads"
        while pending or active or not self._run_updates[run_id].empty():
            if self.remaining_seconds <= 0:
                self._run_deferred_contracts[run_id].extend(pending.values())
                pending.clear()
                items = list(active.items())
                for _, task in items:
                    if not task.done():
                        task.cancel()
                results = await asyncio.gather(
                    *(task for _, task in items),
                    return_exceptions=True,
                )
                for (head_id, _), result in zip(items, results):
                    if isinstance(result, AgentOutcome):
                        self._run_head_outcomes[run_id][head_id].append(result)
                        self._run_progress[run_id].update_status(
                            head_id, TaskStatus.FAILED, result.summary[:300]
                        )
                active.clear()
                await self.record_event("master_budget_exhausted", {
                    "timeout_seconds": self.budget.timeout_seconds,
                })
                return
            await self._apply_queued_updates(run_id, pending, active, root_progress)
            latest_by_role = self._latest_outcomes_by_role(run_id)
            ready = [
                role for role, contract in pending.items()
                if set(contract.dependencies).issubset(latest_by_role)
            ]
            available = max(0, self.budget.max_children - len(self._run_heads[run_id]))
            for role in ready[:available]:
                contract = pending.pop(role)
                contract.context["dependency_outcomes"] = {
                    dep: latest_by_role[dep].model_dump(mode="json")
                    for dep in contract.dependencies
                }
                head = await self._create_head(run_id, contract, root_progress)
                active[head.id] = asyncio.create_task(
                    self._run_head_guarded(head, head.run())
                )

            if pending and not active and not ready:
                # A malformed dependency cycle must degrade explicitly, never deadlock.
                role, contract = pending.popitem()
                contract.context["dependency_warning"] = (
                    f"Unresolved or cyclic dependencies: {contract.dependencies}"
                )
                head = await self._create_head(run_id, contract, root_progress)
                active[head.id] = asyncio.create_task(
                    self._run_head_guarded(head, head.run())
                )

            if pending and not active and available == 0:
                self._run_deferred_contracts[run_id].extend(pending.values())
                await self.record_event("head_budget_exhausted", {
                    "deferred_contracts": [
                        contract.model_dump(mode="json") for contract in pending.values()
                    ]
                })
                pending.clear()

            if active:
                message = await self.receive_message(timeout=0.25)
                if message:
                    await self._handle_master_message(
                        run_id, message, pending, active, root_progress
                    )
            elif pending:
                await asyncio.sleep(0)

            for head_id, task in list(active.items()):
                if not task.done():
                    continue
                head = self._run_heads[run_id][head_id]
                try:
                    outcome = task.result()
                except Exception as exc:
                    outcome = AgentOutcome(
                        agent_id=head_id,
                        task_id=head.contract.task_id,
                        status=OutcomeStatus.FAILED,
                        summary=f"Uncaught Head failure: {exc}",
                        unresolved=[head.contract.goal],
                    )
                self._run_head_outcomes[run_id][head_id].append(outcome)
                applied = set(outcome.metadata.get("applied_revisions", []))
                required = self._run_required_revisions[run_id].get(head_id, {})
                missing = [
                    revision for revision in required
                    if revision not in applied
                    and self._run_revision_attempts[run_id].get((head_id, revision), 0) < 1
                ]
                if missing:
                    for revision in missing:
                        key = (head_id, revision)
                        self._run_revision_attempts[run_id][key] = (
                            self._run_revision_attempts[run_id].get(key, 0) + 1
                        )
                    guidance = "\n".join(
                        f"Revision {revision}: {required[revision]}" for revision in missing
                    )
                    active[head_id] = asyncio.create_task(
                        self._run_head_guarded(head, head.resume(guidance, missing))
                    )
                    self._run_progress[run_id].update_status(
                        head_id,
                        TaskStatus.IN_PROGRESS,
                        f"Applying revisions {missing}",
                    )
                    continue
                status = TaskStatus.COMPLETED if outcome.status in {
                    OutcomeStatus.COMPLETED, OutcomeStatus.PARTIAL
                } else TaskStatus.FAILED
                self._run_progress[run_id].update_status(head_id, status, outcome.summary[:300])
                del active[head_id]

    async def _run_head_guarded(
        self,
        head: HeadAgent,
        coroutine: Coroutine[Any, Any, AgentOutcome],
    ) -> AgentOutcome:
        try:
            return await asyncio.wait_for(coroutine, timeout=head.budget.timeout_seconds)
        except asyncio.TimeoutError:
            head.stop()
            outcome = AgentOutcome(
                agent_id=head.id,
                task_id=head.contract.task_id,
                status=OutcomeStatus.PARTIAL,
                summary="Head timed out; all agent state was retained.",
                unresolved=[head.contract.goal],
                metadata={"timeout_seconds": head.budget.timeout_seconds},
            )
        except asyncio.CancelledError:
            head.stop()
            outcome = AgentOutcome(
                agent_id=head.id,
                task_id=head.contract.task_id,
                status=OutcomeStatus.CANCELLED,
                summary="Head cancelled by Master.",
                unresolved=[head.contract.goal],
            )
        except Exception as exc:
            logger.exception("Head task %s crashed", head.id)
            outcome = AgentOutcome(
                agent_id=head.id,
                task_id=head.contract.task_id,
                status=OutcomeStatus.FAILED,
                summary=f"Head crashed: {exc}",
                unresolved=[head.contract.goal],
            )
        if head.journal:
            await head.journal.snapshot_agent(
                head,
                {
                    "contract": head.contract.model_dump(mode="json"),
                    "outcome": outcome.model_dump(mode="json"),
                },
            )
        return outcome

    async def _apply_queued_updates(
        self,
        run_id: str,
        pending: dict[str, TaskContract],
        active: dict[str, asyncio.Task[AgentOutcome]],
        root_progress: TaskProgress,
    ) -> None:
        queue = self._run_updates[run_id]
        while not queue.empty():
            revision, update = queue.get_nowait()
            await self._apply_user_update(
                run_id, revision, update, pending, active, root_progress
            )

    async def _apply_user_update(
        self,
        run_id: str,
        revision: int,
        update: str,
        pending: dict[str, TaskContract],
        active: dict[str, asyncio.Task[AgentOutcome]],
        root_progress: TaskProgress,
    ) -> None:
        self._phase = "applying_user_revision"
        roles = self._run_head_roles[run_id]
        self.context.add_message({
            "role": "user",
            "content": f"[User update revision {revision}] {update}",
        })
        raw = await self.think(
            f"A user changed requirements during run {run_id}.\n"
            f"Update: {update}\n"
            f"Heads: {roles}\nPending roles: {list(pending)}\n"
            "Return JSON with action=master_only|broadcast|target|add_head, "
            "target_roles, guidance, and optional new_head contract. Preserve completed "
            "work unless the update invalidates it."
        )
        decision = extract_json_value(raw, dict) or {
            "action": "broadcast",
            "target_roles": list(roles.values()),
            "guidance": update,
        }
        action = str(decision.get("action", "broadcast"))
        guidance = str(decision.get("guidance") or update)
        target_roles = decision.get("target_roles", [])
        if not isinstance(target_roles, list):
            target_roles = []
        if action == "broadcast":
            target_ids = list(self._run_heads[run_id])
        else:
            wanted = {str(role) for role in target_roles}
            target_ids = [
                head_id for head_id, role in roles.items()
                if role in wanted or head_id in wanted
            ]

        if action != "master_only":
            for contract in pending.values():
                if action == "broadcast" or contract.role in target_roles:
                    contract.context[f"user_revision_{revision}"] = guidance

            for head_id in target_ids:
                head = self._run_heads[run_id][head_id]
                self._run_required_revisions[run_id][head_id][revision] = guidance
                if head_id in active:
                    await self.send_message(
                        head_id,
                        MessageType.GUIDANCE,
                        {"text": guidance, "revision": revision},
                    )
                else:
                    active[head_id] = asyncio.create_task(
                        self._run_head_guarded(head, head.resume(guidance, revision))
                    )
                    self._run_progress[run_id].update_status(
                        head_id, TaskStatus.IN_PROGRESS, f"Applying revision {revision}"
                    )

        if action == "add_head" and len(self._run_heads[run_id]) < self.budget.max_children:
            spec = decision.get("new_head")
            if isinstance(spec, dict):
                contract = self._contract_from_spec(
                    spec,
                    self._run_journals[run_id].manifest.user_request,
                )
                contract.context[f"user_revision_{revision}"] = update
                contract.role = self._unique_role(run_id, contract.role, pending)
                pending[contract.role] = contract
        await self.record_event("user_update_applied", {
            "revision": revision,
            "update": update,
            "decision": decision,
        })
        self._phase = "coordinating_heads"

    async def _handle_master_message(
        self,
        run_id: str,
        message: Message,
        pending: dict[str, TaskContract],
        active: dict[str, asyncio.Task[AgentOutcome]],
        root_progress: TaskProgress,
    ) -> None:
        if message.msg_type == MessageType.DISCOVERY:
            await self._handle_head_discovery(
                run_id, message, pending, active, root_progress
            )
        elif message.msg_type == MessageType.PROGRESS:
            self._run_progress[run_id].update_status(
                message.sender_id,
                TaskStatus.IN_PROGRESS,
                str(message.content.get("text", "")),
            )
        # REPORT does not control completion. The task result is authoritative.

    async def _handle_head_discovery(
        self,
        run_id: str,
        message: Message,
        pending: dict[str, TaskContract],
        active: dict[str, asyncio.Task[AgentOutcome]],
        root_progress: TaskProgress,
    ) -> None:
        self._phase = "triaging_head_discovery"
        description = str(message.content.get("description") or message.content.get("text") or "")
        fingerprint = " ".join(description.lower().split())[:300]
        seen = self._run_discoveries[run_id]
        if not fingerprint or fingerprint in seen:
            return
        seen.add(fingerprint)
        if len(seen) > self.budget.max_discoveries:
            await self.record_event("discovery_deferred", message.content, target=message.sender_id)
            return

        raw = await self.think(
            f"A Head reported an out-of-contract discovery: {message.content}\n"
            f"Current Head roles: {self._run_head_roles[run_id]}\n"
            "Return JSON with action=assign_existing|create_head|defer, reason, "
            "target_agent_id, guidance, and optional new_head contract. Expansion is a last resort."
        )
        decision = extract_json_value(raw, dict) or {"action": "defer", "reason": raw}
        action = str(decision.get("action", "defer"))
        if action == "assign_existing":
            target = str(decision.get("target_agent_id", ""))
            if target in self._run_heads[run_id]:
                guidance = str(decision.get("guidance") or description)
                head = self._run_heads[run_id][target]
                if target in active:
                    await self.send_message(target, MessageType.GUIDANCE, {"text": guidance})
                else:
                    active[target] = asyncio.create_task(
                        self._run_head_guarded(head, head.resume(guidance))
                    )
        elif action == "create_head" and len(self._run_heads[run_id]) < self.budget.max_children:
            spec = decision.get("new_head")
            if isinstance(spec, dict):
                contract = self._contract_from_spec(
                    spec,
                    self._run_journals[run_id].manifest.user_request,
                )
                contract.context["discovery"] = message.content
                contract.role = self._unique_role(run_id, contract.role, pending)
                pending[contract.role] = contract
        await self.record_event("head_discovery_triaged", {
            "discovery": message.content,
            "decision": decision,
        })
        self._phase = "coordinating_heads"

    async def _create_head(
        self,
        run_id: str,
        contract: TaskContract,
        parent_progress: TaskProgress,
    ) -> HeadAgent:
        head_id = AgentRegistry.generate_id("head")
        channel = self.router.register(
            head_id,
            AgentRole.HEAD,
            parent_id=self.id,
            run_id=run_id,
        )
        self._run_head_roles[run_id][head_id] = contract.role
        head = HeadAgent(
            agent_id=head_id,
            role=contract.role,
            task=contract.goal,
            master_id=self.id,
            llm_client=self.llm_client,
            node_llm_client=self._executor_client,
            router=self.router,
            channel=channel,
            tool_registry=self.tool_registry,
            agent_registry=self.agent_registry,
            skill_catalog=self._builtin_toolset.skill_catalog,
            peer_roster=self._build_peer_roster(run_id, head_id),
            memory_store=MemoryStore(f"head-{contract.role}", self._memory_root),
            contract=contract,
            budget=self.head_budget.model_copy(deep=True),
            child_budget=self.node_budget.model_copy(deep=True),
            run_id=run_id,
            journal=self._run_journals[run_id],
            permission_manager=self.permission_manager,
        )
        self.agent_registry.register(head, {
            "role": contract.role,
            "task": contract.goal,
            "run_id": run_id,
            "parent_id": self.id,
        })
        self._run_heads[run_id][head_id] = head
        await self._run_journals[run_id].register_agent(head_id, contract.role, self.id)
        self._update_all_peer_rosters(run_id)
        parent_progress.add_sub_task(TaskProgress(
            task_id=head_id,
            description=contract.goal,
            status=TaskStatus.IN_PROGRESS,
            assigned_to=head_id,
        ))
        return head

    def _build_peer_roster(self, run_id: str, exclude_id: str) -> str:
        lines = [
            f"- {head_id}: {role}: {self._run_heads[run_id][head_id].contract.goal}"
            for head_id, role in self._run_head_roles[run_id].items()
            if head_id != exclude_id and head_id in self._run_heads[run_id]
        ]
        return "\n".join(lines) if lines else "No peers yet."

    def _update_all_peer_rosters(self, run_id: str) -> None:
        for head_id, head in self._run_heads[run_id].items():
            head.update_peer_roster(self._build_peer_roster(run_id, head_id))

    async def _aggregate_final_results(self, run_id: str, original_request: str) -> str:
        self._phase = "final_synthesis"
        latest = self._latest_outcomes(run_id)
        requirements = self._run_journals[run_id].manifest.requirements
        results = [item.model_dump(mode="json") for item in latest]
        deferred = [
            contract.model_dump(mode="json")
            for contract in self._run_deferred_contracts[run_id]
        ]
        results_for_prompt: Any = results or (
            "No Head result was available; provide the best reasoned response and explain the gap."
        )
        await self.record_event("master_step_started", {
            "step": "final_synthesis",
            "message": "Master is validating results and composing the user response.",
        })
        response = await self.think(
            "Produce the final response addressed directly to the user. Use the user's "
            "language and answer their actual request. Verify coverage against every "
            "requirement, resolve contradictions, and disclose material gaps. Do not expose "
            "the Master/Head/Node workflow, contracts, progress bookkeeping, or return an "
            "internal JSON validation object unless the user explicitly requested that format. "
            "Do not concatenate reports.\n\n"
            f"Original request: {original_request}\n"
            f"Current requirements: {requirements}\n"
            f"Head outcomes: {results_for_prompt}\n"
            f"Deferred contracts caused by the hard Head budget: {deferred}\n"
            f"Progress:\n{await self.get_progress_report(run_id)}"
        )
        await self.record_event("master_step_finished", {
            "step": "final_synthesis",
            "message": "Master finished the user-facing response.",
        })
        return response

    def _contracts_from_plan(
        self,
        plan: dict[str, Any],
        original_request: str,
    ) -> dict[str, TaskContract]:
        pending: dict[str, TaskContract] = {}
        for spec in plan.get("sub_tasks", [])[:4]:
            if not isinstance(spec, dict):
                continue
            contract = self._contract_from_spec(spec, original_request)
            contract.role = self._unique_role("", contract.role, pending)
            pending[contract.role] = contract
        return pending

    @staticmethod
    def _contract_from_spec(
        spec: dict[str, Any],
        original_request: str = "",
    ) -> TaskContract:
        goal = str(spec.get("description") or spec.get("goal") or spec.get("task") or "")
        return TaskContract(
            role=str(spec.get("role") or "general_owner"),
            goal=goal or "Complete the assigned part of the request",
            scope=str(spec.get("scope") or goal),
            deliverable=str(spec.get("expected_output") or spec.get("deliverable") or "Task result"),
            acceptance_criteria=[str(item) for item in spec.get("acceptance_criteria", [])],
            dependencies=[str(item) for item in spec.get("dependencies", [])],
            context={"user_request": original_request} if original_request else {},
        )

    async def _commit_direct_response(
        self,
        run_id: str,
        request: str,
        response: str,
        reason: str,
    ) -> str:
        lock = self._run_locks[run_id]
        async with lock:
            if not self._run_updates[run_id].empty():
                # A steering event arrived while Master was answering. Re-route it
                # before exposing an already-obsolete checkpoint.
                reroute = True
            else:
                reroute = False
                return await self._commit_direct_response_locked(
                    run_id, request, response, reason
                )
        if reroute:
            return await self._resume_master_only_run(run_id)
        return response

    async def _commit_direct_response_locked(
        self,
        run_id: str,
        request: str,
        response: str,
        reason: str,
    ) -> str:
        tracker = ProgressTracker()
        root = tracker.create_root(run_id, request)
        root.mark_completed(response[:300])
        self._run_progress[run_id] = tracker
        self._run_active_head_tasks[run_id] = {}
        self.context.add_message({"role": "user", "content": request})
        self.context.add_message({"role": "assistant", "content": response})
        await self.record_event("master_direct_response", {
            "reason": reason,
            "message": "Master answered directly; no delegation was needed.",
        })
        await self._retain_master_state(run_id, response)
        journal = self._run_journals[run_id]
        checkpoint = await journal.commit_checkpoint(response, reason=reason)
        await journal.record(
            "run_checkpoint_committed",
            source=self.id,
            task_id=run_id,
            payload={
                "checkpoint_id": checkpoint.checkpoint_id,
                "revision": checkpoint.revision,
                "response_chars": len(response),
            },
        )
        return response

    async def _resume_master_only_run(self, run_id: str) -> str:
        """Re-route steering on a retained Master-only Run without inventing a Head."""
        self._phase = "rerouting_user_revision"
        journal = self._run_journals[run_id]
        applied_updates = []
        while not self._run_updates[run_id].empty():
            revision, update = self._run_updates[run_id].get_nowait()
            applied_updates.append((revision, update))
            await self.record_event("user_update_applied", {
                "revision": revision,
                "update": update,
                "decision": {"action": "reroute_from_master"},
            })

        requirements = journal.manifest.requirements
        current_request = requirements[0]
        if len(requirements) > 1:
            current_request += "\n\nAdditional user requirements:\n" + "\n".join(
                f"- {item}" for item in requirements[1:]
            )

        await self.record_event("master_step_started", {
            "step": "rerouting",
            "message": "Master is re-evaluating the retained Run after user guidance.",
            "revisions": [revision for revision, _ in applied_updates],
        })
        tool_categories = list(self.tool_registry.get_categories())
        force_hierarchical = any(
            self.task_decomposer.explicit_hierarchy_requested(update)
            for _, update in applied_updates
        )
        plan = await self.task_decomposer.decompose(
            current_request,
            tool_categories,
            force_hierarchical=force_hierarchical,
            runtime_awareness=self.begin_llm_turn_awareness,
        )
        pending = self._contracts_from_plan(plan, current_request)
        await self.record_event("master_step_finished", {
            "step": "rerouting",
            "head_count": len(pending),
            "direct": not pending,
        })

        if not pending:
            response = str(plan.get("direct_response") or "").strip()
            if not response:
                response = await self._execute_master_direct(
                    run_id,
                    current_request,
                    str(plan.get("direct_instruction") or ""),
                )
            return await self._commit_direct_response(
                run_id,
                current_request,
                response,
                reason="master_direct_revision",
            )

        tracker = ProgressTracker()
        root = tracker.create_root(run_id, current_request)
        root.mark_in_progress()
        self._run_progress[run_id] = tracker
        active: dict[str, asyncio.Task[AgentOutcome]] = {}
        self._run_active_head_tasks[run_id] = active
        return await self._drive_and_commit(
            run_id,
            current_request,
            pending,
            active,
            root,
        )

    async def _execute_master_direct(
        self,
        run_id: str,
        request: str,
        instruction: str,
    ) -> str:
        """Let Master complete one coherent task, with bounded ordinary tools."""
        self._phase = "direct_execution"
        await self.record_event("master_step_started", {
            "step": "direct_execution",
            "message": "Master is completing this request without delegation.",
        })
        prompt = (
            "Complete the user's request yourself. You are the primary executor; Heads and "
            "Nodes are unnecessary for this task. Use an available tool only when it is "
            "actually needed. A plain text assistant message is not a final response in "
            "this phase. When the work is genuinely complete, call "
            "framework_commit_response exactly once with the complete user-facing answer. "
            "Never commit a progress note, future-tense action, or unfinished checklist. "
            "Do not mention routing, agents, contracts, or internal JSON.\n\n"
            f"User request: {request}\n"
            f"Routing guidance: {instruction or 'Use your best judgment.'}"
        )
        schemas = self.tool_registry.get_openai_schemas() + [_master_finish_tool_schema()]

        current_input = prompt
        # Reserve one reasoning turn for a truthful best-effort synthesis if ordinary
        # tool execution does not explicitly commit first.
        for _ in range(max(0, self.remaining_turns - 1)):
            if self.remaining_seconds <= 0:
                break
            completion = await self.think_with_tools(current_input, schemas)
            message = completion.choices[0].message
            if not message.tool_calls:
                current_input = (
                    "That was not committed because it was plain assistant text. If work "
                    "remains, use the necessary tools now. If it is complete, call "
                    "framework_commit_response with the full final answer."
                )
                continue

            finish_calls = [
                call for call in message.tool_calls
                if str(call.function.name) == "framework_commit_response"
            ]
            business_calls = [
                call for call in message.tool_calls
                if str(call.function.name) != "framework_commit_response"
            ]
            for tool_call in message.tool_calls:
                if str(tool_call.function.name) == "framework_commit_response":
                    continue
                result = await self._execute_master_tool_call(run_id, tool_call)
                self.add_tool_result(tool_call.id, result)

            if finish_calls:
                finish_call = finish_calls[0]
                if len(finish_calls) != 1 or business_calls:
                    for invalid_finish in finish_calls:
                        self.add_tool_result(
                            invalid_finish.id,
                            json.dumps({
                                "error": (
                                    "framework_commit_response must be the only tool call "
                                    "in its turn and may be called exactly once"
                                )
                            }),
                        )
                else:
                    try:
                        finish_arguments = json.loads(finish_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        finish_arguments = {}
                    response = str(finish_arguments.get("response") or "").strip()
                    if response:
                        self.add_tool_result(
                            finish_call.id,
                            json.dumps({"status": "committed"}),
                        )
                        await self.record_event("master_step_finished", {
                            "step": "direct_execution",
                            "message": "Master explicitly committed the completed response.",
                        })
                        return response
                    self.add_tool_result(
                        finish_call.id,
                        json.dumps({"error": "response cannot be blank"}),
                    )
            current_input = (
                "Use the tool results to complete the original request. Call another "
                "business tool only if necessary; otherwise call "
                "framework_commit_response with the complete final answer now."
            )

        final_data: dict[str, Any] = {}
        if self.remaining_turns > 0 and self.remaining_seconds > 0:
            self._phase = "forced_final_synthesis"
            forced = await self.think(
                "The direct-execution phase must end now. Synthesize the best complete "
                "user-facing response from evidence already in context. Do not describe future "
                "work. Return exactly one JSON object: "
                '{"response":"complete answer","complete":true}. '
                "If the task could not be completed, response must clearly state the exact gap."
            )
            final_data = extract_json_value(forced, dict) or {}
        response = str(final_data.get("response") or "").strip()
        if not response:
            response = (
                "The configured direct-execution budget ended before Master produced a "
                "complete answer. Partial actions remain retained in the Run journal; no "
                "progress note was misrepresented as a final response."
            )
        await self.record_event("master_step_finished", {
            "step": "direct_execution",
            "message": "Master reached the turn budget and performed a forced final synthesis.",
        })
        return response

    async def _execute_master_tool_call(self, run_id: str, tool_call: Any) -> str:
        name = str(tool_call.function.name)
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON arguments"}
            self.observe_action_result(f"tool {name}", result)
            return json.dumps(result)
        self._run_master_tool_call_counts[run_id] += 1
        await self.record_event("tool_call", {
            "name": name,
            "arguments": arguments,
            "owner": "master",
        })
        result = await self.tool_executor.execute(
            name,
            arguments,
            ApprovalContext(
                run_id=run_id,
                agent_id=self.id,
                task_id=self.task_id,
                journal=self._run_journals.get(run_id),
            ),
        )
        await self.record_event("tool_result", {
            "name": name,
            "result": str(result)[:4000],
            "owner": "master",
        })
        self.observe_action_result(f"tool {name}", result)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def _unique_role(
        self,
        run_id: str,
        base_role: str,
        pending: dict[str, TaskContract],
    ) -> str:
        used = set(pending)
        if run_id:
            used.update(self._run_head_roles[run_id].values())
        role = base_role
        suffix = 2
        while role in used:
            role = f"{base_role}_{suffix}"
            suffix += 1
        return role

    def _latest_outcomes(self, run_id: str) -> list[AgentOutcome]:
        return [
            outcomes[-1]
            for outcomes in self._run_head_outcomes[run_id].values()
            if outcomes
        ]

    def _latest_outcomes_by_role(self, run_id: str) -> dict[str, AgentOutcome]:
        result: dict[str, AgentOutcome] = {}
        for head_id, outcomes in self._run_head_outcomes[run_id].items():
            if outcomes:
                result[self._run_head_roles[run_id][head_id]] = outcomes[-1]
        return result

    async def _build_run_topology(self, run_id: str) -> dict[str, Any]:
        active = self._run_active_head_tasks.get(run_id, {})
        heads = []
        for head_id, head in self._run_heads.get(run_id, {}).items():
            outcomes = self._run_head_outcomes[run_id].get(head_id, [])
            latest = outcomes[-1].model_dump(mode="json") if outcomes else None
            head_status = "running" if head_id in active else (
                latest["status"] if latest else "retained"
            )
            nodes = []
            for node_id, node in head._node_agents.items():
                node_outcomes = head._node_outcomes.get(node_id, [])
                node_latest = (
                    node_outcomes[-1].model_dump(mode="json")
                    if node_outcomes else None
                )
                node_status = "running" if node_id in head._node_tasks else (
                    node_latest["status"] if node_latest else "retained"
                )
                nodes.append({
                    "agent_id": node_id,
                    "role": node.role,
                    "goal": node.contract.goal,
                    "status": node_status,
                    "outcome": node_latest,
                })
            heads.append({
                "agent_id": head_id,
                "role": head.role,
                "goal": head.contract.goal,
                "status": head_status,
                "outcome": latest,
                "nodes": nodes,
            })
        if not heads:
            return await self._build_snapshot_topology(run_id)
        return {
            "master": {
                "agent_id": self.id,
                "role": "master",
                "status": self._run_journals[run_id].manifest.status.value,
            },
            "heads": heads,
        }

    async def _build_snapshot_topology(self, run_id: str) -> dict[str, Any]:
        journal = self._run_journals[run_id]
        snapshots: dict[str, dict[str, Any]] = {}
        for agent_id in journal.manifest.agent_ids:
            snapshot = await journal.read_agent_snapshot(agent_id)
            if snapshot:
                snapshots[agent_id] = snapshot

        master_snapshot = next(
            (item for item in snapshots.values() if item.get("role") == "master"),
            {},
        )
        head_ids = master_snapshot.get("head_ids") or [
            agent_id for agent_id, item in snapshots.items()
            if "node_ids" in item and item.get("role") != "master"
        ]
        heads = []
        for head_id in head_ids:
            head = snapshots.get(head_id, {})
            contract = head.get("contract") or {}
            outcome = head.get("outcome") or {}
            nodes = []
            for node_id in head.get("node_ids") or []:
                node = snapshots.get(node_id, {})
                node_contract = node.get("contract") or {}
                node_outcome = node.get("outcome") or {}
                nodes.append({
                    "agent_id": node_id,
                    "role": node.get("role", "node"),
                    "goal": node_contract.get("goal", "Retained Node snapshot"),
                    "status": node_outcome.get("status", "retained"),
                    "outcome": node_outcome or None,
                })
            heads.append({
                "agent_id": head_id,
                "role": head.get("role", "head"),
                "goal": contract.get("goal", "Retained Head snapshot"),
                "status": outcome.get("status", "retained"),
                "outcome": outcome or None,
                "nodes": nodes,
            })

        return {
            "master": {
                "agent_id": master_snapshot.get("agent_id", self.id),
                "role": "master",
                "status": journal.manifest.status.value,
            },
            "heads": heads,
        }

    def _load_persisted_runs(self) -> None:
        """Expose prior journals as read-only cold snapshots after process restart."""
        for journal in RunJournal.discover(self._run_root):
            run_id = journal.run_id
            self._run_journals[run_id] = journal
            self._run_updates[run_id] = asyncio.Queue()
            self._run_locks[run_id] = asyncio.Lock()
            self._cold_run_ids.add(run_id)

    async def _activate_master_context(self, run_id: str, fresh: bool) -> None:
        self.journal = self._run_journals[run_id]
        self.run_id = run_id
        self.task_id = run_id
        self._sent_counts.clear()
        self._running = True
        self._phase = "planning" if fresh else "resuming"
        if fresh:
            self.context.clear_messages()
            self.context.clear_pinned()
            self._runtime_observations.clear()
        elif run_id in self._master_context_states:
            self.context.restore_state(self._master_context_states[run_id])
        self.start_clock()

    async def _retain_master_state(self, run_id: str, final_response: str) -> None:
        self._running = False
        self._phase = "retained"
        self._master_context_states[run_id] = self.context.export_state()
        await self._run_journals[run_id].snapshot_agent(
            self,
            {
                "final_response": final_response,
                "head_ids": list(self._run_heads[run_id]),
            },
        )

    async def _post_run_learning(
        self,
        keyword: str | None,
        execution_log: str,
        request: str,
        strategy: str,
        final: str,
    ) -> None:
        try:
            if keyword:
                await self.plan_cache.store(keyword, execution_log)
            await self.memory_store.append(
                f"## Request: {request[:120]}\n"
                f"Strategy: {strategy}\n"
                f"Result: {final[:500]}\n"
            )
        except Exception:
            logger.exception("Post-run learning failed")

    def _new_active_task_map(self, run_id: str) -> dict[str, asyncio.Task[AgentOutcome]]:
        active: dict[str, asyncio.Task[AgentOutcome]] = {}
        self._run_active_head_tasks[run_id] = active
        return active

    async def _enforce_retention_limit(self) -> None:
        if self.retained_run_limit is None or self.retained_run_limit < 1:
            return
        retained = [
            journal for journal in self._run_journals.values()
            if journal.manifest.status != RunStatus.ARCHIVED
        ]
        retained.sort(key=lambda item: item.manifest.created_at)
        while len(retained) >= self.retained_run_limit:
            oldest = retained.pop(0)
            task = self._run_tasks.get(oldest.run_id)
            if task and not task.done():
                break
            await self.archive_run(oldest.run_id)
