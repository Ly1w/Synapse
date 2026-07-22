from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Coroutine

from ..communication.channel import Channel
from ..communication.message import Message, MessageType
from ..communication.router import AgentRole, Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient
from ..memory.memory_store import MemoryStore
from ..skills import SkillCatalog
from ..system_prompts import build_head_system_prompt
from ..tools.registry import ToolRegistry
from ..tools.permissions import PermissionManager
from .base_agent import BaseAgent
from .contracts import AgentBudget, AgentOutcome, OutcomeStatus, TaskContract
from .node_agent import NodeAgent
from .parsing import extract_json_value
from .registry import AgentRegistry

logger = logging.getLogger(__name__)


PREPARATION_PROMPT = """\
Prepare a bounded execution strategy for your contract.

First reason about the problem yourself. Then create 0-{max_nodes} Node assignments only
where delegation is useful. A Node must receive a concrete contract, not a vague role.

Return JSON:
{{
  "head_analysis": "your initial analysis, risks, and integration strategy",
  "node_assignments": [
    {{
      "role": "focused role",
      "goal": "specific goal",
      "scope": "explicit boundary",
      "deliverable": "concrete output",
      "acceptance_criteria": ["checkable condition"]
    }}
  ]
}}
"""


SYNTHESIS_PROMPT = """\
Act as the accountable owner of this contract. Synthesize the evidence below, resolve
conflicts, and check every acceptance criterion. Do not merely concatenate Node reports.

Contract:
{contract}

Your initial analysis:
{head_analysis}

Node outcomes:
{node_outcomes}

Return JSON:
{{
  "status": "completed|partial|blocked|failed",
  "summary": "integrated deliverable",
  "evidence": ["key evidence"],
  "unresolved": ["remaining gap"],
  "follow_up_tasks": [
    {{"goal":"one necessary repair", "resume_agent_id":"optional prior Node id"}}
  ]
}}

Only request follow-up work when a contract criterion cannot otherwise be assessed.
"""


class HeadAgent(BaseAgent):
    """Sub-problem owner that reasons, delegates selectively, validates, and converges."""

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
        skill_catalog: SkillCatalog | None = None,
        peer_roster: str = "",
        memory_store: MemoryStore | None = None,
        context_manager: ContextManager | None = None,
        max_turns: int = 20,
        contract: TaskContract | None = None,
        budget: AgentBudget | None = None,
        child_budget: AgentBudget | None = None,
        run_id: str = "",
        journal: Any | None = None,
        node_llm_client: LLMClient | None = None,
        permission_manager: PermissionManager | None = None,
    ):
        self.contract = contract or TaskContract(role=role, goal=task, scope=task)
        self.task = self.contract.goal
        self.master_id = master_id
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.skill_catalog = skill_catalog
        self.node_llm_client = node_llm_client or llm_client
        self.permission_manager = permission_manager
        self.peer_roster_text = peer_roster or "No peers yet."
        self.memory_store = memory_store
        self.child_budget = child_budget or AgentBudget(
            max_turns=10,
            max_peer_messages=2,
            max_discoveries=1,
            max_children=0,
            max_revision_rounds=0,
            timeout_seconds=240,
        )

        self._node_agents: dict[str, NodeAgent] = {}
        self._node_tasks: dict[str, asyncio.Task[AgentOutcome]] = {}
        self._node_outcomes: dict[str, list[AgentOutcome]] = defaultdict(list)
        self._node_descriptions: dict[str, str] = {}
        self._head_analysis = ""
        self._discoveries_seen: set[str] = set()
        self._discoveries_handled = 0
        self._revision_rounds = 0
        self._cancel_requested = False
        self._memory_loaded = False
        self._applied_revisions: set[int] = set()

        actual_budget = budget or AgentBudget(
            max_turns=max_turns,
            max_peer_messages=4,
            max_discoveries=2,
            max_children=4,
            max_revision_rounds=1,
            timeout_seconds=600,
        )
        system_prompt = build_head_system_prompt(
            self.contract.to_prompt(),
            self.peer_roster_text,
            actual_budget,
            self.child_budget,
            self.permission_manager.get_state() if self.permission_manager else None,
            self._skill_inventory(),
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

    def update_peer_roster(self, peer_roster: str) -> None:
        self.peer_roster_text = peer_roster or "No peers yet."
        self.refresh_system_prompt()

    def _skill_inventory(self) -> str:
        if self.skill_catalog is None:
            return "No model-invocable Skills are currently installed."
        return self.skill_catalog.model_inventory()

    def refresh_system_prompt(self) -> None:
        self.system_prompt = build_head_system_prompt(
            self.contract.to_prompt(),
            self.peer_roster_text,
            self.budget,
            self.child_budget,
            self.permission_manager.get_state() if self.permission_manager else None,
            self._skill_inventory(),
        )
        self.context.set_system_prompt(self.system_prompt)
        for node in self._node_agents.values():
            node.refresh_system_prompt()

    def runtime_state(self) -> dict[str, Any]:
        state = super().runtime_state()
        peer_messages = (
            self.sent_count(MessageType.PEER_REQUEST)
            + self.sent_count(MessageType.PEER_RESPONSE)
        )
        settled_phases = sum(len(items) for items in self._node_outcomes.values())
        state.update({
            "role_position": {
                "level": "Head",
                "parent_master_id": self.master_id,
                "responsibility": "Own, validate, and integrate exactly this contract",
                "goal": self.contract.goal,
                "scope": self.contract.scope or self.contract.goal,
                "deliverable": self.contract.deliverable,
                "may_create": "Node only",
                "must_not": [
                    "create Heads or redefine the global plan",
                    "delegate integration or acceptance checking",
                    "answer the end user",
                ],
            },
            "current_topology": {
                "peer_head_ids": self.router.get_peers(self.id),
                "node_ids": list(self._node_agents),
                "active_node_ids": list(self._node_tasks),
                "settled_node_phases": settled_phases,
                "inbox_pending": self.channel.pending,
            },
            "remaining_communication_and_expansion": {
                "node_slots_used": len(self._node_agents),
                "node_slots_max": self.budget.max_children,
                "node_slots_remaining": max(
                    0, self.budget.max_children - len(self._node_agents)
                ),
                "peer_messages_used": peer_messages,
                "peer_messages_max": self.budget.max_peer_messages,
                "peer_messages_remaining": max(
                    0, self.budget.max_peer_messages - peer_messages
                ),
                "discoveries_triaged": self._discoveries_handled,
                "discoveries_max": self.budget.max_discoveries,
                "discoveries_remaining": max(
                    0, self.budget.max_discoveries - self._discoveries_handled
                ),
                "revision_rounds_used": self._revision_rounds,
                "revision_rounds_max": self.budget.max_revision_rounds,
                "revision_rounds_remaining": max(
                    0, self.budget.max_revision_rounds - self._revision_rounds
                ),
            },
            "steering_and_control": {
                "applied_user_revisions": sorted(self._applied_revisions),
                "cancel_requested": self._cancel_requested,
            },
            "permission": (
                self.permission_manager.get_state() if self.permission_manager else None
            ),
        })
        return state

    async def run(self) -> AgentOutcome:
        self._running = True
        self._phase = "preparation"
        self.start_clock()
        await self.record_event("agent_phase_started", {"contract": self.contract.model_dump(mode="json")})
        logger.info("HeadAgent %s starting: %s", self.id, self.task[:100])

        try:
            await self._load_memory_once()
            preparation = await self._prepare_work()
            self._head_analysis = str(preparation.get("head_analysis", ""))
            await self._launch_assignments(preparation.get("node_assignments", []))
            await self._monitor_nodes()
            await self._drain_pending_messages()
            if self._node_tasks:
                await self._monitor_nodes()
            outcome = await self._synthesize_with_bounded_revision()
        except Exception as exc:
            logger.exception("HeadAgent %s failed", self.id)
            outcome = AgentOutcome(
                agent_id=self.id,
                task_id=self.contract.task_id,
                status=OutcomeStatus.FAILED,
                summary=f"Head failed: {exc}",
                unresolved=[self.contract.goal],
            )
        finally:
            await self._cancel_active_nodes("Head is stopping")
            self._running = False
            self._phase = "retained"

        await self._finish_phase(outcome)
        return outcome

    async def resume(
        self,
        guidance: str,
        revision: int | list[int] | None = None,
    ) -> AgentOutcome:
        """Resume the same Head and its retained Nodes for a user-approved revision."""
        self._running = True
        self._phase = "applying_user_revision"
        self.start_clock()
        self._cancel_requested = False
        self._sent_counts.clear()
        self._revision_rounds = 0
        self.context.add_message({"role": "user", "content": f"[New user requirement] {guidance}"})
        self.contract.context[f"user_revision_{self._revision_rounds + 1}"] = guidance
        await self.record_event("agent_resumed", {"guidance": guidance})

        try:
            await self._apply_guidance(guidance, allow_resume=True, revision=revision)
            await self._monitor_nodes()
            await self._drain_pending_messages()
            if self._node_tasks:
                await self._monitor_nodes()
            outcome = await self._synthesize_with_bounded_revision()
        except Exception as exc:
            logger.exception("HeadAgent %s resume failed", self.id)
            outcome = AgentOutcome(
                agent_id=self.id,
                task_id=self.contract.task_id,
                status=OutcomeStatus.FAILED,
                summary=f"Head revision failed: {exc}",
                unresolved=[guidance],
            )
        finally:
            await self._cancel_active_nodes("Head revision is stopping")
            self._running = False
            self._phase = "retained"

        await self._finish_phase(outcome)
        return outcome

    async def _load_memory_once(self) -> None:
        if self._memory_loaded or not self.memory_store:
            return
        await self.memory_store.initialize()
        memory = await self.memory_store.read()
        if memory:
            self.context.add_pinned(self.memory_store.get_injection_prompt(memory))
        self._memory_loaded = True

    async def _prepare_work(self) -> dict[str, Any]:
        await self.record_event("agent_step_started", {
            "step": "head_preparation",
            "message": "Head is analyzing its contract and deciding whether Nodes are useful.",
        })
        raw = await self.think(PREPARATION_PROMPT.format(max_nodes=self.budget.max_children))
        data = extract_json_value(raw, dict)
        if not data:
            await self.record_event("agent_step_finished", {
                "step": "head_preparation",
                "node_count": 0,
            })
            return {"head_analysis": raw, "node_assignments": []}
        assignments = data.get("node_assignments", [])
        if not isinstance(assignments, list):
            assignments = []
        data["node_assignments"] = assignments[:self.budget.max_children]
        await self.record_event("agent_step_finished", {
            "step": "head_preparation",
            "node_count": len(data["node_assignments"]),
        })
        return data

    async def _launch_assignments(self, assignments: list[dict[str, Any]]) -> None:
        new_nodes: list[NodeAgent] = []
        for spec in assignments:
            if len(self._node_agents) >= self.budget.max_children or not isinstance(spec, dict):
                break
            node = await self._create_node(spec)
            new_nodes.append(node)
        self._refresh_sibling_rosters()
        for node in new_nodes:
            self._node_tasks[node.id] = asyncio.create_task(
                self._run_node_guarded(node, node.run())
            )

    async def _create_node(self, spec: dict[str, Any]) -> NodeAgent:
        role = str(spec.get("role") or "executor")
        goal = str(spec.get("goal") or spec.get("task") or self.task)
        contract = TaskContract(
            role=role,
            goal=goal,
            scope=str(spec.get("scope") or goal),
            deliverable=str(spec.get("deliverable") or "Focused task result"),
            acceptance_criteria=[str(item) for item in spec.get("acceptance_criteria", [])],
            context={
                "parent_contract": self.contract.task_id,
                "parent_goal": self.contract.goal,
                "user_request": self.contract.context.get("user_request", ""),
            },
        )
        node_id = AgentRegistry.generate_id("node")
        channel = self.router.register(
            node_id,
            AgentRole.NODE,
            parent_id=self.id,
            run_id=self.run_id,
        )
        node = NodeAgent(
            agent_id=node_id,
            role=role,
            task=goal,
            head_id=self.id,
            llm_client=self.node_llm_client,
            router=self.router,
            channel=channel,
            tool_registry=self.tool_registry,
            skill_catalog=self.skill_catalog,
            contract=contract,
            budget=self.child_budget.model_copy(deep=True),
            run_id=self.run_id,
            journal=self.journal,
            permission_manager=self.permission_manager,
        )
        self.agent_registry.register(node, {
            "role": role,
            "task": goal,
            "run_id": self.run_id,
            "parent_id": self.id,
        })
        self._node_agents[node_id] = node
        self._node_descriptions[node_id] = f"{role}: {goal}"
        if self.journal:
            await self.journal.register_agent(node_id, role, self.id)
        logger.info("HeadAgent %s created node %s (role=%s)", self.id, node_id, role)
        return node

    def _refresh_sibling_rosters(self) -> None:
        for node_id, node in self._node_agents.items():
            roster = {
                peer_id: description
                for peer_id, description in self._node_descriptions.items()
                if peer_id != node_id
            }
            node.set_sibling_roster(roster)

    async def _run_node_guarded(
        self,
        node: NodeAgent,
        coroutine: Coroutine[Any, Any, AgentOutcome],
    ) -> AgentOutcome:
        try:
            return await asyncio.wait_for(coroutine, timeout=node.budget.timeout_seconds)
        except asyncio.TimeoutError:
            node.stop()
            outcome = AgentOutcome(
                agent_id=node.id,
                task_id=node.contract.task_id,
                status=OutcomeStatus.PARTIAL,
                summary="Node timed out; partial context is retained.",
                unresolved=[node.contract.goal],
                metadata={"timeout_seconds": node.budget.timeout_seconds},
            )
        except asyncio.CancelledError:
            node.stop()
            outcome = AgentOutcome(
                agent_id=node.id,
                task_id=node.contract.task_id,
                status=OutcomeStatus.CANCELLED,
                summary="Node cancelled by Head.",
                unresolved=[node.contract.goal],
            )
        except Exception as exc:
            logger.exception("Node task %s crashed", node.id)
            outcome = AgentOutcome(
                agent_id=node.id,
                task_id=node.contract.task_id,
                status=OutcomeStatus.FAILED,
                summary=f"Node crashed: {exc}",
                unresolved=[node.contract.goal],
            )
        if node.journal:
            await node.journal.snapshot_agent(
                node,
                {
                    "contract": node.contract.model_dump(mode="json"),
                    "outcome": outcome.model_dump(mode="json"),
                },
            )
        return outcome

    async def _monitor_nodes(self) -> None:
        self._phase = "monitoring_nodes"
        while self._node_tasks and not self._cancel_requested and self.remaining_seconds > 0:
            message = await self.receive_message(timeout=min(0.25, self.remaining_seconds))
            if message:
                await self._handle_message(message)

            for node_id, task in list(self._node_tasks.items()):
                if not task.done():
                    continue
                try:
                    outcome = task.result()
                except Exception as exc:
                    node = self._node_agents[node_id]
                    outcome = AgentOutcome(
                        agent_id=node_id,
                        task_id=node.contract.task_id,
                        status=OutcomeStatus.FAILED,
                        summary=f"Uncaught Node failure: {exc}",
                        unresolved=[node.contract.goal],
                    )
                self._node_outcomes[node_id].append(outcome)
                del self._node_tasks[node_id]

        if self.remaining_seconds <= 0:
            await self._cancel_active_nodes("Head timeout reached")

    async def _handle_message(self, message: Message) -> None:
        if message.msg_type == MessageType.DISCOVERY:
            await self._handle_discovery(message)
        elif message.msg_type in (MessageType.PEER_REQUEST, MessageType.PEER_MSG):
            await self._handle_peer_request(message)
        elif message.msg_type == MessageType.PEER_RESPONSE:
            self.context.add_message({
                "role": "user",
                "content": f"[Peer Head {message.sender_id}] {message.content}",
            })
        elif message.msg_type in (MessageType.GUIDANCE, MessageType.CLARIFICATION):
            revision_value = message.content.get("revision")
            revision = int(revision_value) if isinstance(revision_value, int) else None
            await self._apply_guidance(
                str(message.content.get("text", message.content)),
                revision=revision,
            )
        elif message.msg_type == MessageType.CANCEL:
            self._cancel_requested = True
        # REPORT is an observation only. asyncio task completion is authoritative.

    async def _handle_discovery(self, message: Message) -> None:
        description = str(message.content.get("description") or message.content.get("text") or "")
        fingerprint = " ".join(description.lower().split())[:300]
        if not fingerprint or fingerprint in self._discoveries_seen:
            return
        self._discoveries_seen.add(fingerprint)
        if self._discoveries_handled >= self.budget.max_discoveries:
            self.context.add_message({"role": "user", "content": f"[Deferred discovery] {description}"})
            return
        self._discoveries_handled += 1

        raw = await self.think(
            "Triage this Node discovery without expanding work by default.\n"
            f"Discovery: {message.content}\n"
            f"My contract: {self.contract.to_prompt()}\n"
            f"Peers: {self.peer_roster_text}\n"
            "Return JSON with action=absorb|new_node|peer_head|escalate_master|defer, "
            "reason, target_agent_id, and optional node_assignment."
        )
        decision = extract_json_value(raw, dict) or {"action": "defer", "reason": raw}
        action = str(decision.get("action", "defer"))
        await self.record_event("discovery_triaged", {"discovery": message.content, "decision": decision})

        if action == "new_node" and len(self._node_agents) < self.budget.max_children:
            spec = decision.get("node_assignment")
            if isinstance(spec, dict):
                await self._launch_assignments([spec])
        elif action == "peer_head":
            target = str(decision.get("target_agent_id", ""))
            if (target in self.router.get_peers(self.id)
                    and self.sent_count(MessageType.PEER_REQUEST) < self.budget.max_peer_messages):
                await self.send_message(
                    target,
                    MessageType.PEER_REQUEST,
                    {"question": description, "reason": decision.get("reason", "")},
                )
        elif action == "escalate_master" and self.sent_count(MessageType.DISCOVERY) < 1:
            await self.send_message(
                self.master_id,
                MessageType.DISCOVERY,
                {
                    "description": description,
                    "impact": message.content.get("impact", ""),
                    "reason": decision.get("reason", ""),
                    "source_head": self.id,
                },
            )
        else:
            self.context.add_message({
                "role": "user",
                "content": f"[Discovery retained for synthesis] {description}",
            })

    async def _handle_peer_request(self, message: Message) -> None:
        peer_count = (
            self.sent_count(MessageType.PEER_REQUEST)
            + self.sent_count(MessageType.PEER_RESPONSE)
        )
        if peer_count >= self.budget.max_peer_messages:
            return
        question = str(message.content.get("question") or message.content.get("text") or "")
        answer = await self.think(
            f"Peer Head {message.sender_id} asks: {question}\n"
            "Give a concise answer using only information already available in your scope."
        )
        await self.send_message(
            message.sender_id,
            MessageType.PEER_RESPONSE,
            {"answer": answer},
            correlation_id=message.correlation_id,
            reply_to=message.id,
        )

    async def _apply_guidance(
        self,
        guidance: str,
        allow_resume: bool = False,
        revision: int | list[int] | None = None,
    ) -> None:
        self._phase = "applying_guidance"
        self.contract.context[f"guidance_{len(self.contract.context) + 1}"] = guidance
        roster = "\n".join(f"- {node_id}: {desc}" for node_id, desc in self._node_descriptions.items())
        raw = await self.think(
            f"Apply this new user guidance to my contract: {guidance}\n"
            f"Existing Nodes:\n{roster or 'None'}\n"
            "Return JSON with head_adjustment, node_guidance=[{agent_id,text}], "
            "resume_nodes=[{agent_id,text}], and additional_nodes=[task contracts]. "
            "Use only the changes genuinely required."
        )
        decision = extract_json_value(raw, dict) or {}
        node_guidance = decision.get("node_guidance", [])
        if not isinstance(node_guidance, list):
            node_guidance = []
        for item in node_guidance:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("agent_id", ""))
            if node_id in self._node_tasks:
                await self.send_message(
                    node_id,
                    MessageType.GUIDANCE,
                    {"text": str(item.get("text") or guidance)},
                )

        if allow_resume:
            resume_nodes = decision.get("resume_nodes", [])
            if not isinstance(resume_nodes, list):
                resume_nodes = []
            for item in resume_nodes:
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("agent_id", ""))
                node = self._node_agents.get(node_id)
                if node and node_id not in self._node_tasks:
                    text = str(item.get("text") or guidance)
                    self._node_tasks[node_id] = asyncio.create_task(
                        self._run_node_guarded(node, node.resume(text))
                    )

        additional = decision.get("additional_nodes", [])
        if isinstance(additional, list):
            await self._launch_assignments(additional)
        if isinstance(revision, list):
            self._applied_revisions.update(revision)
        elif revision is not None:
            self._applied_revisions.add(revision)
        await self.record_event("guidance_applied", {"guidance": guidance, "decision": decision})

    async def _synthesize_with_bounded_revision(self) -> AgentOutcome:
        outcome, follow_ups = await self._synthesize_results()
        while (
            follow_ups
            and self._revision_rounds < self.budget.max_revision_rounds
            and not self._cancel_requested
        ):
            self._revision_rounds += 1
            launched = False
            for item in follow_ups:
                if not isinstance(item, dict):
                    continue
                goal = str(item.get("goal", "")).strip()
                node_id = str(item.get("resume_agent_id", ""))
                node = self._node_agents.get(node_id)
                if goal and node and node_id not in self._node_tasks:
                    self._node_tasks[node_id] = asyncio.create_task(
                        self._run_node_guarded(node, node.resume(goal))
                    )
                    launched = True
                elif goal and len(self._node_agents) < self.budget.max_children:
                    await self._launch_assignments([{
                        "role": "verification",
                        "goal": goal,
                        "scope": "Only the identified synthesis gap",
                        "deliverable": "Evidence resolving the gap",
                        "acceptance_criteria": [goal],
                    }])
                    launched = True
                if launched:
                    break
            if not launched:
                break
            await self._monitor_nodes()
            outcome, follow_ups = await self._synthesize_results()
        return outcome

    async def _synthesize_results(self) -> tuple[AgentOutcome, list[dict[str, Any]]]:
        self._phase = "synthesis"
        serialized = []
        for node_id, outcomes in self._node_outcomes.items():
            for phase, outcome in enumerate(outcomes, start=1):
                serialized.append({
                    "node_id": node_id,
                    "phase": phase,
                    **outcome.model_dump(mode="json"),
                })
        await self.record_event("agent_step_started", {
            "step": "head_synthesis",
            "message": "Head is integrating evidence and checking acceptance criteria.",
        })
        raw = await self.think(SYNTHESIS_PROMPT.format(
            contract=self.contract.to_prompt(),
            head_analysis=self._head_analysis or "No separate initial analysis.",
            node_outcomes=serialized or "No Node delegation was needed.",
        ))
        await self.record_event("agent_step_finished", {
            "step": "head_synthesis",
            "message": "Head finished its integrated outcome.",
        })
        data = extract_json_value(raw, dict) or {}
        status_text = str(data.get("status", "partial" if not data else "completed"))
        try:
            status = OutcomeStatus(status_text)
        except ValueError:
            status = OutcomeStatus.PARTIAL
        outcome = AgentOutcome(
            agent_id=self.id,
            task_id=self.contract.task_id,
            status=status,
            summary=str(data.get("summary") or raw),
            evidence=[str(item) for item in data.get("evidence", [])],
            unresolved=[str(item) for item in data.get("unresolved", [])],
            metadata={
                "node_count": len(self._node_agents),
                "node_phases": sum(len(items) for items in self._node_outcomes.values()),
                "revision_rounds": self._revision_rounds,
                "applied_revisions": sorted(self._applied_revisions),
            },
        )
        follow_ups = data.get("follow_up_tasks", [])
        return outcome, follow_ups if isinstance(follow_ups, list) else []

    async def _finish_phase(self, outcome: AgentOutcome) -> None:
        await self.send_message(
            self.master_id,
            MessageType.REPORT,
            outcome.to_message_content(),
        )
        if self.memory_store and outcome.successful:
            await self.memory_store.append(
                f"## Task: {self.task[:100]}\nResult: {outcome.summary[:500]}\n"
            )
        await self.record_event("agent_phase_finished", outcome.to_message_content())
        if self.journal:
            await self.journal.snapshot_agent(
                self,
                {
                    "contract": self.contract.model_dump(mode="json"),
                    "outcome": outcome.model_dump(mode="json"),
                    "node_ids": list(self._node_agents),
                },
            )
        logger.info("HeadAgent %s completed phase (%s)", self.id, outcome.status.value)

    async def _cancel_active_nodes(self, reason: str) -> None:
        items = list(self._node_tasks.items())
        for node_id, task in items:
            if not task.done():
                await self.send_message(node_id, MessageType.CANCEL, {"text": reason})
                task.cancel()
        results = await asyncio.gather(
            *(task for _, task in items),
            return_exceptions=True,
        )
        for (node_id, _), result in zip(items, results):
            if isinstance(result, AgentOutcome):
                prior = self._node_outcomes[node_id]
                if not prior or prior[-1] is not result:
                    prior.append(result)
        self._node_tasks.clear()

    async def _drain_pending_messages(self) -> None:
        for message in self.drain_messages():
            await self._handle_message(message)

    def export_state(self) -> dict[str, Any]:
        state = super().export_state()
        state.update({
            "contract": self.contract.model_dump(mode="json"),
            "head_analysis": self._head_analysis,
            "node_ids": list(self._node_agents),
            "node_outcomes": {
                node_id: [item.model_dump(mode="json") for item in outcomes]
                for node_id, outcomes in self._node_outcomes.items()
            },
            "discoveries_seen": sorted(self._discoveries_seen),
            "revision_rounds": self._revision_rounds,
            "applied_revisions": sorted(self._applied_revisions),
        })
        return state
