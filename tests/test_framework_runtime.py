from __future__ import annotations

import asyncio
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent_framework.communication.message import Message, MessageType
from agent_framework.communication.router import AgentRole, Router
from agent_framework.context.manager import ContextManager
from agent_framework.core.contracts import AgentBudget, OutcomeStatus, TaskContract
from agent_framework.core.master_agent import MasterAgent
from agent_framework.core.node_agent import NodeAgent
from agent_framework.llm.config import LLMConfig
from agent_framework.memory import LongTermMemory
from agent_framework.planning.decomposer import TaskDecomposer
from agent_framework.runtime.run import RunJournal, RunStatus
from agent_framework.skills import SkillCatalog
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.mcp_provider import MCPServerConfig
from agent_framework.tools.permissions import ApprovalContext
from deep_research import fetch as research_fetch
from deep_research import notes as research_notes
from deep_research import report as research_report
from deep_research import search as research_search


def _completion(content: str = "", tool_calls: list | None = None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DeterministicLLM:
    def __init__(self, gate_node: bool = False):
        self.gate_node = gate_node
        self.chat_text_calls = 0
        self.node_started = asyncio.Event()
        self.node_release = asyncio.Event()
        self.route_messages = []
        if not gate_node:
            self.node_release.set()

    async def chat_text(self, messages, **kwargs):
        self.chat_text_calls += 1
        prompt = str(messages[-1].get("content", ""))
        if "routing work for a Master Agent" in prompt:
            self.route_messages = list(messages)
            if "User request: 你好" in prompt:
                return json.dumps({
                    "mode": "direct",
                    "reason": "No delegation benefit",
                    "direct_response": "你好！有什么我可以帮你的吗？",
                    "direct_instruction": "",
                    "sub_tasks": [],
                    "overall_strategy": "Master answers directly",
                })
            return json.dumps({
                "mode": "hierarchical",
                "sub_tasks": [{
                    "role": "research_owner",
                    "description": "Produce the requested result",
                    "scope": "The complete request",
                    "expected_output": "Verified answer",
                    "acceptance_criteria": ["A concrete result is present"],
                    "dependencies": [],
                }],
                "overall_strategy": "One accountable Head",
                "direct_response": "",
                "direct_instruction": "",
            })
        if "Prepare a bounded execution strategy" in prompt:
            return json.dumps({
                "head_analysis": "I own integration and verification.",
                "node_assignments": [{
                    "role": "focused_worker",
                    "goal": "Collect one concrete result",
                    "scope": "Only the assigned result",
                    "deliverable": "One verified result",
                    "acceptance_criteria": ["Result is explicit"],
                }],
            })
        if "A user changed requirements" in prompt:
            if "master-only" in prompt:
                return json.dumps({
                    "action": "master_only",
                    "target_roles": [],
                    "guidance": "Apply at final synthesis",
                })
            return json.dumps({
                "action": "broadcast",
                "target_roles": ["research_owner"],
                "guidance": "Apply the new requirement",
            })
        if "Apply this new user guidance" in prompt:
            return json.dumps({
                "head_adjustment": "Requirement incorporated",
                "node_guidance": [],
                "resume_nodes": [],
                "additional_nodes": [],
            })
        if "Act as the accountable owner" in prompt:
            return json.dumps({
                "status": "completed",
                "summary": "Head verified and integrated the result.",
                "evidence": ["Node returned a concrete result"],
                "unresolved": [],
                "follow_up_tasks": [],
            })
        if "Produce the final response addressed directly" in prompt:
            return "FINAL REVISED" if "master-only" in prompt else "FINAL"
        return json.dumps({"action": "defer", "reason": "No expansion needed"})

    async def chat_with_tools(self, messages, tools, **kwargs):
        self.node_started.set()
        await self.node_release.wait()
        return _completion(json.dumps({
            "status": "completed",
            "summary": "Node produced the concrete result.",
            "evidence": ["deterministic evidence"],
            "unresolved": [],
        }))


class RepeatingDiscoveryLLM:
    def __init__(self):
        self.counter = 0

    async def chat_with_tools(self, messages, tools, **kwargs):
        self.counter += 1
        call = SimpleNamespace(
            id=f"call_{self.counter}",
            function=SimpleNamespace(
                name="framework_report_discovery",
                arguments=json.dumps({
                    "description": "same discovery",
                    "impact": "low",
                }),
            ),
        )
        return _completion(tool_calls=[call])

    async def chat_text(self, messages, **kwargs):
        return "unused"


class AwarenessCaptureLLM:
    def __init__(self):
        self.calls = []

    async def chat_with_tools(self, messages, tools, **kwargs):
        self.calls.append(list(messages))
        return _completion(json.dumps({
            "status": "completed",
            "summary": "captured",
            "evidence": [],
            "unresolved": [],
        }))

    async def chat_text(self, messages, **kwargs):
        self.calls.append(list(messages))
        return "captured"


class DirectToolLLM:
    def __init__(self):
        self.tool_turns = 0

    async def chat_text(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        if "routing work for a Master Agent" in prompt:
            return json.dumps({
                "mode": "direct",
                "reason": "One tool call is sufficient",
                "direct_response": "",
                "direct_instruction": "Call echo once and answer",
                "sub_tasks": [],
                "overall_strategy": "Master direct tool use",
            })
        return "TOOL RESULT DELIVERED"

    async def chat_with_tools(self, messages, tools, **kwargs):
        self.tool_turns += 1
        if self.tool_turns == 1:
            call = SimpleNamespace(
                id="call_echo",
                function=SimpleNamespace(
                    name="echo",
                    arguments=json.dumps({"text": "payload"}),
                ),
            )
            return _completion(tool_calls=[call])
        call = SimpleNamespace(
            id="call_finish",
            function=SimpleNamespace(
                name="framework_commit_response",
                arguments=json.dumps({"response": "TOOL RESULT DELIVERED"}),
            ),
        )
        return _completion(tool_calls=[call])


class ProgressThenFinishLLM:
    def __init__(self):
        self.tool_turns = 0

    async def chat_text(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        if "routing work for a Master Agent" in prompt:
            return json.dumps({
                "mode": "direct",
                "reason": "Master can inspect this directly",
                "direct_response": "",
                "direct_instruction": "Inspect and report",
                "sub_tasks": [],
                "overall_strategy": "Direct inspection",
            })
        return json.dumps({"response": "FALLBACK", "complete": True})

    async def chat_with_tools(self, messages, tools, **kwargs):
        self.tool_turns += 1
        assert any(
            item["function"]["name"] == "framework_commit_response"
            for item in tools
        )
        if self.tool_turns == 1:
            return _completion("再读取测试文件和剩余核心文件：")
        call = SimpleNamespace(
            id="call_finish_after_progress",
            function=SimpleNamespace(
                name="framework_commit_response",
                arguments=json.dumps({"response": "完整审查结果"}, ensure_ascii=False),
            ),
        )
        return _completion(tool_calls=[call])


class ForcedHierarchyLLM(DeterministicLLM):
    async def chat_text(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        if "routing work for a Master Agent" in prompt:
            self.chat_text_calls += 1
            if "Additional user requirements:" not in prompt:
                return json.dumps({
                    "mode": "direct",
                    "reason": "Greeting",
                    "direct_response": "初始回复",
                    "direct_instruction": "",
                    "sub_tasks": [],
                    "overall_strategy": "Direct greeting",
                })
            # Deliberately violate the explicit preference. The decomposer must repair it.
            return json.dumps({
                "mode": "direct",
                "reason": "Incorrect direct downgrade",
                "direct_response": "",
                "direct_instruction": "Review directly",
                "sub_tasks": [],
                "overall_strategy": "Invalid",
            })
        if "Repair an invalid routing decision" in prompt:
            self.chat_text_calls += 1
            return json.dumps({
                "mode": "hierarchical",
                "reason": "User explicitly required multiple agents",
                "direct_response": "",
                "direct_instruction": "",
                "sub_tasks": [{
                    "role": "code_review_owner",
                    "description": "Review the requested repository",
                    "scope": "Read-only code review",
                    "expected_output": "Evidence-backed findings",
                    "acceptance_criteria": ["Core files and tests are reviewed"],
                    "dependencies": [],
                }],
                "overall_strategy": "Head-owned review with bounded Node work",
            })
        return await super().chat_text(messages, **kwargs)


class RefusesForcedHierarchyLLM(DeterministicLLM):
    async def chat_text(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        if (
            "routing work for a Master Agent" in prompt
            or "Repair an invalid routing decision" in prompt
        ):
            self.chat_text_calls += 1
            forced = (
                "Delegation requirement: FORCED" in prompt
                or "Repair an invalid routing decision" in prompt
            )
            return json.dumps({
                "mode": "direct",
                "reason": "deliberately ignores forced delegation",
                "direct_response": "HELLO" if not forced else "",
                "direct_instruction": "work directly",
                "sub_tasks": [],
                "overall_strategy": "invalid direct response",
            })
        return await super().chat_text(messages, **kwargs)


class DirectRevisionLLM:
    def __init__(self):
        self.route_count = 0

    async def chat_text(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        if "routing work for a Master Agent" in prompt:
            self.route_count += 1
            revised = "Additional user requirements:" in prompt
            return json.dumps({
                "mode": "direct",
                "reason": "The Master owns this coherent request",
                "direct_response": "DIRECT TWO" if revised else "DIRECT ONE",
                "direct_instruction": "",
                "sub_tasks": [],
                "overall_strategy": "Master answers directly",
            })
        return "unused"

    async def chat_with_tools(self, messages, tools, **kwargs):
        return _completion("unused")


class DirectMemoryLLM:
    def __init__(self):
        self.route_messages: list[list[dict]] = []

    async def chat_text(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        if "routing work for a Master Agent" in prompt:
            self.route_messages.append(list(messages))
            first = "Research StateELF architecture" in prompt
            return json.dumps({
                "mode": "direct",
                "reason": "The Master can answer from available evidence",
                "direct_response": (
                    "StateELF uses token-causal attention and recurrent summaries."
                    if first else "Prior StateELF work was recalled."
                ),
                "direct_instruction": "",
                "sub_tasks": [],
                "overall_strategy": "Direct answer",
            })
        return "unused"

    async def chat_with_tools(self, messages, tools, **kwargs):
        return _completion("unused")


def _make_master(tmp_path: Path, fake) -> MasterAgent:
    master = MasterAgent(
        LLMConfig(api_key="test", model="fake"),
        memory_root=str(tmp_path / "memory"),
        run_root=str(tmp_path / "runs"),
        workspace_root=str(tmp_path),
    )
    master.llm_client = fake
    master.task_decomposer.llm_client = fake
    master._executor_client = fake
    return master


def test_end_to_end_run_is_retained_and_explicitly_archived(tmp_path: Path):
    async def scenario():
        fake = DeterministicLLM()
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("Do the work")
        result = await master.wait_for_run(run_id)

        assert result == "FINAL"
        state = await master.get_run_state(run_id)
        assert state["status"] == RunStatus.COMPLETED_RETAINED.value
        assert len(state["live_agents"]) == 3  # Master, Head, and retained Node
        assert master.agent_registry.count == 3
        head = next(iter(master._run_heads[run_id].values()))
        assert head.contract.context["user_request"] == "Do the work"
        node = next(iter(head._node_agents.values()))
        assert node.contract.context["user_request"] == "Do the work"
        assert "<current_task_contract>" in head.system_prompt
        assert "Do the work" in node.system_prompt
        assert "Permission mode: auto" in node.system_prompt
        head_awareness = head.runtime_awareness()
        node_awareness = node.runtime_awareness()
        assert '"level": "Head"' in head_awareness
        assert '"parent_master_id"' in head_awareness
        assert '"node_slots_remaining"' in head_awareness
        assert '"level": "Node"' in node_awareness
        assert '"parent_head_id"' in node_awareness
        assert '"business_tool_call_limit": null' in node_awareness
        run_dir = tmp_path / "runs" / run_id
        assert (run_dir / "events.jsonl").exists()
        assert len(list((run_dir / "agents").glob("*.json"))) == 3

        await master.archive_run(run_id)
        assert master.agent_registry.count == 1
        archived = await master.get_run_state(run_id)
        assert archived["status"] == RunStatus.ARCHIVED.value
        assert len(archived["topology"]["heads"][0]["nodes"]) == 1
        archived_master = await master.get_agent_snapshot(run_id, master.id)
        assert archived_master["final_response"] == "FINAL"

    asyncio.run(scenario())


def test_router_keeps_greeting_at_master_without_delegation(tmp_path: Path):
    async def scenario():
        fake = DeterministicLLM()
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("你好")
        result = await asyncio.wait_for(master.wait_for_run(run_id), timeout=1)

        assert result == "你好！有什么我可以帮你的吗？"
        assert fake.chat_text_calls == 1
        assert fake.route_messages[0]["role"] == "system"
        assert len(fake.route_messages[0]["content"]) > 14_000
        assert "<delegation_policy>" in fake.route_messages[0]["content"]
        assert "Bash, Read, Write, Edit, Glob, Grep" in fake.route_messages[0]["content"]
        assert fake.route_messages[1]["role"] == "system"
        assert "<runtime_awareness>" in fake.route_messages[1]["content"]
        assert '"level": "Master"' in fake.route_messages[1]["content"]
        assert '"phase"' in fake.route_messages[1]["content"]
        assert '"queued_user_revisions": 0' in fake.route_messages[1]["content"]
        state = await master.get_run_state(run_id)
        assert state["status"] == RunStatus.COMPLETED_RETAINED.value
        assert state["topology"]["heads"] == []
        assert state["live_agents"] == [master.id]
        events = await master.get_run_events(run_id)
        assert events[-1]["type"] == "run_checkpoint_committed"

    asyncio.run(scenario())


def test_runtime_awareness_refreshes_each_turn_without_polluting_context(tmp_path: Path):
    async def scenario():
        router = Router()
        router.register("master", AgentRole.MASTER)
        router.register("head", AgentRole.HEAD, "master", run_id="aware-run")
        node_channel = router.register(
            "node", AgentRole.NODE, "head", run_id="aware-run"
        )
        fake = AwarenessCaptureLLM()
        node = NodeAgent(
            agent_id="node",
            role="focused_reader",
            task="inspect one file",
            head_id="head",
            llm_client=fake,
            router=router,
            channel=node_channel,
            tool_registry=ToolRegistry(),
            contract=TaskContract(
                role="focused_reader",
                goal="inspect one file",
                scope="read-only",
                deliverable="one finding",
            ),
            budget=AgentBudget(
                max_turns=4,
                max_peer_messages=2,
                max_discoveries=1,
                max_children=0,
                max_revision_rounds=0,
                timeout_seconds=5,
            ),
            run_id="aware-run",
        )
        node._running = True
        node._phase = "contract_execution"
        node.start_clock()

        await node.think_with_tools("first", [])
        node._tool_call_count = 7
        node._sent_counts[MessageType.PEER_REQUEST] = 1
        node.add_runtime_observation("tool Read", "failed", "path was missing")
        node._started_at -= 4.5
        await node.think_with_tools("second", [])

        first = fake.calls[0][0]["content"]
        second = fake.calls[1][0]["content"]
        assert "<runtime_awareness>" in first
        assert '"remaining_turns_after_this_call": 3' in first
        assert '"peer_messages_remaining": 2' in first
        assert '"remaining_turns_after_this_call": 2' in second
        assert '"peer_messages_remaining": 1' in second
        assert '"business_tool_calls_observed": 7' in second
        assert "tool Read: failed (path was missing)" in second
        assert "critical: stop exploration and synthesize the best result now" in second
        assert "max_tool_calls" not in second

        retained = node.context.export_state()
        assert '"elapsed_seconds":' not in retained["system_prompt"]
        assert all(
            '"elapsed_seconds":' not in str(message.get("content", ""))
            for message in retained["messages"]
        )

    asyncio.run(scenario())


def test_simple_tool_task_stays_with_master(tmp_path: Path):
    async def scenario():
        fake = DirectToolLLM()
        master = _make_master(tmp_path, fake)

        async def echo(text: str):
            return {"echo": text}

        master.tool_registry.register(
            "echo",
            "Echo a value",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            category="utility",
            callable_fn=echo,
        )
        run_id = await master.start_request("Use the echo tool once")
        result = await asyncio.wait_for(master.wait_for_run(run_id), timeout=2)

        assert result == "TOOL RESULT DELIVERED"
        state = await master.get_run_state(run_id)
        assert state["topology"]["heads"] == []
        events = await master.get_run_events(run_id)
        assert any(
            item["type"] == "tool_call" and item["payload"].get("owner") == "master"
            for item in events
        )

    asyncio.run(scenario())


def test_master_does_not_commit_progress_narration_as_final_response(tmp_path: Path):
    async def scenario():
        fake = ProgressThenFinishLLM()
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("审查代码")
        result = await asyncio.wait_for(master.wait_for_run(run_id), timeout=2)

        assert result == "完整审查结果"
        assert fake.tool_turns == 2
        state = await master.get_run_state(run_id)
        assert state["final_response"] == "完整审查结果"
        assert state["checkpoints"][0]["response"] == "完整审查结果"

    asyncio.run(scenario())


def test_tool_calls_have_no_arbitrary_cumulative_budget(tmp_path: Path):
    async def scenario():
        master = _make_master(tmp_path, DeterministicLLM())

        async def echo(value: int):
            return {"value": value}

        master.tool_registry.register(
            "counted_echo",
            "Return one integer",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            callable_fn=echo,
        )
        for value in range(40):
            call = SimpleNamespace(function=SimpleNamespace(
                name="counted_echo",
                arguments=json.dumps({"value": value}),
            ))
            result = json.loads(await master._execute_master_tool_call("no-run", call))
            assert result == {"value": value}
        assert "max_tool_calls" not in AgentBudget().model_dump()

    asyncio.run(scenario())


def test_completed_master_only_run_can_be_revised_without_creating_agents(tmp_path: Path):
    async def scenario():
        fake = DirectRevisionLLM()
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("Give me one direct answer")

        assert await master.wait_for_run(run_id) == "DIRECT ONE"
        agent_ids = set(master.agent_registry.get_all_ids())
        assert agent_ids == {master.id}

        revision = await master.steer(run_id, "Now revise that answer")
        assert revision == 1
        assert await asyncio.wait_for(master.wait_for_run(run_id), timeout=2) == "DIRECT TWO"
        assert set(master.agent_registry.get_all_ids()) == agent_ids
        state = await master.get_run_state(run_id)
        assert state["revision"] == 1
        assert state["topology"]["heads"] == []
        assert [item["response"] for item in state["checkpoints"]] == [
            "DIRECT ONE",
            "DIRECT TWO",
        ]
        assert [item["revision"] for item in state["checkpoints"]] == [0, 1]
        assert fake.route_count == 2

    asyncio.run(scenario())


def test_explicit_multi_agent_revision_creates_visible_live_topology(tmp_path: Path):
    async def scenario():
        fake = ForcedHierarchyLLM(gate_node=True)
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("你好，介绍一下你自己")
        assert await master.wait_for_run(run_id) == "初始回复"

        revision = await master.steer(
            run_id,
            "调用多agent，审查代码，只读不要做任何修改",
        )
        assert revision == 1
        await asyncio.wait_for(fake.node_started.wait(), timeout=2)

        live = await master.get_run_state(run_id)
        assert live["status"] == RunStatus.RUNNING.value
        assert len(live["topology"]["heads"]) == 1
        assert len(live["topology"]["heads"][0]["nodes"]) == 1
        assert len(live["agent_ids"]) == 3

        fake.node_release.set()
        assert await asyncio.wait_for(master.wait_for_run(run_id), timeout=5) == "FINAL"
        settled = await master.get_run_state(run_id)
        assert [item["revision"] for item in settled["checkpoints"]] == [0, 1]
        assert [item["response"] for item in settled["checkpoints"]] == [
            "初始回复",
            "FINAL",
        ]
        events = await master.get_run_events(run_id)
        assert sum(item["type"] == "agent_registered" for item in events) == 3

    asyncio.run(scenario())


def test_explicit_multi_agent_uses_bounded_fallback_if_planner_refuses(tmp_path: Path):
    async def scenario():
        fake = RefusesForcedHierarchyLLM()
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("你好")
        # This fake deliberately returns an invalid direct routing decision, so the
        # initial AUTO request remains Master-only.
        assert await master.wait_for_run(run_id) != ""

        await master.steer(run_id, "调用多agent仔细审查模型结构和训练代码")
        result = await asyncio.wait_for(master.wait_for_run(run_id), timeout=5)

        assert result == "FINAL"
        state = await master.get_run_state(run_id)
        assert state["status"] == RunStatus.COMPLETED_RETAINED.value
        assert len(state["topology"]["heads"]) == 2
        roles = [head["role"] for head in state["topology"]["heads"]]
        assert roles == ["primary_owner", "independent_verifier"]
        assert all(len(head["nodes"]) == 1 for head in state["topology"]["heads"])
        verifier = next(
            head for head in master._run_heads[run_id].values()
            if head.role == "independent_verifier"
        )
        assert "primary_owner" in verifier.contract.dependencies
        assert verifier.contract.context["dependency_outcomes"]["primary_owner"]

    asyncio.run(scenario())


def test_failed_revision_preserves_checkpoint_and_exposes_error(tmp_path: Path):
    async def scenario():
        journal = RunJournal("original", str(tmp_path / "runs"), run_id="failed-run")
        await journal.initialize()
        await journal.commit_checkpoint("GOOD CHECKPOINT", reason="test")
        await journal.add_requirement("revision")
        await journal.set_status(RunStatus.RUNNING)
        await journal.set_status(RunStatus.FAILED_RETAINED, error="revision failed")

        state = await journal.read_state()
        assert state["last_error"] == "revision failed"
        assert state["final_response"] == "GOOD CHECKPOINT"
        assert state["checkpoints"][0]["response"] == "GOOD CHECKPOINT"

        # Simulate the pre-last_error manifest format and verify read-time migration.
        journal.manifest.last_error = ""
        journal.manifest.final_response = "legacy revision failure"
        migrated = await journal.read_state()
        assert migrated["last_error"] == "legacy revision failure"
        assert migrated["final_response"] == "GOOD CHECKPOINT"

    asyncio.run(scenario())


def test_explicit_multi_agent_detection_is_a_control_directive():
    assert TaskDecomposer.explicit_hierarchy_requested("调用多agent，审查代码")
    assert TaskDecomposer.explicit_hierarchy_requested("Please use multiple agents to review")
    assert not TaskDecomposer.explicit_hierarchy_requested("介绍一下多agent系统的设计")
    assert not TaskDecomposer.explicit_hierarchy_requested("不要调用多agent")


def test_running_run_accepts_user_update(tmp_path: Path):
    async def scenario():
        fake = DeterministicLLM(gate_node=True)
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("Do the work")
        await asyncio.wait_for(fake.node_started.wait(), timeout=2)
        revision = await master.steer(run_id, "Add an extra requirement")
        assert revision == 1
        # Give Master and Head a chance to route the revision before Node completes.
        await asyncio.sleep(0.35)
        fake.node_release.set()
        result = await asyncio.wait_for(master.wait_for_run(run_id), timeout=5)

        assert result == "FINAL"
        state = await master.get_run_state(run_id)
        assert state["revision"] == 1
        assert state["requirements"][-1] == "Add an extra requirement"
        head = next(iter(master._run_heads[run_id].values()))
        assert 1 in head._applied_revisions

    asyncio.run(scenario())


def test_completed_run_can_resume_without_recreating_agents(tmp_path: Path):
    async def scenario():
        fake = DeterministicLLM()
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("Do the work")
        await master.wait_for_run(run_id)
        agent_ids = set(master.agent_registry.get_all_ids())

        await master.steer(run_id, "master-only: make the final concise")
        revised = await asyncio.wait_for(master.wait_for_run(run_id), timeout=5)
        assert revised == "FINAL REVISED"
        assert set(master.agent_registry.get_all_ids()) == agent_ids
        assert (await master.get_run_state(run_id))["revision"] == 1

    asyncio.run(scenario())


def test_persisted_runs_reload_as_inspectable_cold_snapshots(tmp_path: Path):
    async def scenario():
        run_root = tmp_path / "runs"
        first = _make_master(tmp_path, DeterministicLLM())
        run_id = await first.start_request("Persist this work")
        await first.wait_for_run(run_id)

        second = _make_master(tmp_path, DeterministicLLM())
        states = await second.list_run_states()
        restored = next(item for item in states if item["run_id"] == run_id)
        assert restored["retention"] == "snapshot"
        assert restored["resumable"] is False
        assert len(restored["topology"]["heads"][0]["nodes"]) == 1
        assert await second.get_agent_snapshot(
            run_id,
            restored["topology"]["heads"][0]["nodes"][0]["agent_id"],
        )
        try:
            await second.steer(run_id, "Continue after restart")
        except RuntimeError as error:
            assert "inspectable but not resumable" in str(error)
        else:
            raise AssertionError("cold Run incorrectly resumed without live agents")

    asyncio.run(scenario())


def test_long_term_memory_backfills_old_runs_and_recalls_direct_checkpoints(tmp_path: Path):
    async def scenario():
        run_root = tmp_path / "runs"
        old = RunJournal(
            "调用多agent调研 /workspace/StateELF",
            str(run_root),
            run_id="run_stateelf_old",
        )
        await old.initialize()
        await old.commit_checkpoint(
            "StateELF uses token-level causal queries while block_size controls summary writes.",
            reason="historical_analysis",
        )
        failed_recall = RunJournal(
            "Do you remember the StateELF work?",
            str(run_root),
            run_id="run_stateelf_failed_recall",
        )
        await failed_recall.initialize()
        await failed_recall.commit_checkpoint(
            "I cannot find any prior record; please provide more context.",
            reason="bad_recall",
        )

        fake = DirectMemoryLLM()
        master = _make_master(tmp_path, fake)
        await master._ensure_memory_index()
        matches = await master.long_term_memory.search("还记得 StateELF 的设计吗")

        assert matches[0]["run_id"] == "run_stateelf_old"
        assert matches[0]["source_path"].endswith("run_stateelf_old")
        assert "token-level causal queries" in matches[0]["response_excerpt"]
        assert {"Recall", "ReadRun"}.issubset(master.get_tool_info()["builtin"])

        run_id = await master.start_request("Research StateELF architecture")
        assert "token-causal attention" in await master.wait_for_run(run_id)
        recalled = await master.tool_executor.execute(
            "Recall",
            {"query": "StateELF", "limit": 5, "max_chars": 8_000},
        )
        assert recalled["match_count"] >= 2
        expanded = await master.tool_executor.execute(
            "ReadRun",
            {"run_id": run_id, "max_chars": 10_000},
        )
        assert expanded["run_id"] == run_id
        assert "token-causal attention" in expanded["response_excerpt"]

        second_id = await master.start_request("Do you remember the StateELF work?")
        assert await master.wait_for_run(second_id) == "Prior StateELF work was recalled."
        routed_context = fake.route_messages[-1]
        memory_message = next(
            message["content"] for message in routed_context
            if str(message.get("content", "")).startswith("<retrieved_long_term_memory>")
        )
        assert run_id in memory_message
        assert "StateELF uses token-causal attention" in memory_message
        events = await master.get_run_events(second_id)
        assert any(item["type"] == "memory_retrieved" for item in events)

    asyncio.run(scenario())


def test_long_term_memory_keeps_failed_partial_evidence_and_redacts_secrets(tmp_path: Path):
    async def scenario():
        journal = RunJournal(
            "Investigate failure-topic with ghp_abcdefghijklmnopqrstuvwxyz123456",
            str(tmp_path / "runs"),
            run_id="run_partial_memory",
        )
        await journal.initialize()
        await journal.set_status(RunStatus.FAILED_RETAINED, error="Training diverged at step 42")
        memory = LongTermMemory(str(tmp_path / "memory"))
        await memory.remember_run(
            journal,
            workspace_root=tmp_path,
            outcomes=[{
                "agent_id": "head_partial",
                "task_id": "task_partial",
                "status": "partial",
                "summary": "Useful gradients were inspected.",
                "evidence": ["loss became non-finite"],
                "unresolved": ["root cause remains unknown"],
            }],
        )
        # Startup backfill contains no derived Head outcomes. It must refresh the
        # journal fields without erasing richer evidence written at Run completion.
        await memory.backfill_runs([journal], tmp_path)

        matches = await memory.search("failure-topic")
        assert matches[0]["status"] == RunStatus.FAILED_RETAINED.value
        assert "Training diverged at step 42" in matches[0]["unresolved"]
        assert "root cause remains unknown" in matches[0]["unresolved"]
        assert matches[0]["agent_outcomes"][0]["agent_id"] == "head_partial"
        assert "[REDACTED]" in matches[0]["request"]
        assert "ghp_" not in matches[0]["request"]

    asyncio.run(scenario())


def test_cancel_retains_partial_agent_state_and_stops_children(tmp_path: Path):
    async def scenario():
        fake = DeterministicLLM(gate_node=True)
        master = _make_master(tmp_path, fake)
        run_id = await master.start_request("Long running work")
        await asyncio.wait_for(fake.node_started.wait(), timeout=2)
        await master.cancel_run(run_id)

        state = await master.get_run_state(run_id)
        assert state["status"] == RunStatus.CANCELLED_RETAINED.value
        assert not master._run_active_head_tasks[run_id]
        assert len(list((tmp_path / "runs" / run_id / "agents").glob("*.json"))) == 3

    asyncio.run(scenario())


def test_node_discovery_budget_prevents_message_storm(tmp_path: Path):
    async def scenario():
        router = Router()
        router.register("master", AgentRole.MASTER)
        head_channel = router.register("head", AgentRole.HEAD, "master", run_id="run")
        node_channel = router.register("node", AgentRole.NODE, "head", run_id="run")
        journal = RunJournal("bounded node", str(tmp_path / "runs"), run_id="run")
        await journal.initialize()
        node = NodeAgent(
            agent_id="node",
            role="worker",
            task="bounded task",
            head_id="head",
            llm_client=RepeatingDiscoveryLLM(),
            router=router,
            channel=node_channel,
            tool_registry=ToolRegistry(),
            contract=TaskContract(role="worker", goal="bounded task"),
            budget=AgentBudget(
                max_turns=3,
                max_peer_messages=0,
                max_discoveries=1,
                max_children=0,
                max_revision_rounds=0,
                timeout_seconds=2,
            ),
            run_id="run",
            journal=journal,
        )
        outcome = await asyncio.wait_for(node.run(), timeout=2)
        messages = head_channel.drain()
        assert outcome.status == OutcomeStatus.PARTIAL
        assert sum(msg.msg_type == MessageType.DISCOVERY for msg in messages) == 1
        assert sum(msg.msg_type == MessageType.REPORT for msg in messages) == 1

    asyncio.run(scenario())


def test_router_isolates_retained_runs():
    async def scenario():
        router = Router()
        router.register("master", AgentRole.MASTER)
        router.register("head_a", AgentRole.HEAD, "master", run_id="run_a")
        router.register("head_b", AgentRole.HEAD, "master", run_id="run_b")
        sent = await router.send(Message(
            sender_id="head_a",
            receiver_id="head_b",
            msg_type=MessageType.PEER_REQUEST,
            content={"question": "cross-run"},
        ))
        assert sent is False
        assert router.get_peers("head_a") == []

    asyncio.run(scenario())


def test_plan_is_bounded_and_roles_are_unique():
    raw = json.dumps({
        "sub_tasks": [
            {"role": "same", "description": str(index)} for index in range(8)
        ]
    })
    plan = TaskDecomposer._parse_plan(raw)
    roles = [item["role"] for item in plan["sub_tasks"]]
    assert len(roles) == 4
    assert len(set(roles)) == 4


def test_plan_can_keep_work_at_master_without_creating_a_head():
    plan = TaskDecomposer._parse_plan(json.dumps({
        "sub_tasks": [],
        "overall_strategy": "Master answers directly",
        "direct_response": "A direct user-facing answer",
    }))
    assert plan["sub_tasks"] == []
    assert plan["direct_response"] == "A direct user-facing answer"


def test_context_compaction_keeps_tool_call_with_results():
    manager = ContextManager(keep_recent=2)
    manager._messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
        {"role": "user", "content": "continue"},
    ]
    manager._auto_compact()
    roles = [message["role"] for message in manager._messages]
    assert roles[0] != "tool"
    if "tool" in roles:
        assert "assistant" in roles[:roles.index("tool")]


def test_ptc_module_was_removed():
    assert not (Path(__file__).parents[1] / "agent_framework" / "tools" / "ptc.py").exists()


def test_plan_cache_module_was_removed():
    memory_dir = Path(__file__).parents[1] / "agent_framework" / "memory"
    assert not (memory_dir / "plan_cache.py").exists()
    assert not (memory_dir / "memory_store.py").exists()


def test_business_tools_cannot_override_control_protocol():
    registry = ToolRegistry()
    try:
        registry.register("framework_report_discovery", "override")
    except ValueError:
        pass
    else:
        raise AssertionError("reserved framework tool prefix was accepted")


def test_deep_research_rejects_unsafe_paths_and_local_urls(tmp_path: Path):
    research_notes.STORAGE_DIR = str(tmp_path / "research")
    research_notes._SESSION_ID = None
    try:
        research_notes.set_session("../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("path-traversing session id was accepted")

    research_report._REPORTS_DIR = tmp_path / "reports"
    outside = tmp_path / "outside.md"
    outside.write_text("do not expose")
    assert "outside the managed reports directory" in research_report.read_report(str(outside))

    result = json.loads(research_fetch.fetch_webpage("http://127.0.0.1/private"))
    assert "not allowed" in result["error"]


def test_keyless_web_search_normalizes_results_and_domain_filters(monkeypatch):
    html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.com%2Fguide">Guide</a>
      <div class="result__snippet">Useful documentation.</div>
    </div>
    <div class="result">
      <a class="result__a" href="https://blocked.example.net/post">Blocked</a>
      <div class="result__snippet">Should not be returned.</div>
    </div>
    """
    response = research_search.httpx.Response(
        200,
        text=html,
        request=research_search.httpx.Request("GET", "https://search.test"),
    )
    monkeypatch.setattr(research_search.httpx, "get", lambda *args, **kwargs: response)

    result = json.loads(research_search.search_web(
        "python docs",
        allowed_domains=["example.com"],
    ))
    assert result["query"] == "python docs"
    assert result["total_results"] == 1
    assert result["results"][0]["url"] == "https://docs.example.com/guide"
    assert result["results"][0]["metadata"]["source"] == "duckduckgo"
    conflict = json.loads(research_search.search_web(
        "python docs",
        allowed_domains=["example.com"],
        blocked_domains=["example.net"],
    ))
    assert "cannot be used together" in conflict["error"]


def test_web_search_is_builtin_and_user_mcp_tools_are_namespaced(tmp_path: Path):
    async def scenario():
        master = _make_master(tmp_path, DeterministicLLM())
        assert {"WebSearch", "WebFetch"}.issubset(master.get_tool_info()["builtin"])
        result = await master.tool_executor.execute(
            "WebFetch",
            {"url": "http://127.0.0.1/private"},
        )
        assert "not allowed" in result

        names = await master.connect_mcp_server(MCPServerConfig(
            name="test-extra",
            command=sys.executable,
            args=(str(Path(__file__).parents[1] / "mcp_server.py"),),
            cwd=Path(__file__).parents[1],
            tool_allowlist=frozenset({"tool_get_session"}),
            category="mcp:test-extra",
        ))
        try:
            assert names == ["mcp__test_extra__tool_get_session"]
            tool = master.tool_registry.get(names[0])
            assert tool is not None
            assert tool.source == "mcp:test-extra"
            assert tool.permission_scope == "mcp"
            call_task = asyncio.create_task(master.tool_executor.execute(names[0], {}))
            await asyncio.sleep(0)
            approval = master.list_pending_approvals()[0]
            assert approval["tool_name"] == names[0]
            assert approval["risk"] == "mcp"
            await master.resolve_approval(approval["approval_id"], False)
            denied = await call_task
            assert "denied by the user" in denied["error"]
        finally:
            await master.disconnect_mcp_server("test-extra")
        assert not master.get_tool_info()["mcp"]

    asyncio.run(scenario())


def test_permission_gate_approves_or_denies_one_concrete_tool_call(tmp_path: Path):
    async def scenario():
        master = _make_master(tmp_path, DeterministicLLM())
        master.set_permission_mode("ask")
        journal = RunJournal("permission test", root_dir=tmp_path / "approval-runs")
        await journal.initialize()
        context = ApprovalContext(
            run_id=journal.run_id,
            agent_id=master.id,
            task_id="permission-task",
            journal=journal,
        )

        approved_path = tmp_path / "approved.txt"
        approved_task = asyncio.create_task(master.tool_executor.execute(
            "Write",
            {"path": str(approved_path), "content": "approved"},
            context,
        ))
        await asyncio.sleep(0)
        pending = master.list_pending_approvals()
        assert len(pending) == 1
        assert not approved_path.exists()
        await master.resolve_approval(pending[0]["approval_id"], True)
        approved_result = await approved_task
        assert approved_result["bytes_written"] == 8
        assert approved_path.read_text() == "approved"

        denied_path = tmp_path / "denied.txt"
        denied_task = asyncio.create_task(master.tool_executor.execute(
            "Write",
            {"path": str(denied_path), "content": "denied"},
            context,
        ))
        await asyncio.sleep(0)
        denied = master.list_pending_approvals()[0]
        await master.resolve_approval(denied["approval_id"], False)
        denied_result = await denied_task
        assert "denied by the user" in denied_result["error"]
        assert not denied_path.exists()

        events = await journal.read_events()
        assert [item["type"] for item in events].count("approval_requested") == 2
        assert [item["type"] for item in events].count("approval_resolved") == 2

        master.set_permission_mode("auto")
        automatic_path = tmp_path / "automatic.txt"
        automatic = await master.tool_executor.execute(
            "Write",
            {"path": str(automatic_path), "content": "workspace write"},
            context,
        )
        assert automatic["bytes_written"] == 15
        assert master.list_pending_approvals() == []

        shell_path = tmp_path / "shell.txt"
        shell_task = asyncio.create_task(master.tool_executor.execute(
            "Bash",
            {"command": f"touch {shell_path}"},
            context,
        ))
        await asyncio.sleep(0)
        shell_approval = master.list_pending_approvals()[0]
        assert shell_approval["risk"] in {"shell", "unsafe_shell"}
        await master.resolve_approval(shell_approval["approval_id"], False)
        await shell_task
        assert not shell_path.exists()

        master.set_permission_mode("full")
        full_result = await master.tool_executor.execute(
            "Bash",
            {"command": f"touch {shell_path}"},
            context,
        )
        assert full_result["exit_code"] == 0
        assert shell_path.exists()
        assert "Permission mode: full" in master.system_prompt

    asyncio.run(scenario())


def test_builtin_coding_tools_cover_read_edit_glob_grep_and_skill(tmp_path: Path):
    async def scenario():
        master = _make_master(tmp_path, DeterministicLLM())
        source = tmp_path / "src" / "sample.py"
        written = await master.tool_executor.execute(
            "Write",
            {"path": str(source), "content": "VALUE = 1\nprint(VALUE)\n"},
        )
        assert written["bytes_written"] > 0

        edited = await master.tool_executor.execute(
            "Edit",
            {
                "path": str(source),
                "old_string": "VALUE = 1",
                "new_string": "VALUE = 2",
            },
        )
        assert edited["replacements"] == 1
        read = await master.tool_executor.execute(
            "Read", {"path": str(source), "offset": 1, "limit": 10}
        )
        assert "VALUE = 2" in read["content"]
        assert read["content"].startswith("     1\t")

        matches = await master.tool_executor.execute(
            "Glob", {"pattern": "**/*.py", "path": str(tmp_path)}
        )
        assert str(source) in matches["matches"]
        grep = await master.tool_executor.execute(
            "Grep",
            {"pattern": r"print\(VALUE\)", "path": str(tmp_path), "glob": "*.py"},
        )
        assert grep["matched_files"] == 1
        assert grep["results"][0]["line"] == 2

        skill_file = tmp_path / ".synapse" / "skills" / "review" / "SKILL.md"
        await master.tool_executor.execute(
            "Write",
            {
                "path": str(skill_file),
                "content": (
                    "---\n"
                    "name: review\n"
                    "description: Review the current code diff for defects.\n"
                    "allowed-tools: [Read, Grep, Bash]\n"
                    "---\n"
                    "# Review\nInspect the current diff. Focus: $ARGUMENTS. First: $0."
                ),
            },
        )
        skills = await master.tool_executor.execute("Skill", {"name": "list"})
        assert skills["skills"][0]["name"] == "review"
        assert skills["skills"][0]["description"].startswith("Review the current")
        skill = await master.tool_executor.execute(
            "Skill", {"name": "review", "arguments": "focus on correctness"}
        )
        assert "Inspect the current diff" in skill["instructions"]
        assert "Focus: focus on correctness" in skill["instructions"]
        assert "First: focus" in skill["instructions"]
        master.set_permission_mode("auto")
        assert '<skill name="review"' in master.system_prompt

        pwd = await master.tool_executor.execute("Bash", {"command": "pwd"})
        assert pwd["exit_code"] == 0
        assert pwd["stdout"].strip() == str(tmp_path)
        assert master.list_pending_approvals() == []

    asyncio.run(scenario())


def test_skill_catalog_uses_real_skill_metadata_and_explicit_precedence(tmp_path: Path):
    low = tmp_path / "low"
    high = tmp_path / "high"
    for root, marker in ((low, "LOW"), (high, "HIGH")):
        path = root / "review" / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            "---\n"
            "name: review\n"
            f"description: {marker} precedence review workflow.\n"
            "---\n"
            f"{marker}: inspect $0 with ${{SKILL_DIR}}. All args: $ARGUMENTS",
            encoding="utf-8",
        )

    disabled = high / "manual" / "SKILL.md"
    disabled.parent.mkdir(parents=True)
    disabled.write_text(
        "---\n"
        "name: manual\n"
        "description: A user-only workflow.\n"
        "disable-model-invocation: true\n"
        "---\n"
        "Do the manual operation.",
        encoding="utf-8",
    )

    catalog = SkillCatalog(tmp_path, [low, high])
    listed = {item["name"]: item for item in catalog.list_skills()}
    assert listed["review"]["description"].startswith("HIGH")
    assert listed["review"]["source"] == "configured"
    assert listed["manual"]["model_invocable"] is False
    assert '<skill name="review"' in catalog.model_inventory()
    assert '<skill name="manual"' not in catalog.model_inventory()

    loaded = catalog.load("review", 'src/main.py "strict mode"')
    assert loaded["instructions"].startswith("HIGH: inspect src/main.py")
    assert "All args: src/main.py \"strict mode\"" in loaded["instructions"]
    assert loaded["base_directory"] == str((high / "review").resolve())
    assert "error" in catalog.load("manual")


def test_web_control_room_runs_against_the_real_runtime_api(tmp_path: Path):
    web_module = importlib.import_module("agent_framework.web.app")
    previous_master = web_module.master
    web_module.master = _make_master(tmp_path, DeterministicLLM())
    try:
        with TestClient(web_module.app) as client:
            index_response = client.get("/")
            assert index_response.status_code == 200
            assert "Synapse" in index_response.text
            assert 'id="guidanceThread"' in index_response.text
            assert "MASTER CHECKPOINTS" in index_response.text
            assert 'id="approvalCount"' in index_response.text
            assert 'id="activityToggleButton"' in index_response.text
            assert 'id="activityPanel"' in index_response.text
            assert 'id="approvalList"' not in index_response.text
            assert index_response.text.index('id="approvalCount"') > index_response.text.index("EVENT STREAM")
            script_response = client.get("/static/app.js")
            assert script_response.status_code == 200
            assert "pendingSteersFor" in script_response.text
            assert "checkpoint-message" in script_response.text
            assert "recoverMissingRun" in script_response.text
            assert "error.status = response.status" in script_response.text
            assert "run-error-message" in script_response.text
            assert "renderPendingApproval" in script_response.text
            assert "data-approval-allow" in script_response.text
            assert "settingsModal.classList.remove" not in script_response.text.split(
                'event.type === "approval_requested"', 1
            )[1].split("function classifyEvent", 1)[0]
            health = client.get("/api/health").json()
            assert health["search"] == {
                "status": "ready", "tools": ["WebSearch", "WebFetch"]
            }
            assert health["mcp"]["servers"] == []
            assert {
                "Bash", "Read", "Write", "Edit", "Glob", "Grep", "Skill",
                "Recall", "ReadRun",
            }.issubset(
                set(health["tools"]["builtin"])
            )
            assert client.put("/api/permissions", json={"mode": "ask"}).json()["mode"] == "ask"
            assert client.get("/api/permissions").json()["pending"] == []
            assert client.get("/api/runs/run_missing").status_code == 404
            assert client.get("/api/runs/run_missing/stream").status_code == 404

            created = client.post("/api/runs", json={"request": "Do the web work"})
            assert created.status_code == 202
            run_id = created.json()["run_id"]

            state = {}
            for _ in range(60):
                state = client.get(f"/api/runs/{run_id}").json()
                if state.get("status") == RunStatus.COMPLETED_RETAINED.value:
                    break
                time.sleep(0.05)

            assert state["status"] == RunStatus.COMPLETED_RETAINED.value
            assert len(state["topology"]["heads"]) == 1
            events = client.get(f"/api/runs/{run_id}/events").json()["events"]
            assert events

            node_id = state["topology"]["heads"][0]["nodes"][0]["agent_id"]
            snapshot = client.get(f"/api/runs/{run_id}/agents/{node_id}")
            assert snapshot.status_code == 200
            assert snapshot.json()["agent_id"] == node_id

            revised = client.post(
                f"/api/runs/{run_id}/steer",
                json={"requirement": "master-only: make the final concise"},
            )
            assert revised.status_code == 202
            assert revised.json()["revision"] == 1
            revised_state = client.get(f"/api/runs/{run_id}").json()
            assert revised_state["requirements"][-1] == "master-only: make the final concise"
            revised_events = client.get(f"/api/runs/{run_id}/events").json()["events"]
            assert any(
                event["type"] == "user_update"
                and event["payload"]["text"] == "master-only: make the final concise"
                for event in revised_events
            )
            assert client.post("/api/runs", json={"request": "   "}).status_code == 422
    finally:
        web_module.master = previous_master
