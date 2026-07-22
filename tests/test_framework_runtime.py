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
from agent_framework.planning.decomposer import TaskDecomposer
from agent_framework.runtime.run import RunJournal, RunStatus
from agent_framework.tools.registry import ToolRegistry
from agent_framework.tools.executor import ToolExecutor
from agent_framework.tools.mcp_provider import MCPServerConfig, MCPToolProvider
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
        self.node_started = asyncio.Event()
        self.node_release = asyncio.Event()
        if not gate_node:
            self.node_release.set()

    async def chat_text(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        if "Extract the core intent" in prompt:
            return "runtime_test"
        if "bounded execution plan" in prompt:
            return json.dumps({
                "sub_tasks": [{
                    "role": "research_owner",
                    "description": "Produce the requested result",
                    "scope": "The complete request",
                    "expected_output": "Verified answer",
                    "acceptance_criteria": ["A concrete result is present"],
                    "dependencies": [],
                }],
                "overall_strategy": "One accountable Head",
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
        if "Produce the Master checkpoint" in prompt:
            return "FINAL REVISED" if "master-only" in prompt else "FINAL"
        if "Extract a reusable plan template" in prompt:
            return "one accountable owner followed by validation"
        if "tool indexing assistant" in prompt:
            return "# Skills Index"
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


def _make_master(tmp_path: Path, fake: DeterministicLLM) -> MasterAgent:
    master = MasterAgent(
        LLMConfig(api_key="test", model="fake"),
        memory_root=str(tmp_path / "memory"),
        cache_dir=str(tmp_path / "cache"),
        run_root=str(tmp_path / "runs"),
    )
    master.llm_client = fake
    master.task_decomposer.llm_client = fake
    master.plan_cache.llm_client = fake
    master.skills_generator.llm_client = fake
    master._lightweight_client = fake
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
                max_tool_calls=0,
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


def test_local_web_search_mcp_registers_and_executes_tools():
    async def scenario():
        registry = ToolRegistry()
        provider = MCPToolProvider(MCPServerConfig(
            name="test-web-search",
            command=sys.executable,
            args=("-m", "agent_framework.tools.web_search_server"),
            cwd=Path(__file__).parents[1],
            tool_allowlist=frozenset({"tool_search_web", "tool_fetch_webpage"}),
            category="web_search",
        ))
        try:
            names = await provider.connect(registry)
            assert set(names) == {"tool_search_web", "tool_fetch_webpage"}
            schema = registry.get("tool_search_web").parameters
            assert "allowed_domains" in schema["properties"]

            result = await ToolExecutor(registry).execute(
                "tool_fetch_webpage",
                {"url": "http://127.0.0.1/private"},
            )
            assert "not allowed" in result
        finally:
            await provider.close()
        assert registry.get_names() == []

    asyncio.run(scenario())


def test_web_control_room_runs_against_the_real_runtime_api(tmp_path: Path):
    web_module = importlib.import_module("agent_framework.web.app")
    previous_master = web_module.master
    web_module.master = _make_master(tmp_path, DeterministicLLM())
    try:
        with TestClient(web_module.app) as client:
            assert client.get("/").status_code == 200
            assert "Synapse" in client.get("/").text
            assert client.get("/static/app.js").status_code == 200
            mcp_health = client.get("/api/health").json()["mcp"]
            assert mcp_health["status"] == "connected"
            assert set(mcp_health["tools"]) == {
                "tool_search_web", "tool_fetch_webpage"
            }

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
            assert client.post("/api/runs", json={"request": "   "}).status_code == 422
    finally:
        web_module.master = previous_master
