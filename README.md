# Synapse

Synapse is a bounded hierarchical multi-agent runtime with a
**Master → Head → Node** architecture. It keeps hierarchy, peer coordination,
persistent memory, tool retrieval, and reusable plans while making ownership and
termination explicit.

## Architecture

```text
User updates ───────────────────────────────┐
                                            ▼
Master (global owner and task graph) ── Head (sub-problem owner) ── Node (executor)
             │                         ▲  │                      ▲
             └── retained Run journal ─┘  └── bounded peers ─────┘
```

- **Master** interprets evolving user requirements, owns the dependency graph and
  resource budget, arbitrates discoveries, and validates the final checkpoint.
- **Head** reasons about its sub-problem, delegates only focused work, coordinates
  with peer Heads, integrates evidence, and checks acceptance criteria. A Head may
  use zero Nodes when delegation is unnecessary.
- **Node** executes one task contract with bounded turns, tool calls, discoveries,
  and sibling request/response messages. Nodes cannot create agents or redefine the
  task graph.

Every child receives a `TaskContract` containing scope, deliverable, dependencies,
and acceptance criteria. Every exit path returns an `AgentOutcome` such as
`completed`, `partial`, `blocked`, `failed`, or `cancelled`.

## Execution properties

- **Interruptible runs**: add user requirements while work is active with
  `MasterAgent.steer()`.
- **Retained agents**: a final response is a checkpoint, not destruction. Head and
  Node contexts remain available for a later revision.
- **Durable action history**: messages, tool actions, decisions, outcomes, and agent
  context snapshots are written under `~/.agent_framework/runs/<run_id>/` by default.
- **Restart-safe inspection**: persisted Runs are rediscovered after server restart as
  read-only cold snapshots. They remain inspectable but are not presented as resumable
  live agents.
- **Bounded organization**: hard budgets limit turns, tools, peer messages,
  discoveries, child agents, revisions, and wall-clock time.
- **Lifecycle-based completion**: parents observe actual child task completion;
  duplicate or missing `REPORT` messages cannot deadlock the run.
- **Dependency scheduling**: Heads start when their declared dependencies have
  outcomes. Invalid cycles degrade explicitly instead of waiting forever.
- **Run isolation**: retained agents from old Runs cannot communicate with agents in
  a new Run.
- **Explicit cleanup**: call `archive_run()` to release live agents after their
  snapshots are persisted. A configurable retention limit archives older Runs.

Programmatic Tool Calling (PTC) is intentionally not supported. Generated Python
execution was both unstable and an unsafe expansion of Node authority; Nodes use
registered, schema-defined tools only.

## Quick start

```bash
pip install -e .
```

For a simple request/response call:

```python
import asyncio

from agent_framework.core.master_agent import MasterAgent
from agent_framework.llm.config import LLMConfig


async def main():
    config = LLMConfig(
        base_url="http://localhost:8000/v1",
        api_key="your-api-key",
        model="your-model",
    )
    master = MasterAgent(llm_config=config)
    result = await master.handle_user_request("Analyze the quarterly sales data")
    print(result)


asyncio.run(main())
```

For an interruptible and resumable Run:

```python
async def interactive(master: MasterAgent):
    run_id = await master.start_request("Analyze the quarterly sales data")

    # This may be called from another UI/event-handler coroutine while work runs.
    await master.steer(run_id, "Focus on revenue quality, not just total revenue")
    first_checkpoint = await master.wait_for_run(run_id)

    # The completed tree is retained. This resumes the same Run and existing context.
    await master.steer(run_id, "Now give me a two-paragraph executive version")
    revised_checkpoint = await master.wait_for_run(run_id)

    state = await master.get_run_state(run_id)
    print(state["storage_path"], revised_checkpoint)

    # Explicitly release live Head/Node objects; persisted history remains on disk.
    await master.archive_run(run_id)
```

`start_request()` intentionally returns immediately. A UI can keep accepting user
input and call `steer()` while `wait_for_run()` is pending.

## Web control room

The bundled control room exposes Run creation, live steering, hierarchy inspection,
the event stream, retained checkpoints, cancellation, and explicit archival.

```bash
pip install -e '.[web]'

export SYNAPSE_BASE_URL=http://localhost:8000/v1
export SYNAPSE_API_KEY=your-api-key
export SYNAPSE_PLANNER_MODEL=your-model
# Optional: SYNAPSE_EXECUTOR_MODEL and SYNAPSE_LIGHTWEIGHT_MODEL

python -m agent_framework.web.app
```

Open `http://127.0.0.1:8008`. The server binds to loopback by default because the
control room can inspect retained agent state and actions.

The Web server automatically starts the bundled keyless Web Search MCP and exposes
only `tool_search_web` and `tool_fetch_webpage` to Nodes. Search supports
`allowed_domains` and `blocked_domains`; webpage fetching rejects local and private
network targets. No search API key is required. Set
`SYNAPSE_WEB_SEARCH_MCP=disabled` to turn it off.

To attach the bundled MCP from Python instead:

```python
import sys

from agent_framework.tools.mcp_provider import MCPServerConfig

await master.connect_mcp_server(MCPServerConfig(
    name="web-search",
    command=sys.executable,
    args=("-m", "agent_framework.tools.web_search_server"),
    tool_allowlist=frozenset({"tool_search_web", "tool_fetch_webpage"}),
    category="web_search",
))
```

## Communication rules

- A Node communicates only with its Head and sibling Nodes in the same Run.
- Node peer communication is a correlated, non-blocking question/response protocol.
- A Node reports a discovery to its Head; it never requests a new Head directly.
- Heads may coordinate with peer Heads or submit one bounded discovery proposal to
  Master.
- Master alone may change the Head topology, subject to the Run budget.
- Cross-layer, cross-group, and cross-Run messages are rejected by the Router.

## Tools, plans, and memory

- Tool schemas are registered in `ToolRegistry`; the generated skills index is a
  compact retrieval catalog rather than executable code.
- Stdio MCP servers can be attached through `MasterAgent.connect_mcp_server()`;
  allowed remote tools become ordinary bounded Node tools and the MCP process stays
  alive for the runtime lifecycle.
- Plan templates are cached only after fully successful Head outcomes.
- Master and Head memory is persistent. Recent memory is injected so old append-only
  files do not permanently hide new information.
- Tool call/result pairs remain atomic during context compaction.

## Deep Research MCP server

`mcp_server.py` exposes the separate `deep_research` toolkit over stdio MCP. Its web
search uses DuckDuckGo's public HTML endpoint and requires no search credential.

```bash
pip install -e '.[research]'
```

```json
{
  "mcpServers": {
    "deep-research": {
      "command": "python",
      "args": ["/absolute/path/to/Synapse/mcp_server.py"]
    }
  }
}
```

## Tests

```bash
python -m pytest -q
```

The offline suite covers retained lifecycle, live user steering, post-checkpoint
revision, bounded Node discoveries, cross-Run routing isolation, plan fan-out, and
tool-call-safe context compaction.
