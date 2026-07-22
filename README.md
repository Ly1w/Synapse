# Synapse

Synapse is a Master-first agent runtime with an optional bounded
**Master → Head → Node** hierarchy. The Master handles ordinary conversation,
coherent tasks, and small tool workflows itself; it creates Heads and Nodes only when
delegation has a concrete benefit. The hierarchy keeps peer coordination, persistent
memory, tool retrieval, and reusable plans while making ownership and termination
explicit.

## Architecture

```text
User updates ───────────────────────────────┐
                                            ▼
Master (global owner and task graph) ── Head (sub-problem owner) ── Node (executor)
             │                         ▲  │                      ▲
             └── retained Run journal ─┘  └── bounded peers ─────┘
```

- **Master** is the primary executor. It interprets evolving user requirements, can
  answer or use registered tools directly, and owns the dependency graph, resource
  budget, discovery arbitration, and final checkpoint when delegation is useful.
- **Head** reasons about its sub-problem, delegates only focused work, coordinates
  with peer Heads, integrates evidence, and checks acceptance criteria. A Head may
  use zero Nodes when delegation is unnecessary.
- **Node** executes one task contract with bounded turns, wall time, discoveries,
  and sibling request/response messages. Nodes cannot create agents or redefine the
  task graph.

Every child receives a `TaskContract` containing scope, deliverable, dependencies,
and acceptance criteria. Every exit path returns an `AgentOutcome` such as
`completed`, `partial`, `blocked`, `failed`, or `cancelled`.

## Execution properties

- **Optional delegation**: routing defaults to direct Master execution. A Head is
  created only for work that benefits from parallel ownership, context isolation,
  independent verification, or a task too large for one coherent execution path.
- **Explicit delegation is binding**: when the user explicitly requests multiple
  agents for the current work, the planner must produce a valid hierarchy and may not
  silently downgrade the request to Master-only execution. If the routing model
  ignores that requirement twice, the runtime uses a bounded primary-owner followed
  by an independent-verifier topology instead of failing the Run.
- **Interruptible runs**: add user requirements while work is active with
  `MasterAgent.steer()`.
- **Retained agents**: a final response is a checkpoint, not destruction. Head and
  Node contexts remain available for a later revision.
- **Checkpoint history**: every Master response is retained with its Run revision;
  later steering creates a new response entry rather than overwriting the prior one.
- **Visible retained failures**: a failed revision records `last_error` separately,
  keeps the last valid checkpoint intact, and exposes the exact failure in the control
  room instead of showing only a generic retained status.
- **Durable action history**: messages, tool actions, decisions, outcomes, and agent
  context snapshots are written under `~/.agent_framework/runs/<run_id>/` by default.
- **Restart-safe inspection**: persisted Runs are rediscovered after server restart as
  read-only cold snapshots. They remain inspectable but are not presented as resumable
  live agents.
- **Bounded organization**: hard budgets limit turns, peer messages,
  discoveries, child agents, revisions, and wall-clock time. Tool calls have no fixed
  cumulative count; phase timeouts, per-tool timeouts, cancellation, and convergence
  rules bound their execution.
- **Lifecycle-based completion**: parents observe actual child task completion;
  duplicate or missing `REPORT` messages cannot deadlock the run.
- **Dependency scheduling**: Heads start when their declared dependencies have
  outcomes. Invalid cycles degrade explicitly instead of waiting forever.
- **Run isolation**: retained agents from old Runs cannot communicate with agents in
  a new Run.
- **Explicit cleanup**: call `archive_run()` to release live agents after their
  snapshots are persisted. A configurable retention limit archives older Runs.
- **Shared permission gate**: Master and Nodes pass every built-in, custom, and MCP
  tool call through the same runtime policy and approval queue.
- **Live self-awareness**: before every Master, Head, or Node model call, Synapse
  injects an ephemeral runtime snapshot with exact identity and role boundaries,
  current phase, remaining time/turns/communication capacity, live topology,
  steering/cancellation state, permissions, and recent action failures. The snapshot
  is refreshed per turn and is not persisted as stale conversation history.

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
export SYNAPSE_WORKSPACE=/absolute/path/to/your/project
export SYNAPSE_PERMISSION_MODE=auto  # ask | auto | full

python -m agent_framework.web.app
```

Open `http://127.0.0.1:8008`. The server binds to loopback by default because the
control room can inspect retained agent state and actions.

The control room includes a permission selector, pending-approval queue, and a user MCP
server panel. Approval events are also written into the Run journal.

## Built-in tools

These tools are registered directly by the framework. They do not require MCP:

| Tool | Purpose | Permission class |
| --- | --- | --- |
| `Read` | Bounded line-numbered UTF-8 file reads | read |
| `Glob` | Bounded recursive path discovery | read |
| `Grep` | Regex content/file/count search | read |
| `Write` | Atomic whole-file creation or replacement | write |
| `Edit` | Exact, uniqueness-checked string replacement | write |
| `Bash` | Bounded `/bin/bash -lc` command execution | shell |
| `Skill` | List or load installed `SKILL.md` workflows | read |
| `WebSearch` | Keyless public web search with domain filters | network |
| `WebFetch` | Readable public-page fetch with SSRF protection | network |

`WebSearch` uses DuckDuckGo's public HTML endpoint and needs no search API key.
`WebFetch` rejects credential-bearing URLs and private, local, or non-global targets.
Bash uses a sanitized child environment, a timeout, output caps, and process-group
cleanup. It is not an OS-level filesystem sandbox.

## Agent Skills

`Skill` means a real filesystem-backed Agent Skill, not an index of tool schemas.
Synapse discovers the name and description from YAML frontmatter at Run start and
loads the Markdown body only when the model invokes `Skill(name=...)`. Every Node sees
the actual registered tool schemas; Skill discovery never guesses which tools to hide.

Project Skills can live at `.synapse/skills/<name>/SKILL.md` or the compatible
`.claude/skills/<name>/SKILL.md` path. User Skills can live at
`~/.agent_framework/skills/<name>/SKILL.md` or `~/.claude/skills/<name>/SKILL.md`.
Nested project directories override parent definitions, and user definitions override
project definitions with the same declared name.

```markdown
---
name: review
description: Review the current code change for correctness and regressions.
---

Inspect the current diff. Apply this focus: $ARGUMENTS
```

The body may reference supporting files relative to the returned `base_directory`.
`$ARGUMENTS`, positional `$0`/`$1`, and `${SKILL_DIR}` are expanded when loaded.
`disable-model-invocation: true` hides a Skill from the model. The Claude-specific
`context: fork` extension is reported but not silently converted into Synapse
delegation; such a Skill is not model-invocable in this runtime.

## Permissions

| Mode | Automatic | Requires approval |
| --- | --- | --- |
| `ask` | Read, Glob, Grep, Skill, strictly read-only Bash | writes, effectful Bash, network, MCP |
| `auto` | reads, workspace Write/Edit, built-in WebSearch/WebFetch | external writes, effectful Bash, MCP |
| `full` | all registered calls | none |

Tool-specific validation still applies in full mode. A denied call returns a structured
tool error so the Agent can choose a safer alternative. Pending calls can be approved
or denied from the Web UI or through `/api/approvals/{approval_id}`; unattended
approval requests expire after a bounded interval.

## User MCP extensions

MCP is reserved for tools the user explicitly adds. It is not used to implement
Synapse's built-in filesystem, shell, Skill, or web tools. Every remote tool is exposed
as `mcp__<server>__<tool>`, remains subject to the permission gate, and cannot replace
a built-in name. MCP child processes inherit a credential-sanitized environment;
credentials must be passed deliberately in that server's explicit configuration.

Attach a user server from Python:

```python
from agent_framework.tools.mcp_provider import MCPServerConfig

await master.connect_mcp_server(MCPServerConfig(
    name="my-tools",
    command="npx",
    args=("-y", "@example/my-mcp-server"),
    tool_allowlist=frozenset({"search", "retrieve"}),
    category="mcp:my-tools",
))
# Registered names: mcp__my_tools__search, mcp__my_tools__retrieve
```

The control room exposes equivalent connect/disconnect endpoints and never starts an
MCP process unless the user submits its configuration.

## Communication rules

- A Node communicates only with its Head and sibling Nodes in the same Run.
- Node peer communication is a correlated, non-blocking question/response protocol.
- A Node reports a discovery to its Head; it never requests a new Head directly.
- Heads may coordinate with peer Heads or submit one bounded discovery proposal to
  Master.
- Master alone may change the Head topology, subject to the Run budget.
- Cross-layer, cross-group, and cross-Run messages are rejected by the Router.

## Tools, plans, and memory

- Built-in and user tool schemas are registered in `ToolRegistry`; the generated
  skills index is a compact retrieval catalog rather than executable code.
- User-requested stdio MCP servers can be attached through
  `MasterAgent.connect_mcp_server()`; allowlisted remote tools stay namespaced and the
  process remains alive for the runtime lifecycle.
- Plan templates are cached only after fully successful Head outcomes.
- Master and Head memory is persistent. Recent memory is injected so old append-only
  files do not permanently hide new information.
- Tool call/result pairs remain atomic during context compaction.

## Deep Research MCP server

`mcp_server.py` remains a separate compatibility server for external MCP clients. Its
web operations reuse the same keyless search implementation. Synapse itself already
has `WebSearch` and `WebFetch`, so it does not start this server automatically.

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
revision, bounded Node discoveries, cross-Run routing isolation, optional delegation,
built-in tools, permission approval/denial, namespaced user MCP tools, and
tool-call-safe context compaction.
