# Agent Framework

A hierarchical multi-agent framework with a Master / Head / Node architecture.

## Architecture

- **Master Agent**: User-facing. Generates tool skills index, decomposes tasks into
  Head agents, aggregates progress, manages plan cache and memory.
- **Head Agent**: Mid-level coordinator. Creates Node agents, aggregates their results,
  communicates with peer Heads, and reports to Master. Can escalate to suggest new Heads.
- **Node Agent**: Task executor. Reads skills index, calls tools (standard or PTC),
  communicates with sibling Nodes, and reports to its Head.

## Key Features

- **Skills**: LLM-generated tool index (`skills.md`) for compact context usage.
- **Programmatic Tool Calling (PTC)**: Node agents generate data-processing code at
  runtime instead of relying on pre-defined wrappers.
- **Plan Caching (APC)**: Extracts reusable plan templates from successful executions
  and adapts them for similar future tasks.
- **Memory**: Persistent file-based memory for Master and Head agents.
- **Context Engineering**: Per-agent context budgets, selective injection, compaction.

## Quick Start

```bash
pip install -e .
```

```python
import asyncio
from agent_framework.core.master_agent import MasterAgent
from agent_framework.llm.config import LLMConfig

config = LLMConfig(base_url="http://localhost:8000/v1", api_key="sk-xxx", model="qwen3-32b")
master = MasterAgent(llm_config=config)
result = asyncio.run(master.handle_user_request("Analyze the quarterly sales data", tools=[...]))
print(result)
```

## Communication Rules

- No cross-layer skipping (Node cannot contact Master directly).
- Heads can communicate with peer Heads and escalate to Master.
- Nodes can communicate with sibling Nodes under the same Head.
- Nodes under Head A cannot message Nodes under Head B.
