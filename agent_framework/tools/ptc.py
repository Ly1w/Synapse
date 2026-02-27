from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class PTCExecutor:
    """
    Programmatic Tool Calling executor.

    Node agents generate Python code at runtime that calls tools
    programmatically. This executor provides tool functions in the
    execution namespace and captures results.

    Key difference from pre-defined PTC: the agent writes the code,
    so it adapts to arbitrary data processing needs.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute_code(self, code: str) -> dict[str, Any]:
        """
        Execute agent-generated Python code with tool functions injected.

        The code has access to:
        - tools: dict of {name: callable} for all registered tools
        - results: dict to store output (code should write to results)
        - json, asyncio modules
        - call_tool(name, **kwargs): convenience wrapper

        Returns the 'results' dict after execution.
        """
        callables = self.registry.get_callable_tools()

        async def call_tool(name: str, **kwargs: Any) -> Any:
            fn = callables.get(name)
            if fn is None:
                return {"error": f"Tool '{name}' not found"}
            if inspect.iscoroutinefunction(fn):
                return await fn(**kwargs)
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: fn(**kwargs)
            )

        namespace: dict[str, Any] = {
            "tools": callables,
            "call_tool": call_tool,
            "asyncio": asyncio,
            "json": json,
            "results": {},
            "print_output": [],
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Handle async code by wrapping in an async function
            if "await " in code or "async " in code:
                wrapped = f"async def __ptc_main__():\n"
                for line in code.splitlines():
                    wrapped += f"    {line}\n"
                wrapped += "\n__ptc_result__ = asyncio.get_event_loop().run_until_complete(__ptc_main__())"

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(compile(wrapped, "<ptc>", "exec"), namespace)
            else:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(compile(code, "<ptc>", "exec"), namespace)

            stdout_text = stdout_capture.getvalue()
            if stdout_text:
                namespace["results"]["__stdout__"] = stdout_text

            return namespace.get("results", {})

        except Exception as e:
            logger.exception("PTC execution error")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stderr": stderr_capture.getvalue(),
            }

    def get_ptc_prompt_instructions(self) -> str:
        """
        Return prompt instructions that teach the agent how to use PTC.
        Include available tool names and the expected code format.
        """
        tool_names = self.registry.get_names()
        tools_list = ", ".join(tool_names) if tool_names else "(none)"

        return f"""\
You can write Python code to call tools programmatically when a task requires:
- Multiple sequential tool calls with data dependencies
- Data aggregation or filtering across tool results
- Complex logic that would need many round-trips

Available tools: {tools_list}

To use PTC, output a code block tagged with ```ptc
Your code has access to:
- `call_tool(name, **kwargs)` — call any registered tool (use await for async)
- `tools` — dict of all tool callables
- `results` — dict where you MUST store your final output
- `json` module

Example:
```ptc
data = call_tool("get_data", query="sales Q3")
filtered = [r for r in data if r["amount"] > 1000]
results["filtered_data"] = filtered
results["total"] = sum(r["amount"] for r in filtered)
```

Always store output in `results`. Only use PTC when it's more efficient than standard tool calls.
"""
