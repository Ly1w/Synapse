from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes registered tool callables by name with given arguments."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(self, tool_name: str, arguments: dict[str, Any] | str) -> Any:
        tool = self.registry.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        if tool.callable is None:
            return {"error": f"Tool '{tool_name}' has no callable implementation"}

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {"error": f"Failed to parse arguments: {arguments}"}

        try:
            if inspect.iscoroutinefunction(tool.callable):
                result = await tool.callable(**arguments)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tool.callable(**arguments)
                )
            return result
        except Exception as e:
            logger.exception("Tool execution error: %s", tool_name)
            return {"error": f"Tool '{tool_name}' failed: {str(e)}"}

    async def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Execute a batch of tool calls (from OpenAI chat response format).

        Each item: {"id": ..., "function": {"name": ..., "arguments": ...}}
        Returns list of {"tool_call_id": ..., "role": "tool", "content": ...}
        """
        results = []
        for tc in tool_calls:
            func = tc.get("function", tc)
            name = func["name"]
            args = func.get("arguments", "{}")
            result = await self.execute(name, args)
            results.append({
                "tool_call_id": tc.get("id", ""),
                "role": "tool",
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })
        return results
