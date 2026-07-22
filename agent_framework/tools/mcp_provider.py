from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from .registry import ToolRegistry

logger = logging.getLogger(__name__)

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_STOP = object()
_SENSITIVE_ENV_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """Configuration for one long-lived stdio MCP tool provider."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    cwd: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    tool_allowlist: frozenset[str] | None = None
    tool_prefix: str = ""
    category: str = "mcp"
    connect_timeout_seconds: float = 20.0
    call_timeout_seconds: float = 30.0


class MCPToolProvider:
    """Expose an MCP server's tools through ``ToolRegistry``.

    One background task owns the stdio and ClientSession contexts from startup to
    shutdown. This avoids crossing AnyIO cancel-scope ownership boundaries while
    still allowing Agent tasks to invoke the session through a request queue.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._worker_task: asyncio.Task[None] | None = None
        self._requests: asyncio.Queue[Any] | None = None
        self._ready: asyncio.Future[list[Any]] | None = None
        self._registered_names: dict[str, str] = {}
        self._registry: ToolRegistry | None = None

    @property
    def connected(self) -> bool:
        return bool(self._worker_task and not self._worker_task.done())

    @property
    def registered_names(self) -> list[str]:
        return list(self._registered_names)

    async def connect(self, registry: ToolRegistry) -> list[str]:
        if self.connected:
            return self.registered_names
        if not self.config.command.strip():
            raise ValueError("MCP server command cannot be blank")

        loop = asyncio.get_running_loop()
        self._registry = registry
        self._requests = asyncio.Queue()
        self._ready = loop.create_future()
        self._worker_task = asyncio.create_task(
            self._session_worker(),
            name=f"mcp:{self.config.name}",
        )
        try:
            tools = await asyncio.wait_for(
                asyncio.shield(self._ready),
                timeout=self.config.connect_timeout_seconds,
            )
            selected = self._select_tools(tools)
            if not selected:
                raise RuntimeError(
                    f"MCP server '{self.config.name}' exposed no allowed tools"
                )

            planned: list[tuple[Any, str]] = []
            planned_names: set[str] = set()
            for tool in selected:
                exposed_name = self._exposed_name(str(tool.name))
                if not _TOOL_NAME_PATTERN.fullmatch(exposed_name):
                    raise ValueError(f"Invalid exposed MCP tool name: {exposed_name}")
                if exposed_name in planned_names or registry.get(exposed_name) is not None:
                    raise ValueError(f"Tool name already registered: {exposed_name}")
                planned_names.add(exposed_name)
                planned.append((tool, exposed_name))

            for tool, exposed_name in planned:
                remote_name = str(tool.name)
                registry.register(
                    name=exposed_name,
                    description=str(tool.description or f"MCP tool from {self.config.name}"),
                    parameters=dict(tool.inputSchema or {"type": "object", "properties": {}}),
                    category=self.config.category,
                    callable_fn=self._make_callable(remote_name),
                    source=f"mcp:{self.config.name}",
                    permission_scope="mcp",
                )
                self._registered_names[exposed_name] = remote_name
            logger.info(
                "Connected MCP server %s with tools: %s",
                self.config.name,
                ", ".join(self.registered_names),
            )
            return self.registered_names
        except BaseException:
            await self.close(log_errors=False)
            raise

    async def call_tool(self, remote_name: str, arguments: dict[str, Any]) -> Any:
        if not self.connected or self._requests is None:
            raise RuntimeError(f"MCP server '{self.config.name}' is not connected")
        if remote_name not in self._registered_names.values():
            raise ValueError(f"MCP tool is not registered by this provider: {remote_name}")

        loop = asyncio.get_running_loop()
        response: asyncio.Future[Any] = loop.create_future()
        await self._requests.put((remote_name, arguments, response))
        return await asyncio.wait_for(
            asyncio.shield(response),
            timeout=self.config.call_timeout_seconds + 1,
        )

    async def close(self, log_errors: bool = True) -> None:
        task = self._worker_task
        if task is None:
            return
        if not task.done() and self._requests is not None:
            await self._requests.put(_STOP)
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            if log_errors:
                logger.exception("MCP server %s stopped with an error", self.config.name)
        finally:
            if self._registry is not None:
                for name in self._registered_names:
                    self._registry.unregister(name)
            self._registered_names.clear()
            self._registry = None
            self._worker_task = None
            self._requests = None
            self._ready = None

    def status(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "connected": self.connected,
            "tools": self.registered_names,
            "transport": "stdio",
        }

    def _select_tools(self, tools: list[Any]) -> list[Any]:
        allowed = self.config.tool_allowlist
        if allowed is None:
            return tools
        available = {str(tool.name): tool for tool in tools}
        missing = sorted(allowed - available.keys())
        if missing:
            raise RuntimeError(
                f"MCP server '{self.config.name}' is missing tools: {', '.join(missing)}"
            )
        return [tool for tool in tools if str(tool.name) in allowed]

    def _exposed_name(self, remote_name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_-]", "_", remote_name).strip("_") or "tool"
        digest = hashlib.sha256(remote_name.encode("utf-8")).hexdigest()[:8]
        if normalized != remote_name:
            normalized = f"{normalized}_{digest}"
        prefix = self.config.tool_prefix
        available = 64 - len(prefix)
        if available < 10:
            raise ValueError(f"MCP tool prefix is too long: {prefix}")
        if len(normalized) > available:
            normalized = f"{normalized[:available - 9]}_{digest}"
        return f"{prefix}{normalized}"

    def _make_callable(self, remote_name: str):
        async def invoke(**arguments: Any) -> Any:
            return await self.call_tool(remote_name, arguments)

        return invoke

    async def _session_worker(self) -> None:
        pending: set[asyncio.Task[None]] = set()
        try:
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "MCP SDK is missing in this Python interpreter. Run: "
                    "python -m pip install -e '.[web]'"
                ) from error

            params = StdioServerParameters(
                command=self.config.command,
                args=list(self.config.args),
                cwd=self.config.cwd,
                env={
                    **{
                        key: value for key, value in os.environ.items()
                        if not any(part in key.lower() for part in _SENSITIVE_ENV_PARTS)
                    },
                    **self.config.env,
                },
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    if self._ready and not self._ready.done():
                        self._ready.set_result(list(listed.tools))

                    assert self._requests is not None
                    while True:
                        request = await self._requests.get()
                        if request is _STOP:
                            break
                        remote_name, arguments, response = request
                        task = asyncio.create_task(
                            self._dispatch_call(session, remote_name, arguments, response)
                        )
                        pending.add(task)
                        task.add_done_callback(pending.discard)
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
        except BaseException as error:
            if self._ready and not self._ready.done():
                self._ready.set_exception(error)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self._fail_waiting_calls(error)
            if isinstance(error, asyncio.CancelledError):
                raise
            if self._ready and self._ready.done() and not self._ready.cancelled():
                # Once startup succeeded, retain the exception on the worker so
                # status and subsequent calls cannot pretend the process is healthy.
                raise

    async def _dispatch_call(
        self,
        session: Any,
        remote_name: str,
        arguments: dict[str, Any],
        response: asyncio.Future[Any],
    ) -> None:
        try:
            result = await session.call_tool(
                remote_name,
                arguments=arguments,
                read_timeout_seconds=timedelta(seconds=self.config.call_timeout_seconds),
            )
            value = self._normalize_result(result, remote_name)
            if not response.done():
                response.set_result(value)
        except BaseException as error:
            if not response.done():
                response.set_exception(error)

    def _fail_waiting_calls(self, error: BaseException) -> None:
        if self._requests is None:
            return
        while not self._requests.empty():
            item = self._requests.get_nowait()
            if item is _STOP:
                continue
            _, _, response = item
            if not response.done():
                response.set_exception(error)

    def _normalize_result(self, result: Any, remote_name: str) -> Any:
        structured = getattr(result, "structuredContent", None)
        if structured is not None and not getattr(result, "isError", False):
            if isinstance(structured, dict) and set(structured) == {"result"}:
                return structured["result"]
            return structured

        content = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                content.append(str(text))
                continue
            if hasattr(block, "model_dump"):
                content.append(json.dumps(block.model_dump(mode="json"), ensure_ascii=False))
            else:
                content.append(str(block))
        rendered = "\n".join(content)
        if getattr(result, "isError", False):
            return {
                "error": rendered or "MCP tool returned an error",
                "server": self.config.name,
                "tool": remote_name,
            }
        return rendered
