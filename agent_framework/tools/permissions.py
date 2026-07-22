from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PermissionMode(str, Enum):
    ASK = "ask"
    AUTO = "auto"
    FULL = "full"


class ApprovalRequest(BaseModel):
    approval_id: str
    run_id: str = ""
    agent_id: str = ""
    task_id: str = ""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str
    risk: str
    permission_mode: PermissionMode
    created_at: float = Field(default_factory=time.time)


@dataclass(slots=True)
class ApprovalContext:
    run_id: str = ""
    agent_id: str = ""
    task_id: str = ""
    journal: Any | None = None


@dataclass(slots=True)
class _PendingApproval:
    request: ApprovalRequest
    future: asyncio.Future[bool]
    journal: Any | None


_STRICT_READ_ONLY_COMMANDS = {
    "pwd", "ls", "grep", "rg", "head", "tail", "wc", "stat", "file",
    "du", "df", "which", "whereis", "type", "git",
}
_SHELL_META = re.compile(r"[|;&<>`]|\$\(")
_DANGEROUS_BASH = re.compile(
    r"(?:^|[\s;&|])(?:sudo|su|doas|rm|rmdir|shred|dd|mkfs(?:\.[a-z0-9]+)?|"
    r"mount|umount|shutdown|reboot|poweroff|kill|pkill|killall|chmod|chown|chgrp|"
    r"curl|wget|ssh|scp|rsync|nc|ncat|socat|pip\s+install|npm\s+(?:install|publish)|"
    r"apt(?:-get)?|yum|dnf|brew)(?:\s|$)|"
    r"git\s+(?:reset|clean|checkout\s+--|restore|push|commit|rebase)|"
    r"(?:>|>>|tee\s+)(?:/|~|\$HOME)",
    re.IGNORECASE,
)
_MUTATING_FIND_OR_SED = re.compile(r"(?:find\b.*(?:-delete|-exec)|sed\b.*\s-i(?:\s|$))")


class PermissionManager:
    """Central approval gate shared by Master, Nodes, built-ins, and MCP tools."""

    def __init__(
        self,
        workspace_root: str | Path,
        mode: PermissionMode | str = PermissionMode.AUTO,
        approval_timeout_seconds: float = 600.0,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.mode = PermissionMode(mode)
        self.approval_timeout_seconds = max(1.0, float(approval_timeout_seconds))
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()

    def get_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "workspace_root": str(self.workspace_root),
            "pending_count": len(self._pending),
            "approval_timeout_seconds": self.approval_timeout_seconds,
            "modes": [item.value for item in PermissionMode],
        }

    def set_mode(self, mode: PermissionMode | str) -> PermissionMode:
        self.mode = PermissionMode(mode)
        return self.mode

    def list_pending(self, run_id: str | None = None) -> list[dict[str, Any]]:
        values = [item.request for item in self._pending.values()]
        if run_id:
            values = [item for item in values if item.run_id == run_id]
        values.sort(key=lambda item: item.created_at)
        return [item.model_dump(mode="json") for item in values]

    async def authorize(
        self,
        tool: Any,
        arguments: dict[str, Any],
        context: ApprovalContext | None = None,
    ) -> tuple[bool, str, str]:
        needs_approval, reason, risk = self._assess(tool, arguments)
        if not needs_approval:
            return True, reason, risk

        context = context or ApprovalContext()
        loop = asyncio.get_running_loop()
        request = ApprovalRequest(
            approval_id=f"approval_{uuid.uuid4().hex[:12]}",
            run_id=context.run_id,
            agent_id=context.agent_id,
            task_id=context.task_id,
            tool_name=str(tool.name),
            arguments=_bounded_arguments(arguments),
            reason=reason,
            risk=risk,
            permission_mode=self.mode,
        )
        pending = _PendingApproval(request, loop.create_future(), context.journal)
        async with self._lock:
            self._pending[request.approval_id] = pending
        await self._record(pending, "approval_requested", {
            **request.model_dump(mode="json"),
            "message": f"{request.tool_name} is waiting for user approval.",
        })
        try:
            allowed = await asyncio.wait_for(
                asyncio.shield(pending.future),
                timeout=self.approval_timeout_seconds,
            )
            return allowed, reason, risk
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending.pop(request.approval_id, None)
            if not pending.future.done():
                pending.future.cancel()
            await self._record(pending, "approval_expired", {
                "approval_id": request.approval_id,
                "tool_name": request.tool_name,
                "reason": "The approval window expired before the user decided.",
            })
            return False, "Approval window expired.", risk
        except asyncio.CancelledError:
            async with self._lock:
                self._pending.pop(request.approval_id, None)
            await self._record(pending, "approval_cancelled", {
                "approval_id": request.approval_id,
                "tool_name": request.tool_name,
                "reason": "The owning agent phase ended before approval.",
            })
            raise

    async def resolve(self, approval_id: str, allow: bool) -> dict[str, Any]:
        async with self._lock:
            pending = self._pending.pop(approval_id, None)
        if pending is None:
            raise KeyError(f"Unknown or already resolved approval: {approval_id}")
        if not pending.future.done():
            pending.future.set_result(bool(allow))
        payload = {
            "approval_id": approval_id,
            "tool_name": pending.request.tool_name,
            "allowed": bool(allow),
            "message": "Tool call approved by user." if allow else "Tool call denied by user.",
        }
        await self._record(pending, "approval_resolved", payload)
        return payload

    def _assess(self, tool: Any, arguments: dict[str, Any]) -> tuple[bool, str, str]:
        scope = str(getattr(tool, "permission_scope", "safe") or "safe")
        if self.mode == PermissionMode.FULL or scope in {"safe", "read", "skill"}:
            return False, "Allowed by the current permission policy.", scope

        if scope == "network":
            return (
                self.mode == PermissionMode.ASK,
                "This tool accesses the public internet.",
                "network",
            )

        if scope == "write":
            path = self._argument_path(arguments)
            outside = path is not None and not _is_relative_to(path, self.workspace_root)
            if self.mode == PermissionMode.ASK:
                return True, "This tool modifies the filesystem.", "filesystem_write"
            return (
                outside,
                "This write targets a path outside the configured workspace."
                if outside else "Workspace write allowed in approve-for-me mode.",
                "external_write" if outside else "filesystem_write",
            )

        if scope == "execute":
            command = str(arguments.get("command", ""))
            cwd = self._argument_path(arguments, key="cwd")
            outside = cwd is not None and not _is_relative_to(cwd, self.workspace_root)
            safe = _is_strict_read_only_bash(command) and not outside
            if safe:
                return False, "Strictly read-only shell command.", "shell_read"
            risky = outside or bool(_DANGEROUS_BASH.search(command))
            return (
                True,
                "The shell command was detected as potentially unsafe."
                if risky else (
                    "Arbitrary shell execution is approval-gated because this runtime "
                    "does not provide an OS-level filesystem sandbox."
                ),
                "unsafe_shell" if risky else "shell",
            )

        if scope == "mcp":
            return True, "User-added MCP tools have unknown external side effects.", "mcp"

        return (
            self.mode != PermissionMode.FULL,
            "This custom tool has side effects that cannot be classified safely.",
            "external",
        )

    def _argument_path(
        self,
        arguments: dict[str, Any],
        key: str = "path",
    ) -> Path | None:
        value = arguments.get(key)
        if not isinstance(value, str) or not value.strip():
            return self.workspace_root if key == "cwd" else None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve(strict=False)

    @staticmethod
    async def _record(
        pending: _PendingApproval,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if pending.journal is None:
            return
        await pending.journal.record(
            event_type,
            source=pending.request.agent_id or "permission_manager",
            task_id=pending.request.task_id,
            payload=payload,
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_strict_read_only_bash(command: str) -> bool:
    command = command.strip()
    if not command or "\n" in command or _SHELL_META.search(command):
        return False
    if _DANGEROUS_BASH.search(command) or _MUTATING_FIND_OR_SED.search(command):
        return False
    parts = command.split()
    if not parts:
        return False
    executable = Path(parts[0]).name
    if executable not in _STRICT_READ_ONLY_COMMANDS:
        return False
    if executable == "git":
        if any(part == "--output" or part.startswith("--output=") for part in parts[2:]):
            return False
        return len(parts) >= 2 and parts[1] in {
            "status", "diff", "log", "show", "branch", "tag", "remote", "rev-parse",
            "ls-files", "grep",
        }
    return True


def _bounded_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _bounded_value(str(key), value)
        for key, value in arguments.items()
    }


def _bounded_value(key: str, value: Any) -> Any:
    if any(part in key.lower() for part in ("api_key", "apikey", "password", "secret", "token")):
        return "[REDACTED]"
    if isinstance(value, str):
        return (
            value[:20_000] + "\n...[truncated for approval display]"
            if len(value) > 20_000 else value
        )
    if isinstance(value, dict):
        return {str(child): _bounded_value(str(child), item) for child, item in value.items()}
    if isinstance(value, list):
        return [_bounded_value(key, item) for item in value[:200]]
    return value
