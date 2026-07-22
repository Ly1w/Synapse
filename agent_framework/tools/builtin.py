from __future__ import annotations

import asyncio
import fnmatch
import glob as globlib
import os
import re
import signal
import tempfile
from pathlib import Path
from typing import Any

from deep_research.fetch import fetch_webpage
from deep_research.search import search_web

from ..skills import SkillCatalog

from .registry import ToolRegistry


_MAX_TOOL_OUTPUT = 40_000
_MAX_READ_LINES = 2_000
_MAX_LINE_CHARS = 2_000
_MAX_GLOB_RESULTS = 1_000
_MAX_GREP_FILES = 10_000
_MAX_GREP_MATCHES = 1_000
_SENSITIVE_ENV_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


class BuiltinToolset:
    """Workspace-aware coding and web tools registered without MCP."""

    def __init__(
        self,
        workspace_root: str | Path,
        skill_roots: list[str | Path] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.skill_catalog = SkillCatalog(
            self.workspace_root,
            skill_roots,
        )

    def register(self, registry: ToolRegistry) -> list[str]:
        definitions = [
            (
                "Bash",
                "Run a shell command in the configured workspace. Returns stdout, stderr, exit code, and timeout status.",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute."},
                        "timeout_seconds": {
                            "type": "integer", "minimum": 1, "maximum": 300, "default": 120,
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory, relative to the workspace or absolute.",
                        },
                    },
                    "required": ["command"],
                },
                "execute",
                self.bash,
            ),
            (
                "Read",
                "Read a UTF-8 text file with line numbers. Supports an offset and bounded line count.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 1, "default": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": _MAX_READ_LINES, "default": 500},
                    },
                    "required": ["path"],
                },
                "read",
                self.read,
            ),
            (
                "Write",
                "Create or completely replace one UTF-8 text file. The write is atomic.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                "write",
                self.write,
            ),
            (
                "Edit",
                "Replace an exact string in one UTF-8 text file. By default the old string must occur exactly once.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
                "write",
                self.edit,
            ),
            (
                "Glob",
                "Find files by a glob pattern. Supports ** recursion and returns bounded absolute paths.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "description": "Search root; defaults to the workspace."},
                    },
                    "required": ["pattern"],
                },
                "read",
                self.glob,
            ),
            (
                "Grep",
                "Search text files with a regular expression and return matching lines or matching file paths.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "description": "File or directory; defaults to the workspace."},
                        "glob": {"type": "string", "description": "Optional filename glob such as *.py."},
                        "output_mode": {
                            "type": "string",
                            "enum": ["content", "files_with_matches", "count"],
                            "default": "content",
                        },
                        "case_insensitive": {"type": "boolean", "default": False},
                        "context": {"type": "integer", "minimum": 0, "maximum": 20, "default": 0},
                        "head_limit": {"type": "integer", "minimum": 1, "maximum": _MAX_GREP_MATCHES, "default": 200},
                    },
                    "required": ["pattern"],
                },
                "read",
                self.grep,
            ),
            (
                "Skill",
                "Load an installed SKILL.md workflow by name. Use name='list' to list available skills.",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "string", "description": "Optional user arguments for the workflow."},
                    },
                    "required": ["name"],
                },
                "skill",
                self.skill,
            ),
            (
                "WebSearch",
                "Search the public web without an API key and return normalized title, URL, and snippet results.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                        "allowed_domains": {"type": "array", "items": {"type": "string"}},
                        "blocked_domains": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query"],
                },
                "network",
                self.web_search,
            ),
            (
                "WebFetch",
                "Fetch readable text from one public HTTP(S) webpage. Private/local network addresses are rejected.",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "max_chars": {"type": "integer", "minimum": 500, "maximum": 8000, "default": 8000},
                    },
                    "required": ["url"],
                },
                "network",
                self.web_fetch,
            ),
        ]
        names: list[str] = []
        for name, description, parameters, permission_scope, callable_fn in definitions:
            registry.register(
                name,
                description,
                parameters,
                category="builtin",
                callable_fn=callable_fn,
                source="builtin",
                permission_scope=permission_scope,
            )
            names.append(name)
        return names

    async def bash(
        self,
        command: str,
        timeout_seconds: int = 120,
        cwd: str = "",
    ) -> dict[str, Any]:
        command = command.strip()
        if not command:
            return {"error": "command cannot be blank"}
        if len(command) > 100_000:
            return {"error": "command exceeds 100000 characters"}
        working_dir = self._path(cwd) if cwd else self.workspace_root
        if not working_dir.is_dir():
            return {"error": f"working directory does not exist: {working_dir}"}
        timeout_seconds = max(1, min(int(timeout_seconds), 300))
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=str(working_dir),
            env=_sanitized_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
                await asyncio.wait_for(process.wait(), timeout=2)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout, stderr = await process.communicate()
        return {
            "command": command,
            "cwd": str(working_dir),
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "stdout": _bounded_text(stdout.decode("utf-8", errors="replace")),
            "stderr": _bounded_text(stderr.decode("utf-8", errors="replace")),
        }

    def read(self, path: str, offset: int = 1, limit: int = 500) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            return {"error": f"file does not exist: {target}"}
        offset = max(1, int(offset))
        limit = max(1, min(int(limit), _MAX_READ_LINES))
        selected: list[str] = []
        lines_seen = 0
        truncated = False
        try:
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                for lines_seen, line in enumerate(handle, start=1):
                    if lines_seen < offset:
                        continue
                    if len(selected) >= limit:
                        truncated = True
                        break
                    selected.append(line.rstrip("\r\n"))
        except OSError as error:
            return {"error": str(error), "path": str(target)}
        rendered = "\n".join(
            f"{number:>6}\t{line[:_MAX_LINE_CHARS]}"
            for number, line in enumerate(selected, start=offset)
        )
        return {
            "path": str(target),
            "offset": offset,
            "lines_returned": len(selected),
            "total_lines": None if truncated else lines_seen,
            "truncated": truncated,
            "next_offset": offset + len(selected) if truncated else None,
            "content": _bounded_text(rendered),
        }

    def write(self, path: str, content: str) -> dict[str, Any]:
        target = self._path(path)
        if target.exists() and target.is_dir():
            return {"error": f"path is a directory: {target}"}
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(target, content)
        return {"path": str(target), "bytes_written": len(content.encode("utf-8"))}

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            return {"error": f"file does not exist: {target}"}
        if not old_string:
            return {"error": "old_string cannot be empty"}
        content = target.read_text(encoding="utf-8", errors="strict")
        occurrences = content.count(old_string)
        if occurrences == 0:
            return {"error": "old_string was not found", "path": str(target)}
        if occurrences > 1 and not replace_all:
            return {
                "error": "old_string is not unique; provide more context or set replace_all=true",
                "occurrences": occurrences,
                "path": str(target),
            }
        updated = content.replace(old_string, new_string, -1 if replace_all else 1)
        self._atomic_write(target, updated)
        return {
            "path": str(target),
            "replacements": occurrences if replace_all else 1,
            "bytes_written": len(updated.encode("utf-8")),
        }

    def glob(self, pattern: str, path: str = "") -> dict[str, Any]:
        if not pattern.strip():
            return {"error": "pattern cannot be blank"}
        root = self._path(path) if path else self.workspace_root
        expression = pattern if Path(pattern).is_absolute() else str(root / pattern)
        matches: list[Path] = []
        for value in globlib.iglob(expression, recursive=True, include_hidden=True):
            matches.append(Path(value).resolve(strict=False))
            if len(matches) > _MAX_GLOB_RESULTS:
                break
        matches.sort(key=lambda item: str(item))
        limited = matches[:_MAX_GLOB_RESULTS]
        return {
            "pattern": pattern,
            "path": str(root),
            "matches": [str(item) for item in limited],
            "truncated": len(matches) > len(limited),
        }

    def grep(
        self,
        pattern: str,
        path: str = "",
        glob: str = "",
        output_mode: str = "content",
        case_insensitive: bool = False,
        context: int = 0,
        head_limit: int = 200,
    ) -> dict[str, Any]:
        try:
            regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        except re.error as error:
            return {"error": f"invalid regular expression: {error}"}
        root = self._path(path) if path else self.workspace_root
        if not root.exists():
            return {"error": f"path does not exist: {root}"}
        context = max(0, min(int(context), 20))
        head_limit = max(1, min(int(head_limit), _MAX_GREP_MATCHES))
        files = [root] if root.is_file() else self._walk_files(root, glob)
        results: list[Any] = []
        matched_files = 0
        for file_path in files[:_MAX_GREP_FILES]:
            try:
                if file_path.stat().st_size > 5_000_000:
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            indexes = [index for index, line in enumerate(lines) if regex.search(line)]
            if not indexes:
                continue
            matched_files += 1
            if output_mode == "files_with_matches":
                results.append(str(file_path))
            elif output_mode == "count":
                results.append({"path": str(file_path), "count": len(indexes)})
            else:
                emitted: set[int] = set()
                for index in indexes:
                    for line_index in range(max(0, index - context), min(len(lines), index + context + 1)):
                        if line_index in emitted:
                            continue
                        emitted.add(line_index)
                        results.append({
                            "path": str(file_path),
                            "line": line_index + 1,
                            "match": line_index == index,
                            "text": lines[line_index][:_MAX_LINE_CHARS],
                        })
                        if len(results) >= head_limit:
                            break
                    if len(results) >= head_limit:
                        break
            if len(results) >= head_limit:
                break
        return {
            "pattern": pattern,
            "path": str(root),
            "output_mode": output_mode,
            "matched_files": matched_files,
            "results": results[:head_limit],
            "truncated": len(results) >= head_limit,
        }

    def skill(self, name: str, arguments: str = "") -> dict[str, Any]:
        return self.skill_catalog.load(name, arguments)

    @staticmethod
    def web_search(
        query: str,
        max_results: int = 5,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> str:
        return search_web(query, max_results, allowed_domains, blocked_domains)

    @staticmethod
    def web_fetch(url: str, max_chars: int = 8000) -> str:
        return fetch_webpage(url, max_chars)

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve(strict=False)

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        existing_mode = target.stat().st_mode if target.exists() else None
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if existing_mode is not None:
                os.chmod(temporary, existing_mode)
            os.replace(temporary, target)
        finally:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _walk_files(root: Path, pattern: str) -> list[Path]:
        results = []
        for current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in {".git", "__pycache__"}]
            for name in files:
                if pattern and not fnmatch.fnmatch(name, pattern):
                    continue
                results.append(Path(current) / name)
                if len(results) >= _MAX_GREP_FILES:
                    return results
        return results

def register_builtin_tools(
    registry: ToolRegistry,
    workspace_root: str | Path,
    skill_roots: list[str | Path] | None = None,
) -> BuiltinToolset:
    toolset = BuiltinToolset(workspace_root, skill_roots)
    toolset.register(registry)
    return toolset


def _bounded_text(value: str) -> str:
    if len(value) <= _MAX_TOOL_OUTPUT:
        return value
    return value[:_MAX_TOOL_OUTPUT] + "\n...[tool output truncated]"


def _sanitized_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.lower() for part in _SENSITIVE_ENV_PARTS)
    }
