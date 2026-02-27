from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_ROOT = os.path.expanduser("~/.agent_framework/memory")
MAX_INJECT_LINES = 200


class MemoryStore:
    """
    File-based persistent memory for Master and Head agents.

    Each agent role gets its own MEMORY.md file that persists across sessions.
    At startup, the first MAX_INJECT_LINES lines are injected into context.
    The agent updates memory after task completion with learned patterns.
    """

    def __init__(self, role_key: str, root_dir: str | None = None):
        self.role_key = role_key
        self.root = Path(root_dir or DEFAULT_MEMORY_ROOT)
        self.memory_dir = self.root / role_key
        self.memory_file = self.memory_dir / "MEMORY.md"

    async def initialize(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            async with aiofiles.open(self.memory_file, "w") as f:
                await f.write(f"# Memory: {self.role_key}\n\n")

    async def read(self, max_lines: int = MAX_INJECT_LINES) -> str:
        """Read the first max_lines of MEMORY.md for context injection."""
        if not self.memory_file.exists():
            return ""
        async with aiofiles.open(self.memory_file, "r") as f:
            lines: list[str] = []
            async for line in f:
                lines.append(line)
                if len(lines) >= max_lines:
                    break
        return "".join(lines)

    async def read_full(self) -> str:
        if not self.memory_file.exists():
            return ""
        async with aiofiles.open(self.memory_file, "r") as f:
            return await f.read()

    async def append(self, content: str) -> None:
        """Append new learning to MEMORY.md."""
        await self.initialize()
        async with aiofiles.open(self.memory_file, "a") as f:
            await f.write(f"\n{content}\n")
        logger.info("Memory appended for %s (%d chars)", self.role_key, len(content))

    async def write(self, content: str) -> None:
        """Overwrite MEMORY.md (use for curated rewrites)."""
        await self.initialize()
        async with aiofiles.open(self.memory_file, "w") as f:
            await f.write(content)

    def get_injection_prompt(self, memory_content: str) -> str:
        """Format memory content for system prompt injection."""
        if not memory_content.strip():
            return ""
        return (
            f"<agent_memory>\n"
            f"The following is your persistent memory from previous sessions. "
            f"Use it to inform your decisions. Update it when you learn important patterns.\n\n"
            f"{memory_content}\n"
            f"</agent_memory>"
        )
