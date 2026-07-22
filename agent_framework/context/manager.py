from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class ContextHandle:
    """Handle returned by inject_temporary; call .remove() to evict the content."""

    def __init__(self, manager: ContextManager, item_id: int):
        self._manager = manager
        self._item_id = item_id

    def remove(self) -> None:
        self._manager._remove_temporary(self._item_id)


class ContextManager:
    """
    Manages the message context for a single agent.

    Tracks token budget, supports pinned items (always present),
    temporary injection (removable), and automatic compaction.
    """

    def __init__(
        self,
        max_tokens: int = 128000,
        compaction_threshold: float = 0.8,
        keep_recent: int = 5,
        model_name: str = "gpt-4o",
    ):
        self.max_tokens = max_tokens
        self.compaction_threshold = compaction_threshold
        self.keep_recent = keep_recent

        try:
            self._enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

        self._system_prompt: str = ""
        self._pinned: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []
        self._temporary: dict[int, dict[str, Any]] = {}
        self._temp_counter = 0
        self._compaction_summary: str = ""

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def add_pinned(self, content: str, role: str = "system") -> None:
        """Add always-present context (memory, config, etc.)."""
        self._pinned.append({"role": role, "content": content})

    def clear_pinned(self) -> None:
        self._pinned.clear()

    def add_message(self, message: dict[str, Any]) -> None:
        self._messages.append(message)
        if self.estimate_tokens() > self.max_tokens * self.compaction_threshold:
            logger.info("Context approaching limit (%d tokens), compacting",
                        self.estimate_tokens())
            self._auto_compact()

    def inject_temporary(self, content: str, role: str = "system") -> ContextHandle:
        """Inject content that can be removed later via the returned handle."""
        self._temp_counter += 1
        item_id = self._temp_counter
        self._temporary[item_id] = {"role": role, "content": content}
        return ContextHandle(self, item_id)

    def _remove_temporary(self, item_id: int) -> None:
        self._temporary.pop(item_id, None)

    def build_messages(self) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        result: list[dict[str, Any]] = []

        # System prompt
        system_parts = [self._system_prompt] if self._system_prompt else []
        if self._compaction_summary:
            system_parts.append(
                f"\n<conversation_summary>\n{self._compaction_summary}\n</conversation_summary>"
            )
        for p in self._pinned:
            system_parts.append(p["content"])
        for t in self._temporary.values():
            if t["role"] == "system":
                system_parts.append(t["content"])

        if system_parts:
            result.append({"role": "system", "content": "\n\n".join(system_parts)})

        # Non-system temporary items
        for t in self._temporary.values():
            if t["role"] != "system":
                result.append(t)

        result.extend(self._messages)
        return result

    def estimate_tokens(self) -> int:
        """Estimate total tokens across all context components."""
        total = 0
        for msg in self.build_messages():
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self._enc.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += len(self._enc.encode(part["text"]))
            # Rough overhead per message
            total += 4
        return total

    def _auto_compact(self) -> None:
        """Compact older messages, keeping recent ones intact."""
        if len(self._messages) <= self.keep_recent:
            return

        cut = len(self._messages) - self.keep_recent
        # Never split an assistant tool-call message from its tool results.  Invalid
        # call/result adjacency would make the next model request fail.
        while cut > 0 and self._messages[cut].get("role") == "tool":
            cut -= 1
        older = self._messages[:cut]
        recent = self._messages[cut:]

        summary_parts: list[str] = []
        if self._compaction_summary:
            summary_parts.append(self._compaction_summary)

        for msg in older:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                truncated = content[:300] + "..." if len(content) > 300 else content
                summary_parts.append(f"[{role}]: {truncated}")

        self._compaction_summary = "\n".join(summary_parts)
        self._messages = recent
        logger.info("Compacted %d messages, kept %d recent", len(older), len(recent))

    def force_compact(self, summary: str) -> None:
        """Replace all messages with an explicit summary (e.g., from LLM compaction)."""
        self._compaction_summary = summary
        self._messages = self._messages[-self.keep_recent:] if self._messages else []

    def clear_messages(self) -> None:
        self._messages.clear()
        self._compaction_summary = ""

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot suitable for retained runs."""
        return {
            "system_prompt": self._system_prompt,
            "pinned": list(self._pinned),
            "messages": list(self._messages),
            "compaction_summary": self._compaction_summary,
            "max_tokens": self.max_tokens,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a prior snapshot without replaying compaction side effects."""
        self._system_prompt = str(state.get("system_prompt", self._system_prompt))
        self._pinned = list(state.get("pinned", []))
        self._messages = list(state.get("messages", []))
        self._compaction_summary = str(state.get("compaction_summary", ""))

    @property
    def message_count(self) -> int:
        return len(self._messages)
