from __future__ import annotations

import logging
from typing import Any

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

COMPACTION_PROMPT = """\
Summarize the following conversation history into a concise summary that preserves:
1. Key decisions and conclusions
2. Important facts and data discovered
3. Current task status and remaining work
4. Any unresolved issues or errors

Be concise but preserve all critical information. Output ONLY the summary.

Conversation:
{conversation}
"""


class ContextCompactor:
    """LLM-based context compaction for long conversations."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def compact(self, messages: list[dict[str, Any]]) -> str:
        """Summarize a list of messages into a compact text summary."""
        conv_text = self._messages_to_text(messages)
        if not conv_text.strip():
            return ""

        response = await self.llm_client.chat_text(
            [{"role": "user", "content": COMPACTION_PROMPT.format(conversation=conv_text)}],
            temperature=0.2,
            max_tokens=1024,
        )
        logger.info("Compacted %d messages into %d chars", len(messages), len(response))
        return response

    @staticmethod
    def _messages_to_text(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                parts.append(f"[{role}]: {content}")
            elif isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and "text" in p
                ]
                if text_parts:
                    parts.append(f"[{role}]: {''.join(text_parts)}")
        return "\n".join(parts)

    async def compact_tool_results(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Clear verbose tool results from older messages while keeping the
        tool call structure intact. This is a lightweight compaction.
        """
        cleaned: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 500:
                    cleaned.append({
                        **msg,
                        "content": content[:200] + "\n...[truncated]...",
                    })
                else:
                    cleaned.append(msg)
            else:
                cleaned.append(msg)
        return cleaned
