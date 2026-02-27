from __future__ import annotations

import logging
from typing import Any

import openai

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper over openai.AsyncOpenAI for any OpenAI-compatible endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> openai.types.chat.ChatCompletion:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.config.max_tokens),
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.pop("tool_choice", "auto")
        params.update(kwargs)

        logger.debug("LLM request: model=%s, messages=%d, tools=%s",
                      self.config.model, len(messages), bool(tools))
        response = await self._client.chat.completions.create(**params)
        logger.debug("LLM response: usage=%s", response.usage)
        return response

    async def chat_text(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Convenience: return the text content of the first choice."""
        response = await self.chat(messages, **kwargs)
        return response.choices[0].message.content or ""

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> openai.types.chat.ChatCompletion:
        """Chat with tool definitions; returns full response for tool_calls inspection."""
        return await self.chat(messages, tools=tools, **kwargs)
