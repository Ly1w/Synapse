from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..communication.channel import Channel
from ..communication.message import Message, MessageType
from ..communication.router import Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all agents in the hierarchy.

    Provides shared functionality: LLM calls through a context-managed
    message history, message sending/receiving via the router, and
    a main run loop.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        system_prompt: str,
        llm_client: LLMClient,
        router: Router,
        channel: Channel,
        context_manager: ContextManager | None = None,
        max_turns: int = 50,
    ):
        self.id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.router = router
        self.channel = channel
        self.context = context_manager or ContextManager()
        self.max_turns = max_turns
        self._running = False

        self.context.set_system_prompt(system_prompt)

    async def think(self, user_input: str, **kwargs: Any) -> str:
        """Single LLM reasoning turn with context-managed messages."""
        self.context.add_message({"role": "user", "content": user_input})
        messages = self.context.build_messages()
        response = await self.llm_client.chat_text(messages, **kwargs)
        self.context.add_message({"role": "assistant", "content": response})
        return response

    async def think_with_tools(
        self,
        user_input: str,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """LLM turn with tool definitions. Returns full completion response."""
        self.context.add_message({"role": "user", "content": user_input})
        messages = self.context.build_messages()
        response = await self.llm_client.chat_with_tools(messages, tools, **kwargs)
        msg = response.choices[0].message

        # Add assistant response to context
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        self.context.add_message(assistant_msg)
        return response

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result to the context."""
        self.context.add_message({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    async def send_message(
        self,
        receiver_id: str,
        msg_type: MessageType,
        content: dict[str, Any],
    ) -> bool:
        msg = Message(
            sender_id=self.id,
            receiver_id=receiver_id,
            msg_type=msg_type,
            content=content,
        )
        return await self.router.send(msg)

    async def receive_message(self, timeout: float | None = None) -> Message | None:
        return await self.channel.receive(timeout=timeout)

    def drain_messages(self) -> list[Message]:
        return self.channel.drain()

    async def run(self) -> Any:
        """Main agent loop. Override in subclasses."""
        raise NotImplementedError

    def stop(self) -> None:
        self._running = False
