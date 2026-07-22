from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from ..communication.channel import Channel
from ..communication.message import Message, MessageType
from ..communication.router import Router
from ..context.manager import ContextManager
from ..llm.client import LLMClient
from .contracts import AgentBudget

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
        max_turns: int | None = None,
        budget: AgentBudget | None = None,
        task_id: str = "",
        run_id: str = "",
        journal: Any | None = None,
    ):
        self.id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.router = router
        self.channel = channel
        self.context = context_manager or ContextManager()
        self.budget = budget or AgentBudget(max_turns=max_turns or 50)
        self.max_turns = self.budget.max_turns
        self.task_id = task_id
        self.run_id = run_id
        self.journal = journal
        self._running = False
        self._started_at = 0.0
        self._sent_counts: dict[MessageType, int] = {}

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
        *,
        correlation_id: str = "",
        reply_to: str = "",
    ) -> bool:
        if msg_type == MessageType.PEER_REQUEST and not correlation_id:
            correlation_id = uuid.uuid4().hex[:12]
        msg = Message(
            sender_id=self.id,
            receiver_id=receiver_id,
            msg_type=msg_type,
            content=content,
            task_id=self.task_id,
            correlation_id=correlation_id,
            reply_to=reply_to,
        )
        sent = await self.router.send(msg)
        if sent:
            self._sent_counts[msg_type] = self._sent_counts.get(msg_type, 0) + 1
            await self.record_event(
                "message_sent",
                payload={
                    "message_id": msg.id,
                    "message_type": msg_type.value,
                    "correlation_id": correlation_id,
                    "content": content,
                },
                target=receiver_id,
            )
        return sent

    async def receive_message(self, timeout: float | None = None) -> Message | None:
        return await self.channel.receive(timeout=timeout)

    def drain_messages(self) -> list[Message]:
        return self.channel.drain()

    async def run(self) -> Any:
        """Main agent loop. Override in subclasses."""
        raise NotImplementedError

    def stop(self) -> None:
        self._running = False

    def start_clock(self) -> None:
        self._started_at = time.monotonic()

    @property
    def remaining_seconds(self) -> float:
        if not self._started_at:
            return self.budget.timeout_seconds
        elapsed = time.monotonic() - self._started_at
        return max(0.0, self.budget.timeout_seconds - elapsed)

    def sent_count(self, msg_type: MessageType) -> int:
        return self._sent_counts.get(msg_type, 0)

    async def record_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        target: str = "",
    ) -> None:
        if self.journal:
            await self.journal.record(
                event_type,
                source=self.id,
                target=target,
                task_id=self.task_id,
                payload=payload,
            )

    def export_state(self) -> dict[str, Any]:
        return {
            "agent_id": self.id,
            "role": self.role,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "running": self._running,
            "budget": self.budget.model_dump(mode="json"),
            "sent_counts": {key.value: value for key, value in self._sent_counts.items()},
            "context": self.context.export_state(),
        }
