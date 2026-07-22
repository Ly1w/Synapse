from __future__ import annotations

import asyncio
import json
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
        self._phase = "idle"
        self._started_at = 0.0
        self._sent_counts: dict[MessageType, int] = {}
        self._llm_turns_used = 0
        self._runtime_observations: list[str] = []

        self.context.set_system_prompt(system_prompt)

    async def think(self, user_input: str, **kwargs: Any) -> str:
        """Single LLM reasoning turn with context-managed messages."""
        self.context.add_message({"role": "user", "content": user_input})
        awareness = self.context.inject_temporary(self.begin_llm_turn_awareness())
        try:
            messages = self.context.build_messages()
            response = await self.llm_client.chat_text(messages, **kwargs)
        finally:
            awareness.remove()
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
        awareness = self.context.inject_temporary(self.begin_llm_turn_awareness())
        try:
            messages = self.context.build_messages()
            response = await self.llm_client.chat_with_tools(messages, tools, **kwargs)
        finally:
            awareness.remove()
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
        else:
            self.add_runtime_observation(
                f"message {msg_type.value} to {receiver_id}",
                "failed",
                "router rejected the target or its channel is unavailable",
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
        self._llm_turns_used = 0

    @property
    def elapsed_seconds(self) -> float:
        if not self._started_at:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def remaining_seconds(self) -> float:
        if not self._started_at:
            return self.budget.timeout_seconds
        return max(0.0, self.budget.timeout_seconds - self.elapsed_seconds)

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self._llm_turns_used)

    def runtime_state(self) -> dict[str, Any]:
        """Current per-turn telemetry; subclasses extend this with role-local state."""
        remaining = self.remaining_seconds
        low_time = min(30.0, self.budget.timeout_seconds * 0.2)
        if remaining <= 0:
            urgency = "expired: stop new work and return retained partial state"
        elif remaining <= low_time or self.remaining_turns <= 1:
            urgency = "critical: stop exploration and synthesize the best result now"
        elif remaining <= self.budget.timeout_seconds * 0.35:
            urgency = "converge: prefer completion over optional investigation"
        else:
            urgency = "normal"
        return {
            "agent": {
                "agent_id": self.id,
                "hierarchy_level": self.__class__.__name__,
                "assigned_role": self.role,
                "run_id": self.run_id or None,
                "task_id": self.task_id or None,
                "running": self._running,
            },
            "phase": {
                "name": self._phase,
                "elapsed_seconds": round(self.elapsed_seconds, 1),
                "remaining_seconds": round(remaining, 1),
                "timeout_seconds": self.budget.timeout_seconds,
                "llm_turn": self._llm_turns_used,
                "max_turns": self.max_turns,
                "remaining_turns_after_this_call": self.remaining_turns,
                "urgency": urgency,
            },
            "communication_sent": {
                key.value: value for key, value in self._sent_counts.items()
            },
            "recent_action_observations": list(self._runtime_observations),
        }

    def runtime_awareness(self) -> str:
        """Authoritative, ephemeral state injected immediately before an LLM call."""
        state = json.dumps(self.runtime_state(), ensure_ascii=False, indent=2)
        return (
            "<runtime_awareness>\n"
            "This block is generated by the runtime immediately before this model call. "
            "It is authoritative for current identity, remaining limits, topology, and "
            "recent action status; older counts or rosters in conversation history may be "
            "stale. Limits are ceilings, not targets. Do not repeat an action that failed "
            "under unchanged arguments and environment: diagnose it, change strategy, or "
            "report the blocker. Follow the urgency field and converge before timeout.\n"
            f"{state}\n"
            "</runtime_awareness>"
        )

    def begin_llm_turn_awareness(self) -> str:
        """Count an LLM call and return the fresh telemetry supplied to that call."""
        self._llm_turns_used += 1
        return self.runtime_awareness()

    def add_runtime_observation(
        self,
        action: str,
        status: str,
        detail: str = "",
    ) -> None:
        """Retain a bounded, non-payload action summary for the next reasoning turn."""
        normalized = " ".join(str(detail).split())[:240]
        text = f"{action}: {status}"
        if normalized:
            text += f" ({normalized})"
        self._runtime_observations.append(text)
        self._runtime_observations = self._runtime_observations[-8:]

    def observe_action_result(self, action: str, result: Any) -> None:
        """Classify a tool/control result without duplicating its full payload."""
        value = result
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                value = None

        if isinstance(value, dict):
            error = value.get("error")
            status = str(value.get("status", "")).lower()
            failed = bool(error) or status in {"blocked", "denied", "error", "failed"}
            detail = str(error or (status if failed else ""))
            self.add_runtime_observation(action, "failed" if failed else "succeeded", detail)
            return
        self.add_runtime_observation(action, "succeeded")

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
            "phase": {
                "name": self._phase,
                "elapsed_seconds": round(self.elapsed_seconds, 1),
                "remaining_seconds": round(self.remaining_seconds, 1),
                "llm_turns_used": self._llm_turns_used,
                "remaining_turns": self.remaining_turns,
            },
            "sent_counts": {key.value: value for key, value in self._sent_counts.items()},
            "recent_action_observations": list(self._runtime_observations),
            "context": self.context.export_state(),
        }
