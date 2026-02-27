from __future__ import annotations

import asyncio
from typing import Any

from .message import Message


class Channel:
    """Async message channel backed by asyncio.Queue."""

    def __init__(self, owner_id: str, maxsize: int = 0):
        self.owner_id = owner_id
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=maxsize)

    async def send(self, message: Message) -> None:
        await self._queue.put(message)

    async def receive(self, timeout: float | None = None) -> Message | None:
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    def receive_nowait(self) -> Message | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def drain(self) -> list[Message]:
        """Drain all pending messages without blocking."""
        msgs: list[Message] = []
        while True:
            m = self.receive_nowait()
            if m is None:
                break
            msgs.append(m)
        return msgs

    @property
    def pending(self) -> int:
        return self._queue.qsize()
