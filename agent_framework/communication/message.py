from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    TASK_ASSIGN = "task_assign"
    REPORT = "report"
    CLARIFICATION = "clarification"
    PEER_MSG = "peer_msg"
    PEER_REQUEST = "peer_request"
    PEER_RESPONSE = "peer_response"
    DISCOVERY = "discovery"
    GUIDANCE = "guidance"
    CANCEL = "cancel"
    ESCALATION = "escalation"
    PROGRESS = "progress"


class Message(BaseModel):
    """Core message exchanged between agents."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender_id: str
    receiver_id: str
    msg_type: MessageType
    content: dict[str, Any] = Field(default_factory=dict)
    task_id: str = ""
    correlation_id: str = ""
    reply_to: str = ""
    timestamp: float = Field(default_factory=time.time)

    def summary(self, max_len: int = 200) -> str:
        """Short human-readable summary for logging / context injection."""
        body = str(self.content.get("text", self.content))
        if len(body) > max_len:
            body = body[:max_len] + "..."
        return f"[{self.msg_type.value}] {self.sender_id}->{self.receiver_id}: {body}"
