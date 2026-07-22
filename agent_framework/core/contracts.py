from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutcomeStatus(str, Enum):
    """Terminal state for an agent-owned task."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentBudget(BaseModel):
    """Hard limits that keep an agent from expanding or chatting forever."""

    max_turns: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=12, ge=0, le=200)
    max_peer_messages: int = Field(default=2, ge=0, le=20)
    max_discoveries: int = Field(default=1, ge=0, le=10)
    max_children: int = Field(default=4, ge=0, le=20)
    max_revision_rounds: int = Field(default=1, ge=0, le=5)
    timeout_seconds: float = Field(default=300.0, gt=0)


class TaskContract(BaseModel):
    """
    The task boundary agreed between a parent and a child agent.

    A contract carries both the work and its stopping condition.  Children may
    report discoveries, but they do not get to redefine this contract.
    """

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    role: str = "executor"
    goal: str
    scope: str = ""
    deliverable: str = "A concise, evidence-backed result"
    acceptance_criteria: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

    def to_prompt(self) -> str:
        criteria = "\n".join(f"- {item}" for item in self.acceptance_criteria)
        if not criteria:
            criteria = "- Address the stated goal within scope."
        context = "\n".join(f"- {key}: {value}" for key, value in self.context.items())
        if not context:
            context = "- None"
        return (
            f"Task id: {self.task_id}\n"
            f"Role: {self.role}\n"
            f"Goal: {self.goal}\n"
            f"Scope: {self.scope or self.goal}\n"
            f"Deliverable: {self.deliverable}\n"
            f"Acceptance criteria:\n{criteria}\n"
            f"Provided context:\n{context}"
        )


class AgentOutcome(BaseModel):
    """Structured result returned by every agent, including failed agents."""

    agent_id: str
    task_id: str
    status: OutcomeStatus
    summary: str
    evidence: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.status == OutcomeStatus.COMPLETED

    def to_message_content(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        # Keep the legacy key so older observers still have useful text.
        data["text"] = self.summary
        return data

