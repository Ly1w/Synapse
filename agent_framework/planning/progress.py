from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskProgress(BaseModel):
    """Progress node in the task tree."""

    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str = ""
    result_summary: str = ""
    sub_tasks: list[TaskProgress] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    completed_at: float | None = None

    def mark_in_progress(self) -> None:
        self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, summary: str = "") -> None:
        self.status = TaskStatus.COMPLETED
        self.result_summary = summary
        self.completed_at = time.time()

    def mark_failed(self, reason: str = "") -> None:
        self.status = TaskStatus.FAILED
        self.result_summary = reason
        self.completed_at = time.time()

    def add_sub_task(self, task: TaskProgress) -> None:
        self.sub_tasks.append(task)

    def find_task(self, task_id: str) -> TaskProgress | None:
        if self.task_id == task_id:
            return self
        for st in self.sub_tasks:
            found = st.find_task(task_id)
            if found:
                return found
        return None

    @property
    def completion_ratio(self) -> float:
        if not self.sub_tasks:
            return 1.0 if self.status == TaskStatus.COMPLETED else 0.0
        completed = sum(1 for t in self.sub_tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.sub_tasks)


class ProgressTracker:
    """Tracks overall task progress across the agent hierarchy."""

    def __init__(self) -> None:
        self.root: TaskProgress | None = None

    def create_root(self, task_id: str, description: str) -> TaskProgress:
        self.root = TaskProgress(task_id=task_id, description=description)
        return self.root

    def find_task(self, task_id: str) -> TaskProgress | None:
        return self.root.find_task(task_id) if self.root else None

    def update_status(
        self, task_id: str, status: TaskStatus, summary: str = ""
    ) -> bool:
        task = self.find_task(task_id)
        if not task:
            return False
        if status == TaskStatus.COMPLETED:
            task.mark_completed(summary)
        elif status == TaskStatus.FAILED:
            task.mark_failed(summary)
        elif status == TaskStatus.IN_PROGRESS:
            task.mark_in_progress()
        else:
            task.status = status
        return True

    def generate_report(self) -> str:
        """Generate a formatted progress report."""
        if not self.root:
            return "No tasks tracked."
        return self._format_task(self.root, indent=0)

    def _format_task(self, task: TaskProgress, indent: int) -> str:
        prefix = "  " * indent
        status_icon = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.IN_PROGRESS: "[~]",
            TaskStatus.COMPLETED: "[x]",
            TaskStatus.FAILED: "[!]",
        }.get(task.status, "[?]")

        lines = [f"{prefix}{status_icon} {task.description}"]
        if task.assigned_to:
            lines[0] += f"  (agent: {task.assigned_to})"
        if task.result_summary:
            lines.append(f"{prefix}    Result: {task.result_summary[:200]}")
        if task.sub_tasks:
            ratio = task.completion_ratio
            lines.append(f"{prefix}    Progress: {ratio:.0%}")
            for st in task.sub_tasks:
                lines.append(self._format_task(st, indent + 1))
        return "\n".join(lines)

    def to_execution_log(self) -> str:
        """Serialize the full progress tree for plan cache extraction."""
        if not self.root:
            return ""
        return self._log_task(self.root)

    def _log_task(self, task: TaskProgress) -> str:
        parts = [
            f"Task: {task.description}",
            f"  Role: {task.assigned_to}",
            f"  Status: {task.status.value}",
        ]
        if task.result_summary:
            parts.append(f"  Result: {task.result_summary[:500]}")
        for st in task.sub_tasks:
            parts.append(self._log_task(st))
        return "\n".join(parts)
