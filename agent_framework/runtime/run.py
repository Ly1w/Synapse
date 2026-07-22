from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, Field


DEFAULT_RUN_ROOT = os.path.expanduser("~/.agent_framework/runs")
_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


def redact_state(value: Any) -> Any:
    """Recursively hide credential-like fields before state leaves the runtime."""
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
                else redact_state(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_state(item) for item in value]
    return value


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED_RETAINED = "completed_retained"
    FAILED_RETAINED = "failed_retained"
    CANCELLED_RETAINED = "cancelled_retained"
    ARCHIVED = "archived"


class RunManifest(BaseModel):
    run_id: str
    user_request: str
    status: RunStatus = RunStatus.CREATED
    revision: int = 0
    requirements: list[str] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    final_response: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_at: float | None = None


class RunJournal:
    """Append-only event journal plus restorable agent snapshots for one run."""

    def __init__(
        self,
        user_request: str,
        root_dir: str | None = None,
        run_id: str | None = None,
    ):
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.root = Path(root_dir or DEFAULT_RUN_ROOT)
        self.run_dir = self.root / self.run_id
        self.agents_dir = self.run_dir / "agents"
        self.events_file = self.run_dir / "events.jsonl"
        self.manifest_file = self.run_dir / "run.json"
        self.manifest = RunManifest(
            run_id=self.run_id,
            user_request=user_request,
            requirements=[user_request],
        )
        self._lock = asyncio.Lock()
        self._sequence = 0

    @classmethod
    def load_existing(
        cls,
        run_dir: str | Path,
    ) -> "RunJournal" | None:
        """Open a persisted Run for inspection without pretending to restore agents."""
        path = Path(run_dir)
        manifest_file = path / "run.json"
        try:
            manifest = RunManifest.model_validate_json(
                manifest_file.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

        journal = cls.__new__(cls)
        journal.run_id = manifest.run_id
        journal.root = path.parent
        journal.run_dir = path
        journal.agents_dir = path / "agents"
        journal.events_file = path / "events.jsonl"
        journal.manifest_file = manifest_file
        journal.manifest = manifest
        journal._lock = asyncio.Lock()
        journal._sequence = 0
        return journal

    @classmethod
    def discover(cls, root_dir: str | None = None) -> list["RunJournal"]:
        """Find valid persisted Run manifests for the control plane."""
        root = Path(root_dir or DEFAULT_RUN_ROOT)
        if not root.is_dir():
            return []
        journals = []
        try:
            paths = list(root.iterdir())
        except OSError:
            return []
        for path in paths:
            if not path.is_dir():
                continue
            journal = cls.load_existing(path)
            if journal is not None:
                journals.append(journal)
        journals.sort(key=lambda item: item.manifest.created_at)
        return journals

    async def initialize(self) -> None:
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        await self._save_manifest()

    async def set_status(self, status: RunStatus, final_response: str = "") -> None:
        self.manifest.status = status
        self.manifest.updated_at = time.time()
        if final_response:
            self.manifest.final_response = final_response
        if status in {
            RunStatus.COMPLETED_RETAINED,
            RunStatus.FAILED_RETAINED,
            RunStatus.CANCELLED_RETAINED,
        }:
            self.manifest.completed_at = time.time()
        await self._save_manifest()

    async def add_requirement(self, text: str) -> int:
        self.manifest.revision += 1
        self.manifest.requirements.append(text)
        self.manifest.updated_at = time.time()
        await self._save_manifest()
        await self.record("user_update", "user", payload={
            "revision": self.manifest.revision,
            "text": text,
        })
        return self.manifest.revision

    async def register_agent(self, agent_id: str, role: str, parent_id: str = "") -> None:
        if agent_id not in self.manifest.agent_ids:
            self.manifest.agent_ids.append(agent_id)
            self.manifest.updated_at = time.time()
            await self._save_manifest()
        await self.record(
            "agent_registered",
            source=parent_id or "runtime",
            target=agent_id,
            payload={"role": role},
        )

    async def record(
        self,
        event_type: str,
        source: str,
        target: str = "",
        task_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "timestamp": time.time(),
                "type": event_type,
                "source": source,
                "target": target,
                "task_id": task_id,
                "payload": redact_state(payload or {}),
            }
            async with aiofiles.open(self.events_file, "a", encoding="utf-8") as handle:
                await handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def snapshot_agent(self, agent: Any, extra: dict[str, Any] | None = None) -> None:
        state = redact_state(agent.export_state())
        if extra:
            state.update(redact_state(extra))
        path = self.agents_dir / f"{agent.id}.json"
        async with self._lock:
            temp_path = path.with_suffix(f".json.tmp-{uuid.uuid4().hex[:8]}")
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(state, ensure_ascii=False, indent=2, default=str))
            os.replace(temp_path, path)

    async def read_state(self) -> dict[str, Any]:
        return self.manifest.model_dump(mode="json")

    async def read_events(
        self,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Read an ordered event slice for UIs and external observers."""
        if not self.events_file.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            async with aiofiles.open(self.events_file, "r", encoding="utf-8") as handle:
                async for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A process crash may leave only the last line incomplete.
                        continue
                    if int(event.get("sequence", 0)) > after_sequence:
                        events.append(event)
                        if len(events) >= limit:
                            break
        except OSError:
            return []
        return events

    async def read_agent_snapshot(self, agent_id: str) -> dict[str, Any] | None:
        if agent_id not in self.manifest.agent_ids:
            return None
        path = self.agents_dir / f"{agent_id}.json"
        if not path.exists():
            return None
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as handle:
                data = json.loads(await handle.read())
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    async def _save_manifest(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            temp_path = self.manifest_file.with_suffix(
                f".json.tmp-{uuid.uuid4().hex[:8]}"
            )
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as handle:
                await handle.write(self.manifest.model_dump_json(indent=2))
            os.replace(temp_path, self.manifest_file)
