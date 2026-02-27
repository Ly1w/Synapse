from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base_agent import BaseAgent


class AgentRegistry:
    """
    Central registry for all active agents.
    Tracks agent instances and provides lookup by ID.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    @staticmethod
    def generate_id(prefix: str = "agent") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def register(self, agent: BaseAgent, metadata: dict[str, Any] | None = None) -> None:
        self._agents[agent.id] = agent
        if metadata:
            self._metadata[agent.id] = metadata

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
        self._metadata.pop(agent_id, None)

    def get(self, agent_id: str) -> BaseAgent | None:
        return self._agents.get(agent_id)

    def get_metadata(self, agent_id: str) -> dict[str, Any]:
        return self._metadata.get(agent_id, {})

    def set_metadata(self, agent_id: str, key: str, value: Any) -> None:
        self._metadata.setdefault(agent_id, {})[key] = value

    def get_all_ids(self) -> list[str]:
        return list(self._agents.keys())

    @property
    def count(self) -> int:
        return len(self._agents)
