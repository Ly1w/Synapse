from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from .channel import Channel
from .message import Message, MessageType

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    MASTER = "master"
    HEAD = "head"
    NODE = "node"


class AgentNode:
    """Represents an agent in the hierarchy tree."""

    def __init__(self, agent_id: str, role: AgentRole, parent_id: str | None = None):
        self.agent_id = agent_id
        self.role = role
        self.parent_id = parent_id
        self.children_ids: list[str] = []


class Router:
    """
    Hierarchy-enforced message router.

    Rules:
    - Master <-> any of its Heads (parent-child)
    - Head <-> its Nodes (parent-child)
    - Head <-> other Heads (peer)
    - Node <-> sibling Nodes under the same Head (peer, same group only)
    - No cross-layer skipping (Node cannot contact Master)
    - No cross-group for Nodes (Node under Head A cannot msg Node under Head B)
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AgentNode] = {}
        self._channels: dict[str, Channel] = {}

    def register(self, agent_id: str, role: AgentRole, parent_id: str | None = None) -> Channel:
        node = AgentNode(agent_id, role, parent_id)
        self._nodes[agent_id] = node
        if parent_id and parent_id in self._nodes:
            self._nodes[parent_id].children_ids.append(agent_id)
        channel = Channel(agent_id)
        self._channels[agent_id] = channel
        logger.info("Router: registered %s (role=%s, parent=%s)", agent_id, role.value, parent_id)
        return channel

    def unregister(self, agent_id: str) -> None:
        node = self._nodes.pop(agent_id, None)
        if node and node.parent_id and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if agent_id in parent.children_ids:
                parent.children_ids.remove(agent_id)
        self._channels.pop(agent_id, None)

    def get_channel(self, agent_id: str) -> Channel | None:
        return self._channels.get(agent_id)

    def _validate_comm(self, sender_id: str, receiver_id: str) -> bool:
        sender = self._nodes.get(sender_id)
        receiver = self._nodes.get(receiver_id)
        if not sender or not receiver:
            return False

        # Parent-child (either direction)
        if sender.parent_id == receiver_id or receiver.parent_id == sender_id:
            return True

        # Head <-> Head peer (both must be HEAD role)
        if sender.role == AgentRole.HEAD and receiver.role == AgentRole.HEAD:
            return True

        # Node <-> Node peer (same parent / same Head group)
        if (sender.role == AgentRole.NODE and receiver.role == AgentRole.NODE
                and sender.parent_id == receiver.parent_id
                and sender.parent_id is not None):
            return True

        return False

    async def send(self, message: Message) -> bool:
        if not self._validate_comm(message.sender_id, message.receiver_id):
            logger.warning(
                "Router: blocked message %s -> %s (hierarchy violation)",
                message.sender_id, message.receiver_id,
            )
            return False

        channel = self._channels.get(message.receiver_id)
        if channel is None:
            logger.warning("Router: no channel for %s", message.receiver_id)
            return False

        await channel.send(message)
        logger.debug("Router: delivered %s", message.summary())
        return True

    def get_peers(self, agent_id: str) -> list[str]:
        """Return IDs of peer agents (same role, valid comm targets)."""
        node = self._nodes.get(agent_id)
        if not node:
            return []

        if node.role == AgentRole.HEAD:
            return [
                aid for aid, n in self._nodes.items()
                if n.role == AgentRole.HEAD and aid != agent_id
            ]
        elif node.role == AgentRole.NODE and node.parent_id:
            parent = self._nodes.get(node.parent_id)
            if parent:
                return [cid for cid in parent.children_ids if cid != agent_id]
        return []

    def get_children(self, agent_id: str) -> list[str]:
        node = self._nodes.get(agent_id)
        return list(node.children_ids) if node else []

    def get_parent(self, agent_id: str) -> str | None:
        node = self._nodes.get(agent_id)
        return node.parent_id if node else None

    def get_all_heads(self) -> list[str]:
        return [aid for aid, n in self._nodes.items() if n.role == AgentRole.HEAD]

    def get_head_roster(self, exclude_id: str | None = None) -> list[dict[str, str]]:
        """Return a brief roster of all Head agents (id + role description placeholder)."""
        roster = []
        for aid, n in self._nodes.items():
            if n.role == AgentRole.HEAD and aid != exclude_id:
                roster.append({"agent_id": aid})
        return roster
