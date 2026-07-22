"""Filesystem-backed Agent Skills.

Skills are reusable instruction packages rooted at ``<name>/SKILL.md``.  Tool
schemas and tool selection deliberately live in :mod:`agent_framework.tools`.
"""

from .catalog import SkillCatalog, SkillDefinition

__all__ = ["SkillCatalog", "SkillDefinition"]
