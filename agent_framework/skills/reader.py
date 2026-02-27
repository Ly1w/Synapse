from __future__ import annotations

import logging
import re
from typing import Any

from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

TOOL_SELECTION_PROMPT = """\
Given the following task and a skills index, identify which tools are needed.

Task: {task}

Skills Index:
{skills_content}

Return ONLY a JSON array of tool names that are relevant. Example: ["tool_a", "tool_b"]
"""


class SkillsReader:
    """Reads skills.md index and retrieves relevant tool definitions."""

    def __init__(
        self,
        skills_content: str,
        registry: ToolRegistry,
        llm_client: LLMClient | None = None,
    ):
        self.skills_content = skills_content
        self.registry = registry
        self.llm_client = llm_client
        self._parsed_categories = self._parse_categories()

    def _parse_categories(self) -> dict[str, list[str]]:
        """Parse skills.md into {category: [tool_names]}."""
        categories: dict[str, list[str]] = {}
        current_cat = "uncategorized"
        for line in self.skills_content.splitlines():
            cat_match = re.match(r"^##\s+Category:\s*(.+)$", line.strip())
            if cat_match:
                current_cat = cat_match.group(1).strip()
                categories.setdefault(current_cat, [])
                continue
            tool_match = re.match(r"^-\s+`(\w+)\(", line.strip())
            if tool_match:
                categories.setdefault(current_cat, []).append(tool_match.group(1))
        return categories

    def get_all_tool_names(self) -> list[str]:
        names: list[str] = []
        for tools in self._parsed_categories.values():
            names.extend(tools)
        return names

    def get_tools_by_category(self, category: str) -> list[dict[str, Any]]:
        """Get full OpenAI schemas for tools in a given category."""
        names = self._parsed_categories.get(category, [])
        return self.registry.get_openai_schemas(names)

    def get_categories_list(self) -> list[str]:
        return list(self._parsed_categories.keys())

    async def find_tools(self, task_description: str) -> list[dict[str, Any]]:
        """
        Use LLM (if available) or keyword matching to find relevant tools
        for a given task, returning full OpenAI schemas.
        """
        if self.llm_client:
            return await self._llm_find(task_description)
        return self._keyword_find(task_description)

    async def _llm_find(self, task_description: str) -> list[dict[str, Any]]:
        messages = [
            {"role": "user", "content": TOOL_SELECTION_PROMPT.format(
                task=task_description,
                skills_content=self.skills_content,
            )},
        ]
        raw = await self.llm_client.chat_text(messages, temperature=0.1)

        # Extract JSON array from response
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            import json
            try:
                names = json.loads(match.group())
                if isinstance(names, list):
                    return self.registry.get_openai_schemas(names)
            except Exception:
                pass

        logger.warning("LLM tool selection failed, falling back to keyword match")
        return self._keyword_find(task_description)

    def _keyword_find(self, task_description: str) -> list[dict[str, Any]]:
        """Simple keyword-based tool matching as fallback."""
        task_lower = task_description.lower()
        matched: list[str] = []

        for cat, tools in self._parsed_categories.items():
            if cat.lower() in task_lower:
                matched.extend(tools)
                continue
            for tool_name in tools:
                tool_def = self.registry.get(tool_name)
                if tool_def and (
                    tool_name.lower() in task_lower
                    or any(w in task_lower for w in tool_def.description.lower().split()
                           if len(w) > 4)
                ):
                    matched.append(tool_name)

        if not matched:
            matched = self.get_all_tool_names()

        return self.registry.get_openai_schemas(list(set(matched)))
