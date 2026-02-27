from __future__ import annotations

import json
import logging
from typing import Any

from ..llm.client import LLMClient
from ..tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SKILLS_GENERATION_PROMPT = """\
You are a tool indexing assistant. Given a list of tool definitions (JSON schemas), \
produce a concise categorized index in Markdown format.

Rules:
1. Group tools into logical categories based on their functionality.
2. Each category has a heading: ## Category: <name>
3. Under each heading, write ONE line describing the category purpose.
4. List each tool as: - `tool_name(param1, param2, ...)`: brief description
5. Keep descriptions to ONE sentence each.
6. Do NOT include full parameter schemas — only parameter names.
7. Output ONLY the Markdown, no extra commentary.

Tool definitions:
{tool_definitions}
"""


class SkillsGenerator:
    """Generates a skills.md index from tool definitions via LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def generate(self, registry: ToolRegistry) -> str:
        """Generate skills.md content from the tool registry."""
        schemas = registry.to_json_schemas()
        if not schemas:
            return "# Skills Index\n\nNo tools registered.\n"

        tool_defs_str = json.dumps(schemas, indent=2)
        messages = [
            {"role": "user", "content": SKILLS_GENERATION_PROMPT.format(
                tool_definitions=tool_defs_str
            )},
        ]
        content = await self.llm_client.chat_text(messages, temperature=0.3)
        logger.info("Generated skills.md (%d chars)", len(content))
        return content

    async def generate_from_schemas(self, schemas: list[dict[str, Any]]) -> str:
        """Generate skills.md content directly from a list of JSON schemas."""
        if not schemas:
            return "# Skills Index\n\nNo tools registered.\n"

        tool_defs_str = json.dumps(schemas, indent=2)
        messages = [
            {"role": "user", "content": SKILLS_GENERATION_PROMPT.format(
                tool_definitions=tool_defs_str
            )},
        ]
        content = await self.llm_client.chat_text(messages, temperature=0.3)
        return content

    @staticmethod
    def generate_fallback(registry: ToolRegistry) -> str:
        """
        Generate a basic skills.md without LLM (rule-based fallback).
        Uses categories already assigned in the registry.
        """
        cats = registry.get_categories()
        lines = ["# Skills Index\n"]
        for cat_name, tools in sorted(cats.items()):
            lines.append(f"## Category: {cat_name}\n")
            for tool in tools:
                lines.append(tool.to_index_line())
            lines.append("")
        return "\n".join(lines)
