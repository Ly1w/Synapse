from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """\
You are a task decomposition expert. Break down the user's request into sub-tasks, \
each to be handled by a separate Head Agent (team lead).

Rules:
1. Each sub-task should be independent enough for a separate team to handle.
2. Identify dependencies between sub-tasks if any.
3. Assign a clear role name to each Head Agent.
4. Be specific about what each Head is responsible for.

User request: {request}

Available tool categories (from skills index):
{tool_categories}

Return a JSON object:
{{
    "sub_tasks": [
        {{
            "role": "short_role_name",
            "description": "What this head agent is responsible for",
            "expected_output": "What this head should produce",
            "dependencies": ["role_name_of_dependency"],
            "suggested_tool_categories": ["category1", "category2"]
        }}
    ],
    "overall_strategy": "Brief description of the approach"
}}
"""

ADAPTED_DECOMPOSITION_PROMPT = """\
Adapt this plan template for the current task. Fill in specifics.

Plan template:
{template}

User request: {request}

Available tool categories:
{tool_categories}

Return a JSON object with the same structure as the template but with specifics filled in:
{{
    "sub_tasks": [
        {{
            "role": "short_role_name",
            "description": "specific description",
            "expected_output": "what to produce",
            "dependencies": [],
            "suggested_tool_categories": []
        }}
    ],
    "overall_strategy": "approach description"
}}
"""


class TaskDecomposer:
    """Decomposes user requests into sub-tasks for Head Agents."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def decompose(
        self,
        request: str,
        tool_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Decompose a user request into sub-tasks from scratch."""
        cats_str = ", ".join(tool_categories) if tool_categories else "not specified"
        messages = [
            {"role": "user", "content": DECOMPOSITION_PROMPT.format(
                request=request,
                tool_categories=cats_str,
            )},
        ]
        raw = await self.llm_client.chat_text(messages, temperature=0.4)
        return self._parse_plan(raw)

    async def decompose_from_template(
        self,
        request: str,
        template: str,
        tool_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Adapt a cached plan template for a new request."""
        cats_str = ", ".join(tool_categories) if tool_categories else "not specified"
        messages = [
            {"role": "user", "content": ADAPTED_DECOMPOSITION_PROMPT.format(
                template=template,
                request=request,
                tool_categories=cats_str,
            )},
        ]
        raw = await self.llm_client.chat_text(messages, temperature=0.3)
        return self._parse_plan(raw)

    @staticmethod
    def _parse_plan(raw: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                plan = json.loads(match.group())
                if "sub_tasks" in plan:
                    return plan
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse decomposition, returning raw")
        return {
            "sub_tasks": [
                {
                    "role": "general_executor",
                    "description": raw,
                    "expected_output": "task result",
                    "dependencies": [],
                    "suggested_tool_categories": [],
                }
            ],
            "overall_strategy": "Single-agent fallback",
        }
