from __future__ import annotations

import logging
from typing import Any

from ..core.parsing import extract_json_value
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

DECOMPOSITION_PROMPT = """\
You are helping the Master Agent design a bounded execution plan. Create a Head \
only when a sub-problem needs an accountable owner. Do not split work merely to \
increase parallelism.

Rules:
1. Use 1-4 Heads. Prefer fewer Heads with coherent ownership.
2. Each Head owns an end-to-end deliverable, not just a mechanical step.
3. Identify real dependencies. Independent Heads may run in parallel.
4. Give every Head an explicit scope and acceptance criteria.
5. A Head may solve work itself or create bounded Node assignments later.

User request: {request}

Available tool categories (from skills index):
{tool_categories}

Return a JSON object:
{{
    "sub_tasks": [
        {{
            "role": "short_role_name",
            "description": "Goal owned by this Head",
            "scope": "What is in and out of scope",
            "expected_output": "Concrete deliverable",
            "acceptance_criteria": ["observable completion condition"],
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
            "scope": "explicit boundary",
            "expected_output": "what to produce",
            "acceptance_criteria": ["completion condition"],
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
        plan = extract_json_value(raw, dict)
        if plan and isinstance(plan.get("sub_tasks"), list):
            return TaskDecomposer._normalize_plan(plan)

        logger.warning("Failed to parse decomposition, returning raw")
        return {
            "sub_tasks": [
                {
                    "role": "general_executor",
                    "description": raw,
                    "scope": raw,
                    "expected_output": "task result",
                    "acceptance_criteria": ["Provide a useful result for the request"],
                    "dependencies": [],
                    "suggested_tool_categories": [],
                }
            ],
            "overall_strategy": "Single-agent fallback",
        }

    @staticmethod
    def _normalize_plan(plan: dict[str, Any], max_heads: int = 4) -> dict[str, Any]:
        """Bound fan-out, make roles unique, and remove invalid dependencies."""
        raw_tasks = [item for item in plan.get("sub_tasks", []) if isinstance(item, dict)]
        tasks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_tasks[:max_heads]):
            base_role = str(item.get("role") or f"head_{index + 1}").strip()
            role = base_role
            suffix = 2
            while role in seen:
                role = f"{base_role}_{suffix}"
                suffix += 1
            seen.add(role)
            description = str(item.get("description") or item.get("task") or "").strip()
            tasks.append({
                "role": role,
                "description": description,
                "scope": str(item.get("scope") or description),
                "expected_output": str(item.get("expected_output") or "Task result"),
                "acceptance_criteria": [
                    str(value) for value in item.get("acceptance_criteria", [])
                    if str(value).strip()
                ],
                "dependencies": [str(value) for value in item.get("dependencies", [])],
                "suggested_tool_categories": list(item.get("suggested_tool_categories", [])),
            })

        if not tasks:
            tasks = [{
                "role": "general_owner",
                "description": "Complete the user request",
                "scope": "The full user request",
                "expected_output": "A complete response",
                "acceptance_criteria": ["Address the user request"],
                "dependencies": [],
                "suggested_tool_categories": [],
            }]

        roles = {item["role"] for item in tasks}
        for item in tasks:
            # Self, unknown, and forward-cycle dependencies are not useful to the scheduler.
            item["dependencies"] = list(dict.fromkeys(
                dep for dep in item["dependencies"]
                if dep in roles and dep != item["role"]
            ))

        return {
            "sub_tasks": tasks,
            "overall_strategy": str(plan.get("overall_strategy") or "Bounded hierarchical execution"),
        }
