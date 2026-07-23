from __future__ import annotations

import logging
import re
from typing import Any, Callable

from ..core.parsing import extract_json_value
from ..llm.client import LLMClient
from ..system_prompts import MASTER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_EXPLICIT_HIERARCHY_DIRECTIVE = re.compile(
    r"(?:^|[\n。；;！!])\s*"
    r"(?:请|帮我|我想|please\s+)?"
    r"(?:调用|使用|启用|开启|创建|派出|让|use|invoke|spawn|create|call)\s*"
    r"(?:多个|多|multi[-\s]?|multiple\s+)?"
    r"(?:agent|agents|subagent|subagents|代理|智能体)",
    re.IGNORECASE,
)

DECOMPOSITION_PROMPT = """\
You are routing work for a Master Agent that is the primary executor. Delegation is
optional. The Master can reason, answer, and call available tools itself. Create Heads
only when delegation has a concrete benefit; never create a generic Head merely to
process the request.

Rules:
0. The user's explicit execution preference is authoritative. If Delegation requirement
   below says FORCED, mode must be hierarchical with concrete Head contracts. Never
   silently downgrade an explicit request to use multiple agents into direct execution.
1. Default to mode=direct for conversation, explanation, translation, summarization,
   a cohesive single-owner task, or a task needing only a few ordinary tool calls.
2. Use mode=hierarchical only when one or more of these is true: independent work can
   run in parallel; distinct expertise/context needs isolation; an independent check is
   valuable; or the work is too large for one coherent execution context.
3. Complexity alone does not require delegation. Use 1-4 Heads only when each has a
   real end-to-end responsibility that remains useful after integration.
4. For direct work that can be answered immediately, put the complete user-facing
   answer in direct_response. If tools or further execution are needed, leave
   direct_response empty and describe the work in direct_instruction.
5. For hierarchical work, direct_response must be empty and every Head needs explicit
   scope, deliverable, acceptance criteria, and real dependencies.

User request: {request}

Delegation requirement: {delegation_requirement}

Available tool categories (from the tool registry, not Agent Skills):
{tool_categories}

Return a JSON object:
{{
    "mode": "direct|hierarchical",
    "reason": "why delegation does or does not add value",
    "direct_response": "complete answer when it is already available, otherwise empty",
    "direct_instruction": "what Master should execute directly when tools/work are needed",
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

FORCED_HIERARCHY_REPAIR_PROMPT = """\
Repair an invalid routing decision. The user explicitly required multiple agents, so a
direct plan is not allowed. Create 1-4 concrete, non-overlapping Head contracts that
collectively satisfy the current cumulative request. Use multiple Heads when there are
real independent review or execution tracks; a Head may later decide it needs zero or
more Nodes. Do not answer the user directly.

User request: {request}

Available tool categories: {tool_categories}

Return the same routing JSON schema with mode="hierarchical", an empty
direct_response, and at least one fully specified sub_task.
"""

class TaskDecomposer:
    """Route coherent work to Master or decompose work that benefits from Heads."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = MASTER_SYSTEM_PROMPT,
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt

    async def decompose(
        self,
        request: str,
        tool_categories: list[str] | None = None,
        force_hierarchical: bool | None = None,
        runtime_awareness: str | Callable[[], str] | None = None,
        retrieved_memory: str = "",
    ) -> dict[str, Any]:
        """Choose direct Master execution or a bounded Head task graph."""
        cats_str = ", ".join(tool_categories) if tool_categories else "not specified"
        forced = (
            self.explicit_hierarchy_requested(request)
            if force_hierarchical is None
            else force_hierarchical
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._runtime_messages(runtime_awareness),
            *self._memory_messages(retrieved_memory),
            {"role": "user", "content": DECOMPOSITION_PROMPT.format(
                request=request,
                tool_categories=cats_str,
                delegation_requirement=(
                    "FORCED by the user's explicit request"
                    if forced else "AUTO; use the smallest useful organization"
                ),
            )},
        ]
        raw = await self.llm_client.chat_text(messages, temperature=0.2, max_tokens=1200)
        plan = self._parse_plan(raw)
        if not forced or (plan.get("mode") == "hierarchical" and plan.get("sub_tasks")):
            return plan

        repair_messages = [
            {"role": "system", "content": self.system_prompt},
            *self._runtime_messages(runtime_awareness),
            *self._memory_messages(retrieved_memory),
            {"role": "user", "content": FORCED_HIERARCHY_REPAIR_PROMPT.format(
                request=request,
                tool_categories=cats_str,
            )},
        ]
        repaired_raw = await self.llm_client.chat_text(
            repair_messages,
            temperature=0.1,
            max_tokens=1600,
        )
        repaired = self._parse_plan(repaired_raw)
        if repaired.get("mode") != "hierarchical" or not repaired.get("sub_tasks"):
            logger.error(
                "Planner ignored an explicit multi-agent request twice; using the "
                "runtime's bounded primary-plus-verification hierarchy"
            )
            return self._forced_hierarchy_fallback(request)
        return repaired

    @staticmethod
    def explicit_hierarchy_requested(text: str) -> bool:
        """Recognize an explicit runtime-control directive, not general topic mentions."""
        return bool(_EXPLICIT_HIERARCHY_DIRECTIVE.search(text.strip()))

    @staticmethod
    def _runtime_messages(
        awareness: str | Callable[[], str] | None,
    ) -> list[dict[str, str]]:
        if awareness is None:
            return []
        content = awareness() if callable(awareness) else awareness
        return [{"role": "system", "content": content}] if content else []

    @staticmethod
    def _memory_messages(memory: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": memory}] if memory else []

    @staticmethod
    def _forced_hierarchy_fallback(request: str) -> dict[str, Any]:
        """Honor explicit delegation even when the routing model refuses its schema."""
        return {
            "mode": "hierarchical",
            "reason": (
                "The routing model twice failed to honor explicit delegation. The runtime "
                "selected a bounded primary-owner plus independent-verifier topology rather "
                "than silently downgrading to Master-only execution or failing the Run."
            ),
            "direct_response": "",
            "direct_instruction": "",
            "sub_tasks": [
                {
                    "role": "primary_owner",
                    "description": request,
                    "scope": (
                        "Own the requested work end to end. Stay within the user's cumulative "
                        "requirements and produce concrete evidence, not process narration."
                    ),
                    "expected_output": "A complete evidence-backed primary deliverable",
                    "acceptance_criteria": [
                        "Every cumulative user requirement is addressed",
                        "Material claims or changes are supported by concrete evidence",
                    ],
                    "dependencies": [],
                    "suggested_tool_categories": [],
                },
                {
                    "role": "independent_verifier",
                    "description": (
                        "Independently verify the primary result against the user's cumulative "
                        "requirements; identify concrete errors, omissions, or unsupported claims."
                    ),
                    "scope": (
                        "Verification and correction only. Do not duplicate the primary owner's "
                        "full execution unless evidence must be checked independently."
                    ),
                    "expected_output": "An evidence-backed verification with necessary corrections",
                    "acceptance_criteria": [
                        "Check the primary result against every user requirement",
                        "Report specific corrections or explicitly justify that none are needed",
                    ],
                    "dependencies": ["primary_owner"],
                    "suggested_tool_categories": [],
                },
            ],
            "overall_strategy": (
                "One accountable owner completes the work; a dependent verifier checks it "
                "before Master integration."
            ),
            "routing_fallback": "forced_hierarchy_after_two_invalid_model_decisions",
        }

    @staticmethod
    def _parse_plan(raw: str) -> dict[str, Any]:
        plan = extract_json_value(raw, dict)
        if plan and isinstance(plan.get("sub_tasks"), list):
            return TaskDecomposer._normalize_plan(plan)

        logger.warning("Failed to parse routing decision; keeping work at Master")
        return {
            "mode": "direct",
            "reason": "Routing output was malformed; do not create speculative agents.",
            "direct_response": "",
            "direct_instruction": "Complete the original user request directly.",
            "sub_tasks": [],
            "overall_strategy": "Master direct fallback",
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

        mode = str(plan.get("mode") or ("hierarchical" if tasks else "direct")).lower()
        if mode not in {"direct", "hierarchical"}:
            mode = "direct"
        direct_response = str(plan.get("direct_response") or "").strip()
        direct_instruction = str(plan.get("direct_instruction") or "").strip()
        if mode == "direct":
            tasks = []
        elif not tasks:
            # A hierarchy without owned work is not a valid topology.
            mode = "direct"
            direct_instruction = direct_instruction or "Complete the original request directly."

        roles = {item["role"] for item in tasks}
        for item in tasks:
            # Self, unknown, and forward-cycle dependencies are not useful to the scheduler.
            item["dependencies"] = list(dict.fromkeys(
                dep for dep in item["dependencies"]
                if dep in roles and dep != item["role"]
            ))

        return {
            "mode": mode,
            "reason": str(plan.get("reason") or ""),
            "sub_tasks": tasks,
            "overall_strategy": str(plan.get("overall_strategy") or "Bounded hierarchical execution"),
            "direct_response": direct_response,
            "direct_instruction": direct_instruction,
        }
