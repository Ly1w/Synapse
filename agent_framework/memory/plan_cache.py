from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiofiles

from ..llm.client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.expanduser("~/.agent_framework/plan_cache")

KEYWORD_EXTRACTION_PROMPT = """\
Extract the core intent keyword(s) from this task query. \
The keyword should capture the high-level task type, not specific details.

Examples:
- "Analyze Q3 sales data for the engineering team" -> "sales_analysis"
- "Search the web for recent papers on LLM agents" -> "web_research"
- "Fix the authentication bug in the login module" -> "bug_fix_authentication"

Task: {task}

Return ONLY the keyword (lowercase, underscores for spaces). No explanation.
"""

PLAN_TEMPLATE_EXTRACTION_PROMPT = """\
Extract a reusable plan template from this execution log. Remove all context-specific \
details (names, numbers, specific values) and keep only the structural plan.

The template should describe:
1. How the task was decomposed into sub-tasks
2. What agent roles were created
3. What types of tools were used at each step
4. The general flow and dependencies

Execution log:
{execution_log}

Return ONLY the plan template in a structured format. Use <PLACEHOLDER> for variable parts.
"""

PLAN_ADAPTATION_PROMPT = """\
Adapt this plan template for the current task. Fill in the <PLACEHOLDER> parts \
with context-specific details from the new task.

Plan template:
{template}

New task: {task}

Additional context: {context}

Return the adapted plan as a structured JSON with:
- "sub_tasks": list of {{role, description, dependencies}}
- "tool_categories": list of tool categories likely needed
"""


class PlanCacheEntry:
    def __init__(self, keyword: str, template: str, usage_count: int = 0,
                 last_used: float = 0.0):
        self.keyword = keyword
        self.template = template
        self.usage_count = usage_count
        self.last_used = last_used or time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "template": self.template,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlanCacheEntry:
        return cls(**d)


class PlanCache:
    """
    Agentic Plan Caching (APC) implementation.

    Extracts, stores, adapts, and reuses structured plan templates
    from successful task executions. Uses keyword extraction for cache
    lookup (exact match) and a lightweight LLM for plan adaptation.
    """

    def __init__(self, llm_client: LLMClient, cache_dir: str | None = None):
        self.llm_client = llm_client
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_file = self.cache_dir / "cache.json"
        self._entries: dict[str, PlanCacheEntry] = {}

    async def initialize(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_file.exists():
            async with aiofiles.open(self.cache_file, "r") as f:
                data = json.loads(await f.read())
            self._entries = {
                k: PlanCacheEntry.from_dict(v) for k, v in data.items()
            }
            logger.info("Loaded %d plan cache entries", len(self._entries))

    async def _save(self) -> None:
        async with aiofiles.open(self.cache_file, "w") as f:
            await f.write(json.dumps(
                {k: v.to_dict() for k, v in self._entries.items()},
                indent=2,
            ))

    async def extract_keyword(self, task: str) -> str:
        """Extract a high-level intent keyword from a task query."""
        response = await self.llm_client.chat_text(
            [{"role": "user", "content": KEYWORD_EXTRACTION_PROMPT.format(task=task)}],
            temperature=0.1,
            max_tokens=64,
        )
        keyword = response.strip().strip('"').strip("'").lower().replace(" ", "_")
        logger.debug("Extracted keyword: %s", keyword)
        return keyword

    async def lookup(self, task: str) -> tuple[str | None, PlanCacheEntry | None]:
        """
        Check cache for a matching plan.
        Returns (keyword, entry) if hit, (keyword, None) if miss.
        """
        keyword = await self.extract_keyword(task)
        entry = self._entries.get(keyword)
        if entry:
            entry.usage_count += 1
            entry.last_used = time.time()
            await self._save()
            logger.info("Plan cache HIT for keyword: %s (usage=%d)",
                        keyword, entry.usage_count)
        else:
            logger.info("Plan cache MISS for keyword: %s", keyword)
        return keyword, entry

    async def adapt_plan(
        self, template: str, task: str, context: str = ""
    ) -> dict[str, Any]:
        """Adapt a cached plan template for a new task using lightweight LLM."""
        response = await self.llm_client.chat_text(
            [{"role": "user", "content": PLAN_ADAPTATION_PROMPT.format(
                template=template, task=task, context=context,
            )}],
            temperature=0.3,
            max_tokens=2048,
        )
        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw_plan": response}

    async def store(self, keyword: str, execution_log: str) -> None:
        """Extract a plan template from execution log and store it."""
        template = await self._extract_template(execution_log)
        self._entries[keyword] = PlanCacheEntry(
            keyword=keyword,
            template=template,
            usage_count=0,
            last_used=time.time(),
        )
        await self._save()
        logger.info("Stored plan template for keyword: %s", keyword)

    async def _extract_template(self, execution_log: str) -> str:
        """Extract a reusable plan template from an execution log."""
        response = await self.llm_client.chat_text(
            [{"role": "user", "content": PLAN_TEMPLATE_EXTRACTION_PROMPT.format(
                execution_log=execution_log,
            )}],
            temperature=0.2,
            max_tokens=2048,
        )
        return response

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "entries": [
                {"keyword": e.keyword, "usage_count": e.usage_count}
                for e in self._entries.values()
            ],
        }
