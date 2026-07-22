from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ToolDef(BaseModel):
    """Internal representation of a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    category: str = ""
    source: str = "custom"
    permission_scope: str = "safe"
    callable: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function-calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

class ToolRegistry:
    """
    Central registry for tool definitions and their callable implementations.

    Tools can be registered from JSON schemas (with optional callable) or
    from decorated Python functions.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        category: str = "",
        callable_fn: Callable[..., Any] | None = None,
        source: str = "custom",
        permission_scope: str = "safe",
    ) -> ToolDef:
        if name.startswith("framework_"):
            raise ValueError("Tool names beginning with 'framework_' are reserved")
        if name in self._tools:
            raise ValueError(f"Tool name is already registered: {name}")
        tool = ToolDef(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            category=category,
            source=source,
            permission_scope=permission_scope,
            callable=callable_fn,
        )
        self._tools[name] = tool
        logger.info("Registered tool: %s (category=%s)", name, category)
        return tool

    def register_from_schema(
        self,
        schema: dict[str, Any],
        category: str = "",
        callable_fn: Callable[..., Any] | None = None,
        source: str = "custom",
        permission_scope: str = "safe",
    ) -> ToolDef:
        """Register a tool from an OpenAI-style function schema."""
        func = schema.get("function", schema)
        return self.register(
            name=func["name"],
            description=func.get("description", ""),
            parameters=func.get("parameters"),
            category=category or str(schema.get("category") or func.get("category") or ""),
            callable_fn=callable_fn,
            source=source,
            permission_scope=permission_scope,
        )

    def register_batch(
        self,
        schemas: list[dict[str, Any]],
        callables: dict[str, Callable[..., Any]] | None = None,
    ) -> list[ToolDef]:
        """Register multiple tools from a list of OpenAI-style schemas."""
        callables = callables or {}
        results = []
        for schema in schemas:
            func = schema.get("function", schema)
            name = func["name"]
            existing = self.get(name)
            if existing is not None:
                if existing.source == "builtin" or existing.source.startswith("mcp:"):
                    raise ValueError(f"Tool name is reserved by {existing.source}: {name}")
                self.unregister(name)
            results.append(
                self.register_from_schema(schema, callable_fn=callables.get(name))
            )
        return results

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_by_names(self, names: list[str]) -> list[ToolDef]:
        return [self._tools[n] for n in names if n in self._tools]

    def get_by_category(self, category: str) -> list[ToolDef]:
        return [t for t in self._tools.values() if t.category == category]

    def get_all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def get_categories(self) -> dict[str, list[ToolDef]]:
        cats: dict[str, list[ToolDef]] = {}
        for t in self._tools.values():
            cats.setdefault(t.category or "uncategorized", []).append(t)
        return cats

    def get_openai_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        tools = self.get_by_names(names) if names else self.get_all()
        return [t.to_openai_schema() for t in tools]

    def get_callable_tools(self) -> dict[str, Callable[..., Any]]:
        """Return a dict of name->callable for tools that have implementations."""
        return {
            name: t.callable
            for name, t in self._tools.items()
            if t.callable is not None
        }

    def to_json_schemas(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._tools.values()]
