from __future__ import annotations

import html
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)

_MAX_SKILL_CHARS = 100_000
_MAX_DESCRIPTION_CHARS = 1_024
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_POSITIONAL_ARGUMENT = re.compile(r"\$(\d+)")


@dataclass(frozen=True)
class SkillDefinition:
    """One discovered Agent Skill and its parsed invocation metadata."""

    name: str
    description: str
    path: Path
    source: str
    body: str
    frontmatter: dict[str, Any]
    model_invocable: bool = True
    user_invocable: bool = True
    allowed_tools: tuple[str, ...] = ()
    context: str = "inline"
    agent: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "path": str(self.path),
            "model_invocable": self.model_invocable,
            "user_invocable": self.user_invocable,
            "allowed_tools": list(self.allowed_tools),
            "context": self.context,
            "agent": self.agent,
        }


@dataclass(frozen=True)
class _SkillRoot:
    path: Path
    source: str


class SkillCatalog:
    """Discover metadata eagerly and load SKILL.md bodies only on invocation.

    Roots are ordered from lowest to highest precedence. A later skill with the
    same declared name replaces an earlier one. The default order is bundled,
    project, then user, with more deeply nested project roots overriding parents.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        roots: Iterable[str | Path] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if roots is None:
            self._roots = self._default_roots()
        else:
            self._roots = [
                _SkillRoot(Path(root).expanduser().resolve(), "configured")
                for root in roots
            ]

    def discover(self) -> dict[str, SkillDefinition]:
        discovered: dict[str, SkillDefinition] = {}
        for root in self._roots:
            if not root.path.is_dir():
                continue
            for path in sorted(root.path.glob("*/SKILL.md")):
                try:
                    definition = self._parse(path, root.source)
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                    logger.warning("Ignoring invalid Skill %s: %s", path, exc)
                    continue
                previous = discovered.get(definition.name)
                if previous is not None:
                    logger.info(
                        "Skill %s from %s overrides %s",
                        definition.name,
                        path,
                        previous.path,
                    )
                discovered[definition.name] = definition
        return discovered

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            item.metadata()
            for item in sorted(self.discover().values(), key=lambda skill: skill.name)
        ]

    def load(self, name: str, arguments: str = "") -> dict[str, Any]:
        requested = name.strip()
        if requested.lower() == "list":
            return {"skills": self.list_skills()}

        definition = self.discover().get(requested)
        if definition is None:
            return {
                "error": f"skill is not installed: {requested}",
                "available": sorted(self.discover()),
            }
        if not definition.model_invocable:
            return {
                "error": f"skill is not available for model invocation: {requested}",
                "user_invocable": definition.user_invocable,
            }

        rendered = self._render_arguments(definition, arguments)
        result = definition.metadata()
        result.update({
            "base_directory": str(definition.path.parent),
            "arguments": arguments,
            "instructions": rendered,
            "instruction_priority": (
                "user_installed_workflow_below_system_and_current_user_requirements"
            ),
        })
        return result

    def model_inventory(self) -> str:
        """Return bounded metadata suitable for lazy-discovery prompt injection."""
        visible = [
            item
            for item in self.discover().values()
            if item.model_invocable
        ]
        if not visible:
            return "No model-invocable Skills are currently installed."
        lines = []
        for item in sorted(visible, key=lambda skill: skill.name):
            description = " ".join(item.description.split())[:_MAX_DESCRIPTION_CHARS]
            lines.append(
                f'- <skill name="{html.escape(item.name, quote=True)}" '
                f'source="{html.escape(item.source, quote=True)}">'
                f"{html.escape(description)}</skill>"
            )
        return "\n".join(lines)

    def _default_roots(self) -> list[_SkillRoot]:
        roots = [
            _SkillRoot(Path(__file__).resolve().parent / "bundled", "bundled"),
        ]
        for base in self._project_bases():
            roots.extend([
                _SkillRoot(base / ".claude" / "skills", "project-claude"),
                _SkillRoot(base / ".synapse" / "skills", "project"),
            ])
        roots.extend([
            _SkillRoot(Path.home() / ".claude" / "skills", "user-claude"),
            _SkillRoot(Path.home() / ".agent_framework" / "skills", "user"),
        ])
        return roots

    def _project_bases(self) -> list[Path]:
        current = self.workspace_root
        repository_root: Path | None = None
        while True:
            if (current / ".git").exists():
                repository_root = current
                break
            if current.parent == current:
                break
            current = current.parent
        if repository_root is None:
            return [self.workspace_root]

        bases: list[Path] = []
        current = self.workspace_root
        while True:
            bases.append(current)
            if current == repository_root:
                break
            current = current.parent
        bases.reverse()
        return bases

    @staticmethod
    def _parse(path: Path, source: str) -> SkillDefinition:
        raw = path.read_text(encoding="utf-8")
        if len(raw) > _MAX_SKILL_CHARS:
            raise ValueError(f"SKILL.md exceeds {_MAX_SKILL_CHARS} characters")

        frontmatter: dict[str, Any] = {}
        body = raw
        if raw.startswith("---\n") or raw.startswith("---\r\n"):
            lines = raw.splitlines(keepends=True)
            end = next(
                (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
                None,
            )
            if end is None:
                raise ValueError("unterminated YAML frontmatter")
            parsed = yaml.safe_load("".join(lines[1:end])) or {}
            if not isinstance(parsed, dict):
                raise ValueError("YAML frontmatter must be a mapping")
            frontmatter = {str(key): value for key, value in parsed.items()}
            body = "".join(lines[end + 1:]).lstrip("\r\n")

        name = str(frontmatter.get("name") or path.parent.name).strip()
        if not _VALID_NAME.fullmatch(name):
            raise ValueError(f"invalid skill name: {name!r}")
        description = str(frontmatter.get("description") or "").strip()
        if not description:
            description = SkillCatalog._fallback_description(body, name)

        raw_allowed = frontmatter.get("allowed-tools", ())
        if isinstance(raw_allowed, str):
            allowed_tools = tuple(item for item in re.split(r"[\s,]+", raw_allowed) if item)
        elif isinstance(raw_allowed, list):
            allowed_tools = tuple(str(item) for item in raw_allowed)
        else:
            allowed_tools = ()

        context = str(frontmatter.get("context") or "inline").strip()
        if context not in {"inline", "fork"}:
            raise ValueError("frontmatter context must be 'inline' or 'fork'")
        disabled_for_model = SkillCatalog._as_bool(
            frontmatter.get("disable-model-invocation"), False
        )
        return SkillDefinition(
            name=name,
            description=description[:_MAX_DESCRIPTION_CHARS],
            path=path.resolve(),
            source=source,
            body=body,
            frontmatter=frontmatter,
            # context: fork is a Claude Code-specific extension. Synapse does not
            # silently reinterpret it as inline execution or forced delegation.
            model_invocable=not disabled_for_model and context == "inline",
            user_invocable=SkillCatalog._as_bool(
                frontmatter.get("user-invocable"), True
            ),
            allowed_tools=allowed_tools,
            context=context,
            agent=str(frontmatter.get("agent") or "").strip(),
        )

    @staticmethod
    def _fallback_description(body: str, name: str) -> str:
        for line in body.splitlines():
            candidate = line.strip().lstrip("#").strip()
            if candidate:
                return candidate
        return f"Instructions provided by the {name} Skill."

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "on", "1"}:
                return True
            if lowered in {"false", "no", "off", "0"}:
                return False
        raise ValueError(f"expected a boolean frontmatter value, got {value!r}")

    @staticmethod
    def _render_arguments(definition: SkillDefinition, arguments: str) -> str:
        rendered = definition.body.replace("$ARGUMENTS", arguments)
        try:
            positional = shlex.split(arguments)
        except ValueError:
            positional = arguments.split()

        def replace_position(match: re.Match[str]) -> str:
            index = int(match.group(1))
            return positional[index] if index < len(positional) else ""

        rendered = _POSITIONAL_ARGUMENT.sub(replace_position, rendered)
        rendered = rendered.replace("${SKILL_DIR}", str(definition.path.parent))
        return rendered
