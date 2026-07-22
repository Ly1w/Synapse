"""Research notes and source tracking tools.

Notes and sources are persisted to ~/.deep_research/<session>/
so they survive across agent turns within the same research session.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import STORAGE_DIR

_SESSION_ID: Optional[str] = None
_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_PATTERN.fullmatch(session_id) or session_id in {".", ".."}:
        raise ValueError("Invalid session id")


def _session_dir() -> Path:
    """Return (and create) the current session directory."""
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    _validate_session_id(_SESSION_ID)
    root = Path(STORAGE_DIR).expanduser().resolve()
    path = (root / _SESSION_ID).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Session path escapes storage directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _notes_path() -> Path:
    return _session_dir() / "notes.json"


def _sources_path() -> Path:
    return _session_dir() / "sources.json"


def _load_json(path: Path) -> dict | list:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(path: Path, data: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_session(session_id: str) -> str:
    """Explicitly set the session ID (useful to resume a previous session).

    Args:
        session_id: Session identifier, e.g. "session_20240227_143000".

    Returns:
        Confirmation message with the session directory path.
    """
    global _SESSION_ID
    _validate_session_id(session_id)
    _SESSION_ID = session_id
    path = _session_dir()
    return f"Session set to '{session_id}'. Storage: {path}"


def get_session() -> str:
    """Return the current session ID and storage directory path."""
    path = _session_dir()
    return json.dumps({"session_id": _SESSION_ID, "storage_dir": str(path)})


def take_note(key: str, content: str) -> str:
    """Store a research note under a named key.

    Overwrites any existing note with the same key.

    Args:
        key: Short identifier for the note (e.g. "main_findings", "gap_analysis").
        content: The note content (plain text or Markdown).

    Returns:
        Confirmation message.
    """
    path = _notes_path()
    notes: dict = _load_json(path)  # type: ignore[assignment]
    notes[key] = {
        "content": content,
        "updated_at": datetime.now().isoformat(),
    }
    _save_json(path, notes)
    return f"Note '{key}' saved ({len(content)} chars). Session: {_SESSION_ID}"


def get_notes(filter: Optional[str] = None) -> str:
    """Retrieve stored notes, optionally filtered by keyword.

    Args:
        filter: Optional keyword. Only notes whose key or content
                contains this string (case-insensitive) are returned.

    Returns:
        JSON string mapping note key → {content, updated_at}.
    """
    path = _notes_path()
    notes: dict = _load_json(path)  # type: ignore[assignment]

    if filter:
        kw = filter.lower()
        notes = {
            k: v
            for k, v in notes.items()
            if kw in k.lower() or kw in v.get("content", "").lower()
        }

    return json.dumps(notes, ensure_ascii=False)


def track_source(url: str, excerpt: str, relevance: str = "medium") -> str:
    """Record a source URL with an excerpt and relevance rating.

    Args:
        url: The source URL.
        excerpt: A short quote or summary from the source.
        relevance: One of "high", "medium", "low".

    Returns:
        Confirmation message and current total source count.
    """
    path = _sources_path()
    sources: list = _load_json(path)  # type: ignore[assignment]
    if not isinstance(sources, list):
        sources = []

    entry = {
        "url": url,
        "excerpt": excerpt,
        "relevance": relevance,
        "added_at": datetime.now().isoformat(),
    }
    # Avoid duplicates by URL
    existing_urls = {s["url"] for s in sources}
    if url not in existing_urls:
        sources.append(entry)
        _save_json(path, sources)

    return json.dumps(
        {
            "status": "tracked" if url not in existing_urls else "already_exists",
            "total_sources": len(sources),
            "session": _SESSION_ID,
        }
    )


def list_sources(relevance_filter: Optional[str] = None) -> str:
    """List all tracked sources, optionally filtered by relevance level.

    Args:
        relevance_filter: One of "high", "medium", "low", or None for all.

    Returns:
        JSON array of source entries.
    """
    path = _sources_path()
    sources: list = _load_json(path)  # type: ignore[assignment]
    if not isinstance(sources, list):
        sources = []

    if relevance_filter:
        sources = [s for s in sources if s.get("relevance") == relevance_filter]

    return json.dumps(sources, ensure_ascii=False)
