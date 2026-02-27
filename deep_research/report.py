"""Report generation and management tools."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from .config import STORAGE_DIR

_REPORTS_DIR = Path(STORAGE_DIR) / "reports"


def _safe_filename(title: str) -> str:
    """Convert a title to a filesystem-safe filename."""
    safe = re.sub(r"[^\w\s\-]", "", title).strip()
    safe = re.sub(r"\s+", "_", safe)
    return safe[:80] or "report"


def write_report(
    title: str,
    content: str,
    format: Literal["markdown", "html", "text"] = "markdown",
) -> str:
    """Write a research report to disk.

    Args:
        title: Report title (used as filename base).
        content: Full report content.
        format: Output format — "markdown" (.md), "html" (.html), or "text" (.txt).

    Returns:
        JSON string with the saved file path and metadata.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ext_map = {"markdown": ".md", "html": ".html", "text": ".txt"}
    ext = ext_map.get(format, ".md")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_filename(title)}_{timestamp}{ext}"
    filepath = _REPORTS_DIR / filename

    # Prepend a title header for markdown/text reports
    if format == "markdown":
        full_content = f"# {title}\n\n*Generated: {datetime.now().isoformat()}*\n\n{content}"
    elif format == "text":
        full_content = f"{title}\n{'=' * len(title)}\nGenerated: {datetime.now().isoformat()}\n\n{content}"
    else:
        full_content = content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return json.dumps(
        {
            "path": str(filepath),
            "title": title,
            "format": format,
            "size_chars": len(full_content),
            "created_at": datetime.now().isoformat(),
        },
        ensure_ascii=False,
    )


def append_section(
    report_path: str,
    section_title: str,
    content: str,
) -> str:
    """Append a new section to an existing report file.

    Useful when multiple Head agents each write their own chapter.

    Args:
        report_path: Absolute path to the existing report file.
        section_title: Title of the new section (rendered as ## heading in markdown).
        content: Section content to append.

    Returns:
        JSON string with updated file info.
    """
    path = Path(report_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {report_path}"})

    suffix = path.suffix.lower()
    if suffix == ".md":
        section_text = f"\n\n## {section_title}\n\n{content}"
    elif suffix == ".html":
        section_text = f"\n<h2>{section_title}</h2>\n{content}"
    else:
        separator = "-" * 40
        section_text = f"\n\n{separator}\n{section_title}\n{separator}\n\n{content}"

    with open(path, "a", encoding="utf-8") as f:
        f.write(section_text)

    stat = path.stat()
    return json.dumps(
        {
            "path": str(path),
            "section_added": section_title,
            "total_size_chars": stat.st_size,
            "updated_at": datetime.now().isoformat(),
        },
        ensure_ascii=False,
    )


def read_report(report_path: str) -> str:
    """Read the content of an existing report file.

    Args:
        report_path: Absolute path to the report file.

    Returns:
        JSON string with the file content.
    """
    path = Path(report_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {report_path}"})

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return json.dumps(
        {
            "path": str(path),
            "content": content,
            "size_chars": len(content),
        },
        ensure_ascii=False,
    )


def list_reports() -> str:
    """List all reports in the reports directory.

    Returns:
        JSON array of report metadata objects.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reports = []
    for f in sorted(_REPORTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file():
            stat = f.stat()
            reports.append(
                {
                    "path": str(f),
                    "name": f.name,
                    "size_chars": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

    return json.dumps(reports, ensure_ascii=False)
