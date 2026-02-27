"""Deep Research toolkit — tools for web search, academic retrieval,
note-taking, and report generation.

All public functions are importable directly from this package:

    from deep_research import search_web, fetch_webpage, take_note, write_report
"""

from .search import search_web, search_papers
from .fetch import fetch_webpage, fetch_paper
from .notes import (
    take_note,
    get_notes,
    track_source,
    list_sources,
    set_session,
    get_session,
)
from .report import write_report, append_section, read_report, list_reports

__all__ = [
    # search
    "search_web",
    "search_papers",
    # fetch
    "fetch_webpage",
    "fetch_paper",
    # notes
    "take_note",
    "get_notes",
    "track_source",
    "list_sources",
    "set_session",
    "get_session",
    # report
    "write_report",
    "append_section",
    "read_report",
    "list_reports",
]
