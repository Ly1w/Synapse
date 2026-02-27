"""Deep Research MCP Server.

Exposes all deep_research tools via the Model Context Protocol (MCP)
in stdio transport mode, making them available to Cursor and any
MCP-compatible client.

Usage:
    python mcp_server.py

Cursor MCP config (~/.cursor/mcp.json or .cursor/mcp.json):
    {
      "mcpServers": {
        "deep-research": {
          "command": "python",
          "args": ["/volume/pt-coder/users/ywli/agent_framework/mcp_server.py"]
        }
      }
    }
"""

import sys
import os

# Ensure the package is importable when run directly
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

from deep_research.search import search_web, search_papers
from deep_research.fetch import fetch_webpage, fetch_paper
from deep_research.notes import (
    take_note,
    get_notes,
    track_source,
    list_sources,
    set_session,
    get_session,
)
from deep_research.report import write_report, append_section, read_report, list_reports

mcp = FastMCP("deep-research")


# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_search_web(query: str, max_results: int = 5) -> str:
    """Search the web using Serper (Google Search API).

    Returns a JSON array of results, each with title, url, and snippet.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 5).
    """
    return search_web(query, max_results)


@mcp.tool()
def tool_search_papers(query: str, limit: int = 5) -> str:
    """Search academic papers via Semantic Scholar (no API key required).

    Returns a JSON array with title, authors, year, abstract, citation_count,
    paper_id, arxiv_id, and doi for each paper.

    Args:
        query: Academic search query.
        limit: Maximum number of papers to return (default 5, max 100).
    """
    return search_papers(query, limit)


# ---------------------------------------------------------------------------
# Fetch tools
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_fetch_webpage(url: str, max_chars: int = 8000) -> str:
    """Fetch a webpage and return its readable text content.

    Strips navigation, scripts, and boilerplate. Returns the main article
    text. Useful for reading full articles found via search_web.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return (default 8000).
    """
    return fetch_webpage(url, max_chars)


@mcp.tool()
def tool_fetch_paper(paper_id: str) -> str:
    """Fetch detailed paper information from Semantic Scholar.

    Args:
        paper_id: Semantic Scholar paper ID (from search_papers results),
                  or a prefixed ID like "arXiv:2310.06825".
    """
    return fetch_paper(paper_id)


# ---------------------------------------------------------------------------
# Notes tools
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_take_note(key: str, content: str) -> str:
    """Store a research note under a named key.

    Notes are persisted to disk across turns. Overwriting a key replaces
    the previous note.

    Args:
        key: Short identifier (e.g. "main_findings", "gaps", "next_steps").
        content: Note content (plain text or Markdown).
    """
    return take_note(key, content)


@mcp.tool()
def tool_get_notes(filter: str = "") -> str:
    """Retrieve stored research notes, optionally filtered by keyword.

    Args:
        filter: Optional keyword to filter notes by key or content.
                Pass empty string to retrieve all notes.
    """
    return get_notes(filter or None)


@mcp.tool()
def tool_track_source(url: str, excerpt: str, relevance: str = "medium") -> str:
    """Record a source URL with an excerpt for citation tracking.

    Args:
        url: The source URL.
        excerpt: A short quote or summary from the source.
        relevance: Importance rating — "high", "medium", or "low".
    """
    return track_source(url, excerpt, relevance)


@mcp.tool()
def tool_list_sources(relevance_filter: str = "") -> str:
    """List all tracked sources, optionally filtered by relevance.

    Args:
        relevance_filter: Filter by "high", "medium", "low", or pass
                          empty string for all sources.
    """
    return list_sources(relevance_filter or None)


@mcp.tool()
def tool_set_session(session_id: str) -> str:
    """Set or resume a research session by ID.

    Use this to resume a previous session and access its notes/sources.

    Args:
        session_id: Session identifier, e.g. "session_20240227_143000".
    """
    return set_session(session_id)


@mcp.tool()
def tool_get_session() -> str:
    """Return the current session ID and storage directory path."""
    return get_session()


# ---------------------------------------------------------------------------
# Report tools
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_write_report(title: str, content: str, format: str = "markdown") -> str:
    """Write a research report to disk.

    Args:
        title: Report title (used as filename base).
        content: Full report content.
        format: Output format — "markdown" (default), "html", or "text".
    """
    return write_report(title, content, format)  # type: ignore[arg-type]


@mcp.tool()
def tool_append_section(report_path: str, section_title: str, content: str) -> str:
    """Append a new section to an existing report.

    Useful for multi-agent parallel writing where each agent contributes
    its own chapter to a shared report file.

    Args:
        report_path: Absolute path to the existing report file.
        section_title: Section heading to add.
        content: Section content to append.
    """
    return append_section(report_path, section_title, content)


@mcp.tool()
def tool_read_report(report_path: str) -> str:
    """Read the content of an existing report file.

    Args:
        report_path: Absolute path to the report file.
    """
    return read_report(report_path)


@mcp.tool()
def tool_list_reports() -> str:
    """List all reports saved in the reports directory.

    Returns a JSON array sorted by modification time (newest first).
    """
    return list_reports()


if __name__ == "__main__":
    mcp.run()
