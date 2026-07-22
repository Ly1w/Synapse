"""Dedicated keyless Web Search MCP server used by the Synapse runtime."""

from mcp.server.fastmcp import FastMCP

from deep_research.fetch import fetch_webpage
from deep_research.search import search_web

mcp = FastMCP("synapse-web-search")


@mcp.tool()
def tool_search_web(
    query: str,
    max_results: int = 5,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> str:
    """Search the public web without a search API key.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 5, max 10).
        allowed_domains: If set, only return these domains and their subdomains.
        blocked_domains: Never return these domains or their subdomains. This cannot
            be combined with allowed_domains.
    """
    return search_web(query, max_results, allowed_domains, blocked_domains)


@mcp.tool()
def tool_fetch_webpage(url: str, max_chars: int = 8000) -> str:
    """Fetch readable text from one public HTTP(S) webpage.

    Local/private network targets and credential-bearing URLs are rejected.

    Args:
        url: Absolute public HTTP(S) URL.
        max_chars: Maximum characters to return (default and hard limit 8000).
    """
    return fetch_webpage(url, max_chars)


if __name__ == "__main__":
    mcp.run()
