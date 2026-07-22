"""Keyless web search and academic paper search tools."""

import json
import re
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .config import (
    SEMANTIC_SCHOLAR_BASE_URL,
    WEB_SEARCH_URL,
)
from .html_parser import SearchResultsParser


_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.8",
}
_DOMAIN_PATTERN = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$", re.I)


def _normalize_domains(domains: list[str] | None) -> list[str]:
    normalized = []
    for value in domains or []:
        domain = value.strip().lower().rstrip(".")
        if domain.startswith("www."):
            domain = domain[4:]
        if _DOMAIN_PATTERN.fullmatch(domain) and domain not in normalized:
            normalized.append(domain)
    return normalized[:20]


def _domain_matches(hostname: str, domain: str) -> bool:
    hostname = hostname.lower().rstrip(".")
    return hostname == domain or hostname.endswith(f".{domain}")


def _unwrap_result_url(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if parsed.hostname and _domain_matches(parsed.hostname, "duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


def search_web(
    query: str,
    max_results: int = 5,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> str:
    """Search the public web without a search API key.

    The contract mirrors the useful part of Claude Code's WebSearch tool: a query,
    optional domain allow/block lists, and normalized title/URL/snippet results.
    DuckDuckGo's HTML endpoint is used as a replaceable backend rather than scraped
    browser JavaScript.
    """
    query = query.strip()
    if not query:
        return json.dumps({
            "error": "query cannot be blank",
            "query": query,
        })
    if len(query) > 500:
        return json.dumps({"error": "query exceeds 500 characters", "query": query})

    max_results = max(1, min(int(max_results), 10))
    allowed = _normalize_domains(allowed_domains)
    blocked = _normalize_domains(blocked_domains)
    if allowed_domains and not allowed:
        return json.dumps({
            "error": "allowed_domains contains no valid domain names",
            "query": query,
        })
    if allowed and blocked:
        return json.dumps({
            "error": "allowed_domains and blocked_domains cannot be used together",
            "query": query,
        })

    search_query = query
    if allowed:
        scopes = " OR ".join(f"site:{domain}" for domain in allowed)
        search_query = f"{query} ({scopes})"
    if blocked:
        search_query += " " + " ".join(f"-site:{domain}" for domain in blocked)

    try:
        resp = httpx.get(
            WEB_SEARCH_URL,
            params={"q": search_query},
            headers=_SEARCH_HEADERS,
            follow_redirects=True,
            timeout=20,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e), "query": query})

    parser = SearchResultsParser()
    parser.feed(resp.text[:2_000_000])
    parser.close()
    results = []
    for item in parser.results:
        url = _unwrap_result_url(item["url"])
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            continue
        hostname = parsed.hostname.lower()
        if allowed and not any(_domain_matches(hostname, domain) for domain in allowed):
            continue
        if any(_domain_matches(hostname, domain) for domain in blocked):
            continue
        results.append(
            {
                "title": item["title"],
                "url": url,
                "snippet": item["snippet"],
                "metadata": {"source": "duckduckgo"},
            }
        )
        if len(results) >= max_results:
            break

    response: dict[str, object] = {
        "query": query,
        "results": results,
        "total_results": len(results),
    }
    if not results:
        response["warning"] = (
            "The public search backend returned no results and may be rate-limited."
        )
    return json.dumps(response, ensure_ascii=False)


def search_papers(
    query: str,
    limit: int = 5,
    fields: Optional[str] = None,
) -> str:
    """Search academic papers via Semantic Scholar (no API key required).

    Args:
        query: Search terms.
        limit: Maximum number of papers to return (max 100).
        fields: Comma-separated fields to include. Defaults to
                title,authors,year,abstract,citationCount,externalIds.

    Returns:
        JSON string with a list of paper objects.
    """
    if fields is None:
        fields = "title,authors,year,abstract,citationCount,externalIds"

    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": fields,
    }
    try:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e)})

    papers = []
    for item in data.get("data", []):
        authors = [a.get("name", "") for a in item.get("authors", [])]
        paper_id = item.get("paperId", "")
        external = item.get("externalIds", {})
        papers.append(
            {
                "paper_id": paper_id,
                "title": item.get("title", ""),
                "authors": authors,
                "year": item.get("year"),
                "abstract": item.get("abstract", ""),
                "citation_count": item.get("citationCount", 0),
                "arxiv_id": external.get("ArXiv"),
                "doi": external.get("DOI"),
            }
        )

    return json.dumps(papers, ensure_ascii=False)
