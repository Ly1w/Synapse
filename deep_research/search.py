"""Web and academic paper search tools."""

import json
from typing import Optional

import httpx

from .config import (
    SERPER_API_KEY,
    SERPER_BASE_URL,
    SEMANTIC_SCHOLAR_BASE_URL,
)


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web via Serper (Google Search API).

    Returns a JSON string containing a list of results, each with:
      - title: page title
      - url: page URL
      - snippet: short excerpt
    """
    payload = {"q": query, "num": max_results}
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(
            f"{SERPER_BASE_URL}/search",
            json=payload,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e)})

    results = []
    for item in data.get("organic", [])[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    # Also include knowledgeGraph if present
    if "knowledgeGraph" in data:
        kg = data["knowledgeGraph"]
        results.insert(
            0,
            {
                "title": kg.get("title", ""),
                "url": kg.get("website", ""),
                "snippet": kg.get("description", ""),
                "type": "knowledge_graph",
            },
        )

    return json.dumps(results, ensure_ascii=False)


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
