"""Webpage and academic paper content fetching tools."""

import json
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from .config import SEMANTIC_SCHOLAR_BASE_URL

_MAX_WEBPAGE_CHARS = 8000

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _extract_text(html: str) -> str:
    """Extract clean readable text from HTML, stripping boilerplate."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Prefer article/main content if available
    main = soup.find("article") or soup.find("main") or soup.find("body") or soup

    paragraphs = []
    for elem in main.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
        text = elem.get_text(separator=" ", strip=True)
        if len(text) > 30:
            paragraphs.append(text)

    text = "\n\n".join(paragraphs)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def fetch_webpage(url: str, max_chars: int = _MAX_WEBPAGE_CHARS) -> str:
    """Fetch a webpage and return its readable text content.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return (default 8000).

    Returns:
        JSON string with keys:
          - url: the fetched URL
          - title: page title
          - content: extracted text (truncated to max_chars)
          - truncated: whether content was cut off
    """
    try:
        resp = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=20)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e), "url": url})

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return json.dumps(
            {"error": f"Non-HTML content type: {content_type}", "url": url}
        )

    soup = BeautifulSoup(resp.text, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    text = _extract_text(resp.text)
    truncated = len(text) > max_chars
    return json.dumps(
        {
            "url": url,
            "title": title,
            "content": text[:max_chars],
            "truncated": truncated,
        },
        ensure_ascii=False,
    )


def fetch_paper(
    paper_id: str,
    fields: Optional[str] = None,
) -> str:
    """Fetch detailed information about a paper from Semantic Scholar.

    Args:
        paper_id: Semantic Scholar paper ID, or prefixed ID such as
                  "arXiv:2310.06825" or "DOI:10.18653/v1/2020.acl-main.702".
        fields: Comma-separated fields to retrieve. Defaults to a comprehensive set.

    Returns:
        JSON string with paper details.
    """
    if fields is None:
        fields = (
            "title,authors,year,abstract,citationCount,"
            "references,externalIds,tldr,publicationDate,"
            "journal,publicationTypes"
        )

    try:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/{paper_id}",
            params={"fields": fields},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return json.dumps({"error": str(e), "paper_id": paper_id})

    authors = [a.get("name", "") for a in data.get("authors", [])]
    references = [
        {
            "paper_id": r.get("paperId", ""),
            "title": r.get("title", ""),
        }
        for r in data.get("references", [])[:20]  # cap to 20 refs
    ]

    tldr = data.get("tldr")
    tldr_text = tldr.get("text", "") if tldr else ""

    result = {
        "paper_id": data.get("paperId", ""),
        "title": data.get("title", ""),
        "authors": authors,
        "year": data.get("year"),
        "publication_date": data.get("publicationDate"),
        "journal": data.get("journal"),
        "abstract": data.get("abstract", ""),
        "tldr": tldr_text,
        "citation_count": data.get("citationCount", 0),
        "external_ids": data.get("externalIds", {}),
        "references": references,
    }
    return json.dumps(result, ensure_ascii=False)
