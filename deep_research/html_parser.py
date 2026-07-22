"""Small bounded HTML parsers used by the keyless built-in web tools."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any


_SPACE = re.compile(r"\s+")
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def _classes(attributes: list[tuple[str, str | None]]) -> set[str]:
    value = next((value for key, value in attributes if key == "class"), "") or ""
    return set(value.split())


def _attribute(attributes: list[tuple[str, str | None]], name: str) -> str:
    return next((value or "" for key, value in attributes if key == name), "")


def _clean(parts: list[str]) -> str:
    return _SPACE.sub(" ", " ".join(parts)).strip()


class SearchResultsParser(HTMLParser):
    """Extract DuckDuckGo HTML result cards without constructing a full DOM."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._result_depth = 0
        self._current: dict[str, Any] | None = None
        self._capture = ""
        self._capture_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        classes = _classes(attributes)
        if self._current is None and "result" in classes:
            self._current = {"title": [], "url": "", "snippet": []}
            self._result_depth = 1
        elif self._current is not None and tag not in _VOID_TAGS:
            self._result_depth += 1

        if self._current is None:
            return
        if "result__a" in classes and tag == "a":
            self._current["url"] = _attribute(attributes, "href")
            self._capture = "title"
            self._capture_depth = 1
        elif "result__snippet" in classes:
            self._capture = "snippet"
            self._capture_depth = 1
        elif self._capture and tag not in _VOID_TAGS:
            self._capture_depth += 1

    def handle_startendtag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attributes)

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture and data.strip():
            self._current[self._capture].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if self._capture:
            self._capture_depth -= 1
            if self._capture_depth <= 0:
                self._capture = ""
                self._capture_depth = 0
        if tag not in _VOID_TAGS:
            self._result_depth -= 1
        if self._result_depth <= 0:
            url = str(self._current["url"])
            title = _clean(self._current["title"])
            if url and title:
                self.results.append({
                    "title": title,
                    "url": url,
                    "snippet": _clean(self._current["snippet"]),
                })
            self._current = None
            self._result_depth = 0
            self._capture = ""
            self._capture_depth = 0


class ReadableHTMLParser(HTMLParser):
    """Extract title and readable block text while ignoring common boilerplate."""

    _IGNORED = {"script", "style", "nav", "footer", "header", "aside", "form"}
    _CONTENT = {"p", "h1", "h2", "h3", "h4", "li"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._title_depth = 0
        self._title_parts: list[str] = []
        self._ignored_depth = 0
        self._body_depth = 0
        self._main_depth = 0
        self._article_depth = 0
        self._capture_depth = 0
        self._capture_parts: list[str] = []
        self._capture_scopes: tuple[bool, bool, bool] = (False, False, False)
        self._all: list[str] = []
        self._body: list[str] = []
        self._main: list[str] = []
        self._article: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        del attributes
        if tag in self._IGNORED:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._title_depth += 1
        if tag == "body":
            self._body_depth += 1
        elif tag == "main":
            self._main_depth += 1
        elif tag == "article":
            self._article_depth += 1

        if self._capture_depth and tag not in _VOID_TAGS:
            self._capture_depth += 1
        elif tag in self._CONTENT:
            self._capture_depth = 1
            self._capture_parts = []
            self._capture_scopes = (
                self._body_depth > 0,
                self._main_depth > 0,
                self._article_depth > 0,
            )

    def handle_startendtag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attributes)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._title_depth and data.strip():
            self._title_parts.append(data)
        if self._capture_depth and data.strip():
            self._capture_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return

        if self._capture_depth:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                text = _clean(self._capture_parts)
                if len(text) > 30:
                    self._all.append(text)
                    in_body, in_main, in_article = self._capture_scopes
                    if in_body:
                        self._body.append(text)
                    if in_main:
                        self._main.append(text)
                    if in_article:
                        self._article.append(text)
                self._capture_parts = []

        if tag == "title":
            self._title_depth = max(0, self._title_depth - 1)
            self.title = _clean(self._title_parts)
        elif tag == "article":
            self._article_depth = max(0, self._article_depth - 1)
        elif tag == "main":
            self._main_depth = max(0, self._main_depth - 1)
        elif tag == "body":
            self._body_depth = max(0, self._body_depth - 1)

    def text(self) -> str:
        paragraphs = self._article or self._main or self._body or self._all
        return "\n\n".join(paragraphs)
