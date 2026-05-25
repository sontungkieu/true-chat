from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str = ""


class WebSearchClient(Protocol):
    def search(self, query: str, *, limit: int) -> list[WebSearchResult]: ...


@dataclass
class DuckDuckGoLiteSearchClient:
    endpoint: str = "https://duckduckgo.com/html/"
    timeout_s: float = 8.0
    user_agent: str = "true-chat-rag-bench/0.1 (+https://duckduckgo.com)"

    def search(self, query: str, *, limit: int) -> list[WebSearchResult]:
        cleaned = " ".join(str(query or "").split())
        if not cleaned or limit <= 0:
            return []
        request = Request(
            f"{self.endpoint}?q={quote_plus(cleaned)}",
            headers={"User-Agent": self.user_agent},
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - user-triggered public web search.
            html = response.read().decode("utf-8", errors="replace")
        return parse_duckduckgo_results(html, limit=limit)


def parse_duckduckgo_results(html: str, *, limit: int) -> list[WebSearchResult]:
    parser = _DuckDuckGoResultParser(limit=max(0, limit))
    parser.feed(html)
    return parser.results[:limit]


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self, *, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.results: list[WebSearchResult] = []
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._snippet_depth = 0
        self._snippet_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.results) >= self.limit:
            return
        attributes = {key: value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            self._link_href = attributes.get("href", "")
            self._link_text = []
            return
        if "result__snippet" in classes:
            self._snippet_depth = 1
            self._snippet_text = []
            return
        if self._snippet_depth:
            self._snippet_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link_href is not None:
            title = _clean_text(" ".join(self._link_text))
            url = clean_duckduckgo_url(self._link_href)
            if title and url:
                self.results.append(WebSearchResult(title=title, url=url))
            self._link_href = None
            self._link_text = []
            return
        if self._snippet_depth:
            self._snippet_depth -= 1
            if self._snippet_depth == 0 and self.results:
                snippet = _clean_text(" ".join(self._snippet_text))
                if snippet:
                    previous = self.results[-1]
                    self.results[-1] = WebSearchResult(
                        title=previous.title,
                        url=previous.url,
                        snippet=snippet,
                    )
                self._snippet_text = []

    def handle_data(self, data: str) -> None:
        if self._link_href is not None:
            self._link_text.append(data)
        if self._snippet_depth:
            self._snippet_text.append(data)


def clean_duckduckgo_url(value: str) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())
