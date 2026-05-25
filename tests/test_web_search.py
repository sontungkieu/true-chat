from __future__ import annotations

from rag_bench.web_search import clean_duckduckgo_url, parse_duckduckgo_results


def test_parse_duckduckgo_results_extracts_title_url_and_snippet() -> None:
    html = """
    <html>
      <body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Fpage">Example result</a>
        <a class="result__snippet">A concise result snippet.</a>
      </body>
    </html>
    """

    results = parse_duckduckgo_results(html, limit=3)

    assert len(results) == 1
    assert results[0].title == "Example result"
    assert results[0].url == "https://example.test/page"
    assert results[0].snippet == "A concise result snippet."


def test_clean_duckduckgo_url_keeps_direct_urls() -> None:
    assert clean_duckduckgo_url("https://example.test/direct") == "https://example.test/direct"
