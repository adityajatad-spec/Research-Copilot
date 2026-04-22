"""Semantic Scholar paper fetching helpers."""

from __future__ import annotations

from semanticscholar import SemanticScholar

try:
    from .fetcher import DEFAULT_MAX_RESULTS, ERROR_EMPTY_TOPIC, ERROR_INVALID_MAX, deduplicate_papers, fetch_papers
    from .models import Paper
except ImportError:  # pragma: no cover - fallback for direct script execution
    from fetcher import DEFAULT_MAX_RESULTS, ERROR_EMPTY_TOPIC, ERROR_INVALID_MAX, deduplicate_papers, fetch_papers
    from models import Paper


ERROR_SCHOLAR_FETCH_FAILED = "Error fetching papers from Semantic Scholar: {error}"
SCHOLAR_FIELDS = [
    "title",
    "abstract",
    "authors",
    "paperId",
    "externalIds",
    "openAccessPdf",
    "venue",
    "year",
    "citationCount",
    "influentialCitationCount",
    "topics",
    "doi",
]


def _paper_data(result: object) -> dict:
    """Return a Semantic Scholar paper as a plain dictionary."""
    if hasattr(result, "raw_data"):
        return getattr(result, "raw_data")
    return dict(result)


def _extract_arxiv_id(data: dict) -> str:
    """Extract a stable paper identifier from Semantic Scholar data."""
    external_ids = data.get("externalIds", {}) or {}
    return data.get("arxivId", "") or external_ids.get("ArXiv", "") or data.get("paperId", "")


def _extract_categories(data: dict) -> list[str]:
    """Extract category-like topic names from Semantic Scholar data."""
    topics = data.get("topics", []) or []
    categories = [item.get("topic", "") for item in topics if item.get("topic")]
    return categories


def fetch_semantic_scholar_papers(topic: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Paper]:
    """Fetch papers from Semantic Scholar for a topic."""
    query = topic.strip()
    if not query:
        print(ERROR_EMPTY_TOPIC)
        return []

    if max_results <= 0:
        print(ERROR_INVALID_MAX)
        return []

    try:
        client = SemanticScholar(timeout=10)
        results = client.search_paper(query, limit=min(max_results, 100), fields=SCHOLAR_FIELDS)
        result_items = list(getattr(results, "items", results))
    except Exception as error:  # pragma: no cover - depends on network/API behavior
        print(ERROR_SCHOLAR_FETCH_FAILED.format(error=error))
        return []

    papers: list[Paper] = []
    for result in result_items:
        data = _paper_data(result)
        authors = [author.get("name", "") for author in data.get("authors", []) if author.get("name")]
        open_access_pdf = data.get("openAccessPdf", {}) or {}
        papers.append(
            Paper(
                title=data.get("title", "") or "",
                authors=authors,
                abstract=data.get("abstract", "") or "",
                arxiv_id=_extract_arxiv_id(data),
                pdf_url=open_access_pdf.get("url", "") or data.get("pdfUrl", "") or "",
                published=str(data.get("year", "") or ""),
                categories=_extract_categories(data),
                source="semanticscholar",
                citation_count=data.get("citationCount", 0),
                influential_citations=data.get("influentialCitationCount", 0),
                venue=data.get("venue", "") or "",
                doi=data.get("doi", "") or "",
            )
        )

    papers.sort(key=lambda paper: paper.citation_count or 0, reverse=True)
    return papers[:max_results]


def fetch_hybrid_papers(topic: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Paper]:
    """Fetch papers from arXiv and Semantic Scholar, then merge and deduplicate them."""
    query = topic.strip()
    if not query:
        print(ERROR_EMPTY_TOPIC)
        return []

    if max_results <= 0:
        print(ERROR_INVALID_MAX)
        return []

    arxiv_count = max_results // 2
    scholar_count = max_results - arxiv_count

    if arxiv_count == 0:
        arxiv_count = 1
    if scholar_count == 0:
        scholar_count = 1

    arxiv_papers = fetch_papers(query, arxiv_count)
    scholar_papers = fetch_semantic_scholar_papers(query, scholar_count)
    combined = arxiv_papers + scholar_papers
    unique_papers = deduplicate_papers(combined)
    unique_papers.sort(
        key=lambda paper: (paper.citation_count or 0, 1 if paper.source == "arxiv" else 0),
        reverse=True,
    )
    return unique_papers[:max_results]
