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
    "authors",
    "abstract",
    "paperId",
    "url",
    "externalIds",
    "openAccessPdf",
    "venue",
    "year",
    "citationCount",
    "influentialCitationCount",
]


def _paper_data(result: object) -> dict:
    """Return a Semantic Scholar paper as a plain dictionary."""
    if hasattr(result, "raw_data"):
        return getattr(result, "raw_data")
    try:
        return dict(result)
    except Exception:
        return {}


def _external_ids(data: dict) -> dict:
    """Return a normalized externalIds dictionary."""
    raw_external_ids = data.get("externalIds", {}) or {}
    if isinstance(raw_external_ids, dict):
        return raw_external_ids
    return {}


def _extract_arxiv_id(data: dict) -> str:
    """Extract a stable paper identifier from Semantic Scholar data."""
    external_ids = _external_ids(data)
    return data.get("arxivId", "") or external_ids.get("ArXiv", "") or data.get("paperId", "")


def _extract_doi(data: dict) -> str:
    """Extract DOI from top-level or external identifiers when available."""
    top_level_doi = data.get("doi", "")
    if isinstance(top_level_doi, str) and top_level_doi.strip():
        return top_level_doi.strip()

    external_ids = _external_ids(data)
    direct_doi = external_ids.get("DOI") or external_ids.get("doi")
    if isinstance(direct_doi, str) and direct_doi.strip():
        return direct_doi.strip()

    for key, value in external_ids.items():
        if isinstance(key, str) and key.lower() == "doi" and isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested_doi = value.get("DOI") or value.get("doi")
            if isinstance(nested_doi, str) and nested_doi.strip():
                return nested_doi.strip()

    return ""


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
        authors_data = data.get("authors", []) or []
        authors = [
            author.get("name", "")
            for author in authors_data
            if isinstance(author, dict) and author.get("name")
        ]
        open_access_pdf = data.get("openAccessPdf", {}) or {}
        if not isinstance(open_access_pdf, dict):
            open_access_pdf = {}
        papers.append(
            Paper(
                title=data.get("title", "") or "",
                authors=authors,
                abstract=data.get("abstract", "") or "",
                arxiv_id=_extract_arxiv_id(data),
                pdf_url=open_access_pdf.get("url", "") or data.get("pdfUrl", "") or "",
                published=str(data.get("year", "") or ""),
                categories=[],
                source="semanticscholar",
                citation_count=data.get("citationCount", 0),
                influential_citations=data.get("influentialCitationCount", 0),
                venue=data.get("venue", "") or "",
                doi=_extract_doi(data),
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
