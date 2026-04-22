"""Core arXiv paper fetching logic."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

import arxiv

try:
    from .models import Paper
    from .utils import normalize_title_for_dedup
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import Paper
    from utils import normalize_title_for_dedup


DEFAULT_MAX_RESULTS = 10
ERROR_EMPTY_TOPIC = "Error: topic cannot be empty."
ERROR_INVALID_MAX = "Error: max_results must be greater than 0."
ERROR_FETCH_FAILED = "Error fetching papers from arXiv: {error}"
ENTRY_ID_PATTERN = re.compile(r"/abs/([^?#]+)")
VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$")


def _normalize_text(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""
    return " ".join(text.split())


def _extract_arxiv_id(entry_id: str) -> str:
    """Convert an arXiv entry URL into a bare arXiv identifier."""
    match = ENTRY_ID_PATTERN.search(entry_id)
    raw_id = match.group(1) if match else entry_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return VERSION_SUFFIX_PATTERN.sub("", raw_id)


def _paper_year(paper: Paper) -> str:
    """Extract a comparable year string from a paper."""
    if not paper.published:
        return ""
    return paper.published.split("-", maxsplit=1)[0]


def _paper_score(paper: Paper) -> int:
    """Return the ranking score used during deduplication."""
    return paper.citation_count or 0


def _prefer_arxiv_on_tie(candidate: Paper, current: Paper) -> bool:
    """Return whether the candidate should win a tie-break in favor of arXiv."""
    candidate_is_arxiv = candidate.source == "arxiv"
    current_is_arxiv = current.source == "arxiv"
    return candidate_is_arxiv and not current_is_arxiv


def _merge_paper_fields(preferred: Paper, other: Paper) -> Paper:
    """Merge complementary metadata from another paper into the preferred paper."""
    if not preferred.abstract and other.abstract:
        preferred.abstract = other.abstract
    if not preferred.pdf_url and other.pdf_url:
        preferred.pdf_url = other.pdf_url
    if not preferred.published and other.published:
        preferred.published = other.published
    if not preferred.authors and other.authors:
        preferred.authors = other.authors
    if not preferred.local_pdf_path and other.local_pdf_path:
        preferred.local_pdf_path = other.local_pdf_path
    if not preferred.full_text and other.full_text:
        preferred.full_text = other.full_text
    if preferred.summary is None and other.summary is not None:
        preferred.summary = other.summary
    if not preferred.venue and other.venue:
        preferred.venue = other.venue
    if not preferred.doi and other.doi:
        preferred.doi = other.doi
    if preferred.influential_citations is None and other.influential_citations is not None:
        preferred.influential_citations = other.influential_citations
    if preferred.citation_count is None and other.citation_count is not None:
        preferred.citation_count = other.citation_count
    preferred.categories = list(dict.fromkeys(preferred.categories + other.categories))
    return preferred


def deduplicate_papers(papers: list[Paper], threshold: float = 0.8) -> list[Paper]:
    """Deduplicate papers by arXiv id or title similarity while preserving better metadata."""
    unique_papers: list[Paper] = []

    for paper in papers:
        matched_index: int | None = None
        paper_title = normalize_title_for_dedup(paper.title)
        paper_year = _paper_year(paper)

        for index, existing in enumerate(unique_papers):
            if paper.arxiv_id and existing.arxiv_id and paper.arxiv_id == existing.arxiv_id:
                matched_index = index
                break

            existing_title = normalize_title_for_dedup(existing.title)
            similarity = SequenceMatcher(None, paper_title, existing_title).ratio()
            existing_year = _paper_year(existing)
            years_compatible = not paper_year or not existing_year or paper_year == existing_year

            if similarity >= threshold and years_compatible:
                matched_index = index
                break

        if matched_index is None:
            unique_papers.append(paper)
            continue

        current = unique_papers[matched_index]
        candidate_score = _paper_score(paper)
        current_score = _paper_score(current)

        if candidate_score > current_score or (
            candidate_score == current_score and _prefer_arxiv_on_tie(paper, current)
        ):
            unique_papers[matched_index] = _merge_paper_fields(paper, current)
        else:
            unique_papers[matched_index] = _merge_paper_fields(current, paper)

    return unique_papers


def fetch_papers(topic: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[Paper]:
    """Fetch papers from arXiv for a topic."""
    query = topic.strip()
    if not query:
        print(ERROR_EMPTY_TOPIC)
        return []

    if max_results <= 0:
        print(ERROR_INVALID_MAX)
        return []

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client(page_size=min(max_results, 100), num_retries=2)

    try:
        results = list(client.results(search))
    except Exception as error:  # pragma: no cover - depends on network/API behavior
        print(ERROR_FETCH_FAILED.format(error=error))
        return []

    papers: list[Paper] = []
    for result in results:
        published_date = result.published.date().isoformat() if result.published else ""
        papers.append(
            Paper(
                title=_normalize_text(result.title),
                authors=[author.name for author in result.authors],
                abstract=_normalize_text(result.summary),
                arxiv_id=_extract_arxiv_id(result.entry_id),
                pdf_url=result.pdf_url or "",
                published=published_date,
                categories=list(result.categories or []),
                source="arxiv",
            )
        )

    return papers
