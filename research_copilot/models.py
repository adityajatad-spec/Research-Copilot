"""Data models for the AI Research Copilot CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PaperSummary:
    """Represent a structured summary generated from a paper abstract."""

    arxiv_id: str
    core_contribution: str
    methodology: str
    key_result: str
    limitation: str
    raw_abstract: str

    def to_dict(self) -> dict:
        """Return the paper summary as a plain dictionary."""
        return {
            "arxiv_id": self.arxiv_id,
            "core_contribution": self.core_contribution,
            "methodology": self.methodology,
            "key_result": self.key_result,
            "limitation": self.limitation,
            "raw_abstract": self.raw_abstract,
        }


@dataclass(slots=True)
class Paper:
    """Represent a paper returned from arXiv."""

    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str
    pdf_url: str
    published: str
    categories: list[str]
    summary: PaperSummary | None = None
    local_pdf_path: str | None = None
    full_text: str | None = None
    source: str | None = None
    citation_count: int | None = None
    influential_citations: int | None = None
    venue: str | None = None
    doi: str | None = None

    def to_dict(self) -> dict:
        """Return the paper as a plain dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "arxiv_id": self.arxiv_id,
            "pdf_url": self.pdf_url,
            "published": self.published,
            "categories": self.categories,
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "local_pdf_path": self.local_pdf_path,
            "full_text": self.full_text,
            "source": self.source,
            "citation_count": self.citation_count,
            "influential_citations": self.influential_citations,
            "venue": self.venue,
            "doi": self.doi,
        }


@dataclass(slots=True)
class InsightReport:
    """Represent cross-paper insights extracted from a set of papers."""

    topic: str
    paper_count: int
    major_themes: list[str]
    common_methodologies: list[str]
    emerging_subtopics: list[str]
    notable_observations: list[str]

    def to_dict(self) -> dict:
        """Return the insight report as a plain dictionary."""
        return {
            "topic": self.topic,
            "paper_count": self.paper_count,
            "major_themes": self.major_themes,
            "common_methodologies": self.common_methodologies,
            "emerging_subtopics": self.emerging_subtopics,
            "notable_observations": self.notable_observations,
        }


@dataclass(slots=True)
class ContradictionItem:
    """Represent a contradiction identified between two papers."""

    paper_a: str
    paper_b: str
    contradiction: str

    def to_dict(self) -> dict:
        """Return the contradiction item as a plain dictionary."""
        return {
            "paper_a": self.paper_a,
            "paper_b": self.paper_b,
            "contradiction": self.contradiction,
        }


@dataclass(slots=True)
class GapReport:
    """Represent cross-paper gaps and contradictions for one topic."""

    topic: str
    paper_count: int
    contradictions: list[ContradictionItem]
    recurring_limitations: list[str]
    underexplored_directions: list[str]
    explicit_research_gaps: list[str]

    def to_dict(self) -> dict:
        """Return the gap report as a plain dictionary."""
        return {
            "topic": self.topic,
            "paper_count": self.paper_count,
            "contradictions": [item.to_dict() for item in self.contradictions],
            "recurring_limitations": self.recurring_limitations,
            "underexplored_directions": self.underexplored_directions,
            "explicit_research_gaps": self.explicit_research_gaps,
        }
