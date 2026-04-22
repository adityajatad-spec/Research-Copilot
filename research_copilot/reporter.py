"""Markdown report generation for summarized research papers."""

from __future__ import annotations

from datetime import date
import re

try:
    from .models import Paper
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import Paper


SUMMARY_NOT_AVAILABLE = "[Summary not available]"
UNKNOWN_AUTHOR = "Unknown"
UNKNOWN_DATE = "Unknown"
UNKNOWN_CATEGORY = "Uncategorized"
UNKNOWN_YEAR = "N/A"
UNKNOWN_THEMES = "none available"
TITLE_LIMIT = 50
CONTRIBUTION_LIMIT = 80
WORD_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9-]*\b")


def _unique_in_order(values: list[str]) -> list[str]:
    """Return unique values while preserving their original order."""
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        normalized_value = value.strip()
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        unique_values.append(normalized_value)

    return unique_values


def _truncate(text: str, limit: int) -> str:
    """Trim text for compact markdown tables."""
    cleaned_text = " ".join(text.split())
    if len(cleaned_text) <= limit:
        return cleaned_text
    return f"{cleaned_text[: limit - 3].rstrip()}..."


def _escape_table_text(text: str) -> str:
    """Escape markdown table content safely."""
    return " ".join(text.split()).replace("|", "\\|")


def _format_abstract(abstract: str) -> str:
    """Format an abstract as a markdown blockquote."""
    cleaned_abstract = abstract.strip() or "Abstract not available."
    return "\n".join(f"> {line}" for line in cleaned_abstract.splitlines())


def _get_summary_field(paper: Paper, field_name: str) -> str:
    """Return a summary field or a placeholder when unavailable."""
    if paper.summary is None:
        return SUMMARY_NOT_AVAILABLE
    return getattr(paper.summary, field_name)


def _collect_categories(papers: list[Paper]) -> list[str]:
    """Collect unique categories across all papers."""
    categories: list[str] = []
    for paper in papers:
        categories.extend(paper.categories)
    unique_categories = _unique_in_order(categories)
    return unique_categories or [UNKNOWN_CATEGORY]


def _collect_theme_words(papers: list[Paper]) -> list[str]:
    """Collect the first five unique words from core contributions."""
    theme_words: list[str] = []
    seen: set[str] = set()

    for paper in papers:
        core_contribution = _get_summary_field(paper, "core_contribution")
        for word in WORD_PATTERN.findall(core_contribution.lower()):
            if word in seen:
                continue
            seen.add(word)
            theme_words.append(word)
            if len(theme_words) == 5:
                return theme_words

    return theme_words


def _collect_years(papers: list[Paper]) -> list[str]:
    """Collect valid published years from the paper list."""
    years: list[str] = []
    for paper in papers:
        if len(paper.published) >= 4:
            year = paper.published[:4]
            if year.isdigit():
                years.append(year)
    return years


def _build_overview(papers: list[Paper], topic: str) -> str:
    """Build the overview paragraph for the report."""
    categories = ", ".join(_collect_categories(papers))
    theme_words = _collect_theme_words(papers)
    themes = ", ".join(theme_words) if theme_words else UNKNOWN_THEMES
    years = _collect_years(papers)
    earliest_year = min(years) if years else UNKNOWN_YEAR
    latest_year = max(years) if years else UNKNOWN_YEAR

    return (
        f"This report covers {len(papers)} papers on the topic of {topic}. "
        f"The research spans areas including {categories}. "
        f"Key themes include: {themes}. "
        f"Papers range from {earliest_year} to {latest_year}."
    )


def _build_paper_section(index: int, paper: Paper) -> str:
    """Build the markdown section for a single paper."""
    authors = ", ".join(paper.authors) if paper.authors else UNKNOWN_AUTHOR
    categories = ", ".join(paper.categories) if paper.categories else UNKNOWN_CATEGORY
    published = paper.published or UNKNOWN_DATE
    core_contribution = _get_summary_field(paper, "core_contribution")
    methodology = _get_summary_field(paper, "methodology")
    key_result = _get_summary_field(paper, "key_result")
    limitation = _get_summary_field(paper, "limitation")

    lines = [
        f"### {index}. {paper.title}",
        "",
        f"**Authors:** {authors}  ",
        f"**Published:** {published}  ",
        f"**arXiv ID:** [{paper.arxiv_id}](https://arxiv.org/abs/{paper.arxiv_id})  ",
        f"**Categories:** {categories}  ",
        "",
        "#### Summary",
        "",
        "| Field | Content |",
        "|-------|---------|",
        f"| Core Contribution | {_escape_table_text(core_contribution)} |",
        f"| Methodology | {_escape_table_text(methodology)} |",
        f"| Key Result | {_escape_table_text(key_result)} |",
        f"| Limitation | {_escape_table_text(limitation)} |",
        "",
        "#### Abstract",
        "",
        _format_abstract(paper.abstract),
        "",
        "---",
    ]
    return "\n".join(lines)


def _build_quick_reference_table(papers: list[Paper]) -> str:
    """Build the compact quick reference markdown table."""
    lines = [
        "| # | Title | First Author | Year | arXiv ID | Core Contribution (short) |",
        "|---|-------|-------------|------|----------|--------------------------|",
    ]

    for index, paper in enumerate(papers, start=1):
        first_author = paper.authors[0] if paper.authors else UNKNOWN_AUTHOR
        year = paper.published[:4] if len(paper.published) >= 4 and paper.published[:4].isdigit() else UNKNOWN_YEAR
        core_contribution = _get_summary_field(paper, "core_contribution")
        lines.append(
            "| "
            f"{index} | "
            f"{_escape_table_text(_truncate(paper.title, TITLE_LIMIT))} | "
            f"{_escape_table_text(first_author)} | "
            f"{year} | "
            f"{paper.arxiv_id} | "
            f"{_escape_table_text(_truncate(core_contribution, CONTRIBUTION_LIMIT))} |"
        )

    return "\n".join(lines)


def generate_report(papers: list[Paper], topic: str) -> str:
    """Generate a structured Markdown research report from summarized papers."""
    report_topic = topic.strip() or UNKNOWN_CATEGORY
    report_date = date.today().isoformat()
    overview = _build_overview(papers, report_topic)
    paper_sections = "\n\n".join(_build_paper_section(index, paper) for index, paper in enumerate(papers, start=1))
    quick_reference_table = _build_quick_reference_table(papers)

    lines = [
        "---",
        "",
        f"# Research Report: {report_topic}",
        "",
        "> Generated by AI Research Copilot  ",
        f"> Papers analyzed: {len(papers)}  ",
        f"> Date: {report_date}  ",
        "",
        "---",
        "",
        "## Overview",
        "",
        overview,
        "",
        "---",
        "",
        "## Papers",
        "",
    ]

    if paper_sections:
        lines.append(paper_sections)
    else:
        lines.extend(
            [
                "No papers were available for this report.",
                "",
                "---",
            ]
        )

    lines.extend(
        [
            "",
            "## Quick Reference Table",
            "",
            quick_reference_table,
            "",
            "---",
            "",
            "## Notes",
            "",
            "- This report is generated from stored paper metadata and summaries.",
            '- Summary fields show "[Summary not available]" when no summary data exists in the input.',
            '- Fields that contain "[extraction failed]" reflect an upstream summarization failure.',
        ]
    )

    return "\n".join(lines)
