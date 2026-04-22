"""Utility helpers for saving and displaying arXiv papers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from .models import Paper, PaperSummary
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import Paper, PaperSummary


DEFAULT_HEADER_TITLE = "arXiv Search Results"
TITLE_LIMIT = 60
UNKNOWN_AUTHOR = "Unknown"
UNKNOWN_YEAR = "-"
TITLE_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9\s]")


def _truncate(text: str, limit: int = TITLE_LIMIT) -> str:
    """Trim long text for compact terminal output."""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _format_authors(authors: list[str]) -> str:
    """Format authors for the summary table."""
    if not authors:
        return UNKNOWN_AUTHOR
    if len(authors) == 1:
        return authors[0]
    return f"{authors[0]} et al."


def _extract_year(published: str) -> str:
    """Extract the publication year from an ISO date string."""
    if not published:
        return UNKNOWN_YEAR
    return published.split("-", maxsplit=1)[0]


def normalize_title_for_dedup(title: str) -> str:
    """Normalize a title for simple cross-source deduplication."""
    normalized_title = title.strip().lower()
    normalized_title = TITLE_NORMALIZE_PATTERN.sub(" ", normalized_title)
    return " ".join(normalized_title.split())


def save_to_json(papers: list[Paper], filepath: str) -> None:
    """Save papers to a JSON file."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paper_dicts = [paper.to_dict() for paper in papers]
    output_path.write_text(json.dumps(paper_dicts, indent=2), encoding="utf-8")


def save_json_data(data: dict, filepath: str) -> None:
    """Save a JSON-serializable dictionary to disk."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_from_json(filepath: str) -> list[Paper]:
    """Load papers from a JSON file."""
    input_path = Path(filepath)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    raw_items = json.loads(input_path.read_text(encoding="utf-8"))
    papers: list[Paper] = []

    for item in raw_items:
        summary_data = item.get("summary")
        if summary_data is not None:
            item["summary"] = PaperSummary(**summary_data)
        papers.append(Paper(**item))

    return papers


def save_report(report: str, filepath: str) -> None:
    """Save a markdown report string to a .md file, creating dirs if needed."""
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as output_file:
        output_file.write(report)


def display_papers(papers: list[Paper], query: str = "") -> None:
    """Display papers in a Rich table."""
    console = Console()
    header_lines = [f"Results: {len(papers)}"]
    if query.strip():
        header_lines.insert(0, f"Query: {query.strip()}")

    console.print(Panel.fit("\n".join(header_lines), title=DEFAULT_HEADER_TITLE))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", justify="right", width=4)
    table.add_column("Title", overflow="fold")
    table.add_column("Authors")
    table.add_column("Year", justify="center", width=8)
    table.add_column("arXiv ID", style="green")

    for index, paper in enumerate(papers, start=1):
        table.add_row(
            str(index),
            _truncate(paper.title),
            _format_authors(paper.authors),
            _extract_year(paper.published),
            paper.arxiv_id,
        )

    console.print(table)
