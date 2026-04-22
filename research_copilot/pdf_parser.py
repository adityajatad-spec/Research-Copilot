"""PDF download and parsing helpers for full-paper enrichment."""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import requests
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

try:
    from .models import Paper
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import Paper


DEFAULT_PDF_DIR = "output/pdfs"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_PAGES = 10
FILENAME_FALLBACK = "paper"
CHUNK_SIZE = 8192
PROGRESS_TITLE_LIMIT = 40


def sanitize_filename(value: str) -> str:
    """Convert a title or arXiv id into a safe deterministic filename."""
    normalized_value = value.strip().lower()
    normalized_value = re.sub(r"\s+", "-", normalized_value)
    normalized_value = re.sub(r"[^a-z0-9._-]", "", normalized_value)
    normalized_value = re.sub(r"-{2,}", "-", normalized_value).strip("-._")
    return normalized_value or FILENAME_FALLBACK


def _short_label(paper: Paper, limit: int = PROGRESS_TITLE_LIMIT) -> str:
    """Build a compact label for progress output."""
    label = paper.arxiv_id or paper.title or FILENAME_FALLBACK
    if len(label) <= limit:
        return label
    return f"{label[: limit - 3].rstrip()}..."


def download_pdf(pdf_url: str, output_path: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Download a PDF to disk and return the saved file path."""
    if not pdf_url or not pdf_url.strip():
        raise ValueError("PDF URL is empty.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(pdf_url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise ValueError(f"Failed to download PDF: received status code {response.status_code}.")

            with output_file.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file_handle.write(chunk)
    except requests.RequestException as error:
        raise ValueError(f"Failed to download PDF: {error}") from error
    except OSError as error:
        raise ValueError(f"Failed to save PDF to disk: {error}") from error

    return str(output_file)


def extract_text_from_pdf(pdf_path: str, max_pages: int | None = DEFAULT_MAX_PAGES) -> str:
    """Extract plain text from a PDF file using PyMuPDF."""
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be greater than 0 or None.")

    try:
        with fitz.open(pdf_path) as document:
            page_limit = document.page_count if max_pages is None else min(max_pages, document.page_count)
            extracted_pages: list[str] = []

            for page_index in range(page_limit):
                page = document.load_page(page_index)
                page_text = page.get_text("text").strip()
                normalized_page_text = re.sub(r"\n\s*\n+", "\n\n", page_text)
                normalized_page_text = re.sub(r"[ \t]+\n", "\n", normalized_page_text).strip()

                if not normalized_page_text:
                    continue

                extracted_pages.append(
                    f"--- PAGE {page_index + 1} ---\n\n{normalized_page_text}"
                )
    except Exception as error:
        raise ValueError(f"Failed to parse PDF: {error}") from error

    extracted_text = "\n\n".join(extracted_pages).strip()
    extracted_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", extracted_text)
    return extracted_text


def enrich_paper_with_pdf_text(
    paper: Paper,
    pdf_dir: str = DEFAULT_PDF_DIR,
    max_pages: int | None = DEFAULT_MAX_PAGES,
) -> Paper:
    """Download a paper PDF, extract text, and attach it to the paper."""
    if not paper.pdf_url or not paper.pdf_url.strip():
        return paper

    safe_name = sanitize_filename(paper.arxiv_id or paper.title)
    output_path = Path(pdf_dir) / f"{safe_name}.pdf"

    try:
        saved_path = download_pdf(paper.pdf_url, str(output_path))
        extracted_text = extract_text_from_pdf(saved_path, max_pages=max_pages)
        paper.local_pdf_path = saved_path
        paper.full_text = extracted_text
    except Exception:
        paper.local_pdf_path = None
        paper.full_text = None

    return paper


def enrich_papers_with_pdf_text(
    papers: list[Paper],
    pdf_dir: str = DEFAULT_PDF_DIR,
    max_pages: int | None = DEFAULT_MAX_PAGES,
    verbose: bool = True,
) -> list[Paper]:
    """Enrich a list of papers with downloaded PDFs and extracted text."""
    console = Console()
    total = len(papers)

    if total == 0:
        return papers

    if not verbose:
        for paper in papers:
            enrich_paper_with_pdf_text(paper, pdf_dir=pdf_dir, max_pages=max_pages)
        return papers

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task("Preparing PDF enrichment...", total=total)

        for index, paper in enumerate(papers, start=1):
            progress.update(
                task_id,
                description=f"Parsing PDF {index}/{total}: {_short_label(paper)}",
            )

            enrich_paper_with_pdf_text(paper, pdf_dir=pdf_dir, max_pages=max_pages)

            if paper.full_text:
                console.print(f"[green]✓ PDF parsed for {_short_label(paper)}[/green]")
            else:
                console.print(f"[yellow]Failed to parse PDF for {_short_label(paper)}[/yellow]")

            progress.advance(task_id)

    return papers


def build_pdf_stats(papers: list[Paper]) -> dict[str, int]:
    """Build summary statistics for PDF enrichment results."""
    total_papers = len(papers)
    pdf_downloaded = sum(1 for paper in papers if paper.local_pdf_path is not None)
    text_extracted = sum(1 for paper in papers if paper.full_text is not None and paper.full_text.strip())
    failures = total_papers - text_extracted

    return {
        "total_papers": total_papers,
        "pdf_downloaded": pdf_downloaded,
        "text_extracted": text_extracted,
        "failures": failures,
    }
