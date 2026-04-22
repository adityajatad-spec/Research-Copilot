"""Structured paper summarization using OpenAI-compatible chat completions."""

from __future__ import annotations

import json

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, track

try:
    from .config import Config, get_client, validate_provider_setup
    from .models import Paper, PaperSummary
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config, get_client, validate_provider_setup
    from models import Paper, PaperSummary


SYSTEM_PROMPT = (
    "You are a research analyst. Given an abstract, extract exactly four fields as a JSON object. "
    "Return ONLY valid JSON. No explanation, no markdown, no code block. "
    "Fields: core_contribution, methodology, key_result, limitation. "
    "Each value must be 1-2 sentences only. Be factual and specific."
)
USER_PROMPT_TEMPLATE = (
    "Abstract:\n{abstract}\n\n"
    "Return JSON with these exact keys:\n"
    "core_contribution, methodology, key_result, limitation"
)
FAILED_EXTRACTION = "[extraction failed]"
TITLE_LIMIT = 50


def _truncate_title(title: str, limit: int = TITLE_LIMIT) -> str:
    """Trim a paper title for progress output."""
    if len(title) <= limit:
        return title
    return f"{title[: limit - 3].rstrip()}..."


def _failed_summary(paper: Paper) -> PaperSummary:
    """Build a fallback summary when extraction fails."""
    return PaperSummary(
        arxiv_id=paper.arxiv_id,
        core_contribution=FAILED_EXTRACTION,
        methodology=FAILED_EXTRACTION,
        key_result=FAILED_EXTRACTION,
        limitation=FAILED_EXTRACTION,
        raw_abstract=paper.abstract,
    )


def summarize_paper(paper: Paper, config: Config) -> PaperSummary:
    """Generate a structured summary for a single paper."""
    validate_provider_setup(config)
    client = get_client(config)
    user_prompt = USER_PROMPT_TEMPLATE.format(abstract=paper.abstract)

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        return PaperSummary(
            arxiv_id=paper.arxiv_id,
            core_contribution=parsed["core_contribution"],
            methodology=parsed["methodology"],
            key_result=parsed["key_result"],
            limitation=parsed["limitation"],
            raw_abstract=paper.abstract,
        )
    except Exception:
        return _failed_summary(paper)


def summarize_all(papers: list[Paper], config: Config, verbose: bool = True) -> list[Paper]:
    """Generate structured summaries for all papers."""
    console = Console()
    total = len(papers)

    if total == 0:
        return papers

    if not verbose:
        for paper in papers:
            paper.summary = summarize_paper(paper, config)
        return papers

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task("Preparing summaries...", total=total)

        for index, paper in enumerate(papers, start=1):
            progress.update(
                task_id,
                description=f"Summarizing paper {index}/{total}: {_truncate_title(paper.title)}",
            )
            paper.summary = summarize_paper(paper, config)

            if paper.summary.core_contribution == FAILED_EXTRACTION:
                console.print(f"[yellow]Failed summary for {paper.arxiv_id}[/yellow]")
            else:
                console.print(f"[green]✓ {paper.arxiv_id}[/green]")

            progress.advance(task_id)

    return papers
