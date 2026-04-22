"""Cross-paper gap and contradiction detection helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from .config import Config, get_client, validate_provider_setup
    from .models import ContradictionItem, GapReport, Paper
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config, get_client, validate_provider_setup
    from models import ContradictionItem, GapReport, Paper


CONTEXT_CHAR_LIMIT = 20_000
TEXT_PREVIEW_LIMIT = 250
FAILED_EXTRACTION = "[gap extraction failed]"
SUMMARY_NOT_AVAILABLE = "[summary not available]"
WHITESPACE_PATTERN = re.compile(r"\s+")
SYSTEM_PROMPT = """
You are an expert research analyst.

You are given summaries and partial full-text context from multiple research papers on one topic.

Your task is to identify:
1. contradictions between papers
2. recurring limitations across papers
3. underexplored directions suggested by the collection
4. explicit research gaps that appear not to be addressed

Return ONLY valid JSON with this schema:
{
  "contradictions": [
    {
      "paper_a": "title of first paper",
      "paper_b": "title of second paper",
      "contradiction": "one concise sentence describing the disagreement"
    }
  ],
  "recurring_limitations": ["..."],
  "underexplored_directions": ["..."],
  "explicit_research_gaps": ["..."]
}

Rules:
- Only identify contradictions if there is reasonable evidence of disagreement
- Prefer conservative, evidence-based outputs over speculative ones
- If there are no strong contradictions, return an empty contradictions list
- Each list item should be one concise sentence
- Focus on cross-paper synthesis, not per-paper summaries
""".strip()
USER_PROMPT_TEMPLATE = """
Topic: {topic}
Papers analyzed: {paper_count}

{gap_context}

Return the JSON only.
""".strip()


def _clean_text(value: str, limit: int | None = None) -> str:
    """Normalize whitespace and optionally truncate text."""
    cleaned_value = WHITESPACE_PATTERN.sub(" ", value.strip())
    if limit is not None and len(cleaned_value) > limit:
        return f"{cleaned_value[:limit].rstrip()} [truncated]"
    return cleaned_value


def _paper_sort_key(paper: Paper) -> tuple[int, int, int, int, str]:
    """Build a priority key for selecting the most informative papers first."""
    has_summary = 1 if paper.summary is not None else 0
    citation_count = paper.citation_count or 0
    influential_count = paper.influential_citations or 0
    has_full_text = 1 if paper.full_text else 0
    return (has_summary, citation_count, influential_count, has_full_text, paper.title.lower())


def _summary_value(value: str | None) -> str:
    """Return a safe summary field value."""
    if not value or not value.strip():
        return SUMMARY_NOT_AVAILABLE
    return _clean_text(value)


def _paper_preview(paper: Paper) -> str:
    """Return a compact preview from full text when available."""
    if not paper.full_text:
        return ""
    return _clean_text(paper.full_text, limit=TEXT_PREVIEW_LIMIT)


def _paper_block(index: int, paper: Paper) -> str:
    """Build one paper block for the gap extraction context."""
    source = paper.source or "unknown"
    categories = ", ".join(paper.categories) if paper.categories else "none"
    citation_count = paper.citation_count if paper.citation_count is not None else "unknown"

    if paper.summary is not None:
        contribution = _summary_value(paper.summary.core_contribution)
        methodology = _summary_value(paper.summary.methodology)
        key_result = _summary_value(paper.summary.key_result)
        limitation = _summary_value(paper.summary.limitation)
    else:
        contribution = SUMMARY_NOT_AVAILABLE
        methodology = SUMMARY_NOT_AVAILABLE
        key_result = SUMMARY_NOT_AVAILABLE
        limitation = SUMMARY_NOT_AVAILABLE

    lines = [
        f'Paper {index}: "{paper.title}"',
        f"Source: {source}",
        f"Categories: {categories}",
        f"Citation Count: {citation_count}",
        f"Core Contribution: {contribution}",
        f"Methodology: {methodology}",
        f"Key Result: {key_result}",
        f"Limitation: {limitation}",
    ]

    preview = _paper_preview(paper)
    if preview:
        lines.append(f"Full Text Preview: {preview}")

    return "\n".join(lines)


def build_gap_context(papers: list[Paper]) -> str:
    """Build an LLM-friendly cross-paper context for gap extraction."""
    sorted_papers = sorted(papers, key=_paper_sort_key, reverse=True)
    blocks: list[str] = []
    current_length = 0

    for index, paper in enumerate(sorted_papers, start=1):
        block = _paper_block(index, paper)
        separator = "\n\n==========\n\n" if blocks else ""
        proposed_length = current_length + len(separator) + len(block)

        if proposed_length > CONTEXT_CHAR_LIMIT:
            remaining_chars = CONTEXT_CHAR_LIMIT - current_length - len(separator)
            if remaining_chars > 120:
                truncated_block = f"{block[: remaining_chars - 14].rstrip()}\n\n[truncated]"
                blocks.append(f"{separator}{truncated_block}" if separator else truncated_block)
            break

        blocks.append(f"{separator}{block}" if separator else block)
        current_length = proposed_length

    return "".join(blocks)


def _normalize_string_list(value: object) -> list[str]:
    """Normalize a parsed JSON field into a list of strings."""
    if not isinstance(value, list):
        return [FAILED_EXTRACTION]

    normalized_values = [str(item).strip() for item in value if str(item).strip()]
    return normalized_values or [FAILED_EXTRACTION]


def _normalize_contradictions(value: object) -> list[ContradictionItem]:
    """Normalize contradiction items from parsed JSON."""
    if not isinstance(value, list):
        return []

    contradictions: list[ContradictionItem] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        paper_a = str(item.get("paper_a", "")).strip()
        paper_b = str(item.get("paper_b", "")).strip()
        contradiction = str(item.get("contradiction", "")).strip()

        if not paper_a or not paper_b or not contradiction:
            continue

        contradictions.append(
            ContradictionItem(
                paper_a=paper_a,
                paper_b=paper_b,
                contradiction=contradiction,
            )
        )

    return contradictions


def fallback_gap_report(topic: str, papers: list[Paper]) -> GapReport:
    """Build a safe fallback gap report when extraction fails."""
    return GapReport(
        topic=topic.strip() or "Unspecified Topic",
        paper_count=len(papers),
        contradictions=[],
        recurring_limitations=[FAILED_EXTRACTION],
        underexplored_directions=[FAILED_EXTRACTION],
        explicit_research_gaps=[FAILED_EXTRACTION],
    )


def extract_gaps_and_contradictions(papers: list[Paper], topic: str, config: Config) -> GapReport:
    """Extract contradictions and research gaps across a collection of papers."""
    report_topic = topic.strip() or "Unspecified Topic"

    if not papers:
        return GapReport(
            topic=report_topic,
            paper_count=0,
            contradictions=[],
            recurring_limitations=[],
            underexplored_directions=[],
            explicit_research_gaps=[],
        )

    validate_provider_setup(config)
    client = get_client(config)
    gap_context = build_gap_context(papers)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        topic=report_topic,
        paper_count=len(papers),
        gap_context=gap_context,
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        return GapReport(
            topic=report_topic,
            paper_count=len(papers),
            contradictions=_normalize_contradictions(parsed.get("contradictions")),
            recurring_limitations=_normalize_string_list(parsed.get("recurring_limitations")),
            underexplored_directions=_normalize_string_list(parsed.get("underexplored_directions")),
            explicit_research_gaps=_normalize_string_list(parsed.get("explicit_research_gaps")),
        )
    except Exception:
        return fallback_gap_report(report_topic, papers)


def save_gap_report(report: GapReport, filepath: str) -> None:
    """Save a gap report as JSON, creating parent directories if needed."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
