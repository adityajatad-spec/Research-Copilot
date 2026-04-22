"""Cross-paper insight extraction helpers."""

from __future__ import annotations

import json
import re

try:
    from .config import Config, get_client, validate_provider_setup
    from .models import InsightReport, Paper
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config, get_client, validate_provider_setup
    from models import InsightReport, Paper


CONTEXT_CHAR_LIMIT = 20_000
TEXT_PREVIEW_LIMIT = 300
FALLBACK_ITEM = "[extraction failed]"
SYSTEM_PROMPT = (
    "You are a research synthesis analyst. "
    "Given a set of papers on one topic, extract cross-paper insights as JSON. "
    "Return ONLY valid JSON with exactly these keys: "
    "major_themes, common_methodologies, emerging_subtopics, notable_observations. "
    "Each value must be a list of short, factual strings. "
    "major_themes should contain 3 to 5 items. "
    "common_methodologies, emerging_subtopics, and notable_observations should each contain 2 to 5 items. "
    "Focus on patterns that appear across papers, not single-paper summaries."
)
USER_PROMPT_TEMPLATE = (
    "Topic: {topic}\n"
    "Paper count: {paper_count}\n\n"
    "Paper context:\n"
    "{context}\n\n"
    "Return JSON with these exact keys:\n"
    "major_themes, common_methodologies, emerging_subtopics, notable_observations"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def _clean_text(value: str, limit: int | None = None) -> str:
    """Normalize whitespace and optionally truncate text."""
    cleaned_value = WHITESPACE_PATTERN.sub(" ", value.strip())
    if limit is not None and len(cleaned_value) > limit:
        return cleaned_value[:limit].rstrip()
    return cleaned_value


def _paper_sort_key(paper: Paper) -> tuple[int, int, str]:
    """Build a stable priority key for context ordering."""
    citation_count = paper.citation_count or 0
    influential_count = paper.influential_citations or 0
    return (citation_count, influential_count, paper.title.lower())


def _paper_preview(paper: Paper) -> str:
    """Return a compact preview from full text or abstract."""
    source_text = paper.full_text or paper.abstract or ""
    preview = _clean_text(source_text, limit=TEXT_PREVIEW_LIMIT)
    if not preview:
        return "[no text available]"
    return f"{preview} [truncated]"


def _paper_context_block(index: int, paper: Paper) -> str:
    """Build a context block for a single paper."""
    source = paper.source or "unknown"
    citation_count = paper.citation_count or 0
    categories = ", ".join(paper.categories) if paper.categories else "none"
    lines = [
        f'Paper {index}: "{paper.title}" [{source}, {citation_count} citations]',
        f"Categories: {categories}",
    ]

    if paper.summary is not None:
        lines.extend(
            [
                f"Contribution: {_clean_text(paper.summary.core_contribution)}",
                f"Methodology: {_clean_text(paper.summary.methodology)}",
                f"Result: {_clean_text(paper.summary.key_result)}",
                f"Limitation: {_clean_text(paper.summary.limitation)}",
            ]
        )

    preview_label = "[full_text preview]" if paper.full_text else "[abstract preview]"
    lines.append(f"{preview_label} {_paper_preview(paper)}")
    return "\n".join(lines)


def build_paper_context(papers: list[Paper]) -> str:
    """Build a compact cross-paper context string for LLM consumption."""
    sorted_papers = sorted(papers, key=_paper_sort_key, reverse=True)
    context_blocks: list[str] = []
    current_length = 0

    for index, paper in enumerate(sorted_papers, start=1):
        block = _paper_context_block(index, paper)
        separator = "\n\n---\n\n" if context_blocks else ""
        proposed_length = current_length + len(separator) + len(block)

        if proposed_length > CONTEXT_CHAR_LIMIT:
            remaining_chars = CONTEXT_CHAR_LIMIT - current_length - len(separator)
            if remaining_chars > 100:
                truncated_block = f"{block[: remaining_chars - 14].rstrip()}\n\n[truncated]"
                context_blocks.append(f"{separator}{truncated_block}" if separator else truncated_block)
            break

        context_blocks.append(f"{separator}{block}" if separator else block)
        current_length = proposed_length

    return "".join(context_blocks)


def _normalize_list(value: object, fallback_count: int) -> list[str]:
    """Normalize a parsed JSON field into a list of strings."""
    if not isinstance(value, list):
        return [FALLBACK_ITEM] * fallback_count

    normalized_items = [str(item).strip() for item in value if str(item).strip()]
    return normalized_items or [FALLBACK_ITEM] * fallback_count


def _failed_insight_report(topic: str, paper_count: int) -> InsightReport:
    """Build a fallback insight report when extraction fails."""
    return InsightReport(
        topic=topic,
        paper_count=paper_count,
        major_themes=[FALLBACK_ITEM],
        common_methodologies=[FALLBACK_ITEM],
        emerging_subtopics=[FALLBACK_ITEM],
        notable_observations=[FALLBACK_ITEM],
    )


def extract_insights(papers: list[Paper], topic: str, config: Config) -> InsightReport:
    """Extract cross-paper insights for a topic using the configured LLM provider."""
    report_topic = topic.strip() or "Unspecified Topic"
    paper_count = len(papers)

    if paper_count == 0:
        return InsightReport(
            topic=report_topic,
            paper_count=0,
            major_themes=[],
            common_methodologies=[],
            emerging_subtopics=[],
            notable_observations=[],
        )

    validate_provider_setup(config)
    client = get_client(config)
    context = build_paper_context(papers)
    user_prompt = USER_PROMPT_TEMPLATE.format(topic=report_topic, paper_count=paper_count, context=context)

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

        return InsightReport(
            topic=report_topic,
            paper_count=paper_count,
            major_themes=_normalize_list(parsed.get("major_themes"), fallback_count=1),
            common_methodologies=_normalize_list(parsed.get("common_methodologies"), fallback_count=1),
            emerging_subtopics=_normalize_list(parsed.get("emerging_subtopics"), fallback_count=1),
            notable_observations=_normalize_list(parsed.get("notable_observations"), fallback_count=1),
        )
    except Exception:
        return _failed_insight_report(report_topic, paper_count)
