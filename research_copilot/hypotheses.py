"""Cross-paper hypothesis generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
import re

try:
    from .config import Config, get_client, validate_provider_setup
    from .models import ExperimentPlan, HypothesisItem, HypothesisReport, Paper
except ImportError:  # pragma: no cover - fallback for direct script execution
    from config import Config, get_client, validate_provider_setup
    from models import ExperimentPlan, HypothesisItem, HypothesisReport, Paper


CONTEXT_CHAR_LIMIT = 20_000
TOP_PAPERS_COUNT = 5
FAILED_EXTRACTION = "[hypothesis extraction failed]"
NO_DATA_AVAILABLE = "[not available]"
WHITESPACE_PATTERN = re.compile(r"\s+")
SYSTEM_PROMPT = """
You are an expert ML research strategist.

You are given structured evidence from a literature analysis:
- explicit research gaps identified across multiple papers
- underexplored directions
- recurring limitations
- contradictions between papers
- common methodologies
- key paper summaries

Your task is to propose exactly 3 strong, evidence-grounded research hypotheses.

For each hypothesis:
- Make it specific, testable, and grounded in the gaps above
- Explain WHY it is novel (what has not been done)
- Explain WHY it is feasible (what already exists that makes this buildable)
- Propose a concrete experiment plan

Return ONLY valid JSON with this exact schema:
{
  "generated_from_gaps": [
    "one-line description of the primary gap this research addresses"
  ],
  "hypotheses": [
    {
      "title": "Short descriptive title for this research direction",
      "hypothesis": "One specific, testable research claim",
      "novelty_rationale": "Why this has not been done or done well",
      "feasibility_rationale": "Why a researcher could execute this now",
      "experiment_plan": {
        "objective": "One-sentence experiment goal",
        "datasets": ["dataset 1", "dataset 2"],
        "baselines": ["model or method 1", "model or method 2"],
        "metrics": ["metric 1", "metric 2"],
        "implementation_notes": [
          "Practical note 1",
          "Practical note 2"
        ]
      }
    }
  ]
}

Rules:
- Return EXACTLY 3 hypotheses in the list
- generated_from_gaps should list the 2-3 gaps most relevant to these hypotheses
- Each hypothesis must be grounded in the literature evidence provided
- Keep experiment plans realistic for a solo ML researcher
- Be conservative and evidence-based
""".strip()
USER_PROMPT_TEMPLATE = """
Topic: {topic}
Papers analyzed: {paper_count}

{context}

Return JSON only.
""".strip()


def load_optional_json(filepath: str) -> dict | None:
    """Load JSON from disk if the file exists; return None otherwise."""
    path = Path(filepath)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean_text(value: str) -> str:
    """Normalize whitespace in text content."""
    return WHITESPACE_PATTERN.sub(" ", value.strip())


def _normalize_list(value: object) -> list[str]:
    """Normalize unknown input into a list of clean strings."""
    if not isinstance(value, list):
        return []
    items = [str(item).strip() for item in value if str(item).strip()]
    return items


def _render_section(title: str, lines: list[str]) -> str:
    """Render one context section with a standardized header."""
    rendered_lines = lines or [NO_DATA_AVAILABLE]
    return f"=== {title} ===\n" + "\n".join(f"- {_clean_text(line)}" for line in rendered_lines)


def _append_with_cap(existing: str, section: str, limit: int = CONTEXT_CHAR_LIMIT) -> str:
    """Append a section while respecting the total context length cap."""
    if not existing:
        candidate = section
    else:
        candidate = f"{existing}\n\n{section}"

    if len(candidate) <= limit:
        return candidate

    remaining_chars = limit - len(existing) - (2 if existing else 0)
    if remaining_chars <= 0:
        return existing[:limit]

    truncated_section = section[: max(0, remaining_chars - 14)].rstrip() + "\n\n[truncated]"
    if not existing:
        return truncated_section[:limit]
    return f"{existing}\n\n{truncated_section}"[:limit]


def _top_paper_lines(papers: list[Paper]) -> list[str]:
    """Render top paper summaries prioritized by citation count."""
    sorted_papers = sorted(
        papers,
        key=lambda paper: (
            paper.citation_count or 0,
            1 if paper.summary is not None else 0,
            paper.title.lower(),
        ),
        reverse=True,
    )
    lines: list[str] = []
    for index, paper in enumerate(sorted_papers[:TOP_PAPERS_COUNT], start=1):
        contribution = NO_DATA_AVAILABLE
        methodology = NO_DATA_AVAILABLE
        limitation = NO_DATA_AVAILABLE

        if paper.summary is not None:
            contribution = paper.summary.core_contribution or NO_DATA_AVAILABLE
            methodology = paper.summary.methodology or NO_DATA_AVAILABLE
            limitation = paper.summary.limitation or NO_DATA_AVAILABLE

        lines.append(
            f'Paper {index}: "{paper.title}" | '
            f"Contribution: {_clean_text(contribution)} | "
            f"Methodology: {_clean_text(methodology)} | "
            f"Limitation: {_clean_text(limitation)}"
        )
    return lines


def build_hypothesis_context(
    papers: list[Paper],
    insights_data: dict | None,
    gap_data: dict | None,
    topic: str,
) -> str:
    """Build a compact, prioritized context string for hypothesis generation."""
    safe_gap_data = gap_data or {}
    safe_insights_data = insights_data or {}

    contradiction_lines: list[str] = []
    if isinstance(safe_gap_data.get("contradictions"), list):
        for entry in safe_gap_data["contradictions"]:
            if isinstance(entry, dict):
                paper_a = str(entry.get("paper_a", "")).strip()
                paper_b = str(entry.get("paper_b", "")).strip()
                contradiction = str(entry.get("contradiction", "")).strip()
                if paper_a and paper_b and contradiction:
                    contradiction_lines.append(f"{paper_a} vs {paper_b}: {contradiction}")
            else:
                text_entry = str(entry).strip()
                if text_entry:
                    contradiction_lines.append(text_entry)

    sections = [
        _render_section("Topic", [topic.strip() or "Unspecified Topic"]),
        _render_section("Research Gaps", _normalize_list(safe_gap_data.get("explicit_research_gaps"))),
        _render_section("Underexplored Directions", _normalize_list(safe_gap_data.get("underexplored_directions"))),
        _render_section("Recurring Limitations", _normalize_list(safe_gap_data.get("recurring_limitations"))),
        _render_section("Contradictions", contradiction_lines),
        _render_section("Major Themes", _normalize_list(safe_insights_data.get("major_themes"))),
        _render_section("Common Methodologies", _normalize_list(safe_insights_data.get("common_methodologies"))),
        _render_section("Top Papers", _top_paper_lines(papers)),
    ]

    context = ""
    for section in sections:
        before = context
        context = _append_with_cap(context, section, limit=CONTEXT_CHAR_LIMIT)
        if len(context) >= CONTEXT_CHAR_LIMIT or context == before:
            break

    return context


def _normalize_string_list(value: object) -> list[str]:
    """Normalize parsed values into a list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_experiment_plan(value: object) -> ExperimentPlan:
    """Parse a nested experiment plan object."""
    if not isinstance(value, dict):
        raise ValueError("Invalid experiment_plan payload.")

    objective = str(value.get("objective", "")).strip()
    datasets = _normalize_string_list(value.get("datasets"))
    baselines = _normalize_string_list(value.get("baselines"))
    metrics = _normalize_string_list(value.get("metrics"))
    implementation_notes = _normalize_string_list(value.get("implementation_notes"))

    if not objective:
        raise ValueError("Missing experiment objective.")

    return ExperimentPlan(
        objective=objective,
        datasets=datasets,
        baselines=baselines,
        metrics=metrics,
        implementation_notes=implementation_notes,
    )


def fallback_hypothesis_report(topic: str, papers: list[Paper]) -> HypothesisReport:
    """Build a safe fallback hypothesis report."""
    return HypothesisReport(
        topic=topic.strip() or "Unspecified Topic",
        paper_count=len(papers),
        generated_from_gaps=[FAILED_EXTRACTION],
        hypotheses=[],
    )


def extract_hypotheses(
    papers: list[Paper],
    topic: str,
    config: Config,
    insights_data: dict | None = None,
    gap_data: dict | None = None,
) -> HypothesisReport:
    """Extract evidence-grounded hypotheses from cross-paper context."""
    report_topic = topic.strip() or "Unspecified Topic"

    if not papers:
        return HypothesisReport(
            topic=report_topic,
            paper_count=0,
            generated_from_gaps=[],
            hypotheses=[],
        )

    validate_provider_setup(config)
    client = get_client(config)
    context = build_hypothesis_context(papers, insights_data, gap_data, report_topic)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        topic=report_topic,
        paper_count=len(papers),
        context=context,
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        generated_from_gaps = _normalize_string_list(parsed.get("generated_from_gaps"))
        raw_hypotheses = parsed.get("hypotheses")
        if not isinstance(raw_hypotheses, list):
            return fallback_hypothesis_report(report_topic, papers)

        hypothesis_items: list[HypothesisItem] = []
        for raw_item in raw_hypotheses[:3]:
            if not isinstance(raw_item, dict):
                return fallback_hypothesis_report(report_topic, papers)

            title = str(raw_item.get("title", "")).strip()
            hypothesis_text = str(raw_item.get("hypothesis", "")).strip()
            novelty_rationale = str(raw_item.get("novelty_rationale", "")).strip()
            feasibility_rationale = str(raw_item.get("feasibility_rationale", "")).strip()
            experiment_plan = _parse_experiment_plan(raw_item.get("experiment_plan"))

            if not title or not hypothesis_text or not novelty_rationale or not feasibility_rationale:
                return fallback_hypothesis_report(report_topic, papers)

            hypothesis_items.append(
                HypothesisItem(
                    title=title,
                    hypothesis=hypothesis_text,
                    novelty_rationale=novelty_rationale,
                    feasibility_rationale=feasibility_rationale,
                    experiment_plan=experiment_plan,
                )
            )

        if len(hypothesis_items) != 3:
            return fallback_hypothesis_report(report_topic, papers)

        return HypothesisReport(
            topic=report_topic,
            paper_count=len(papers),
            generated_from_gaps=generated_from_gaps or [NO_DATA_AVAILABLE],
            hypotheses=hypothesis_items,
        )
    except Exception:
        return fallback_hypothesis_report(report_topic, papers)


def save_hypothesis_report(report: HypothesisReport, filepath: str) -> None:
    """Save a hypothesis report to JSON."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
