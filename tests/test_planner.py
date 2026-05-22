"""Tests for artifact-aware planner progression."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda *args, **kwargs: None)

from research_copilot.agent_state import AgentState
from research_copilot.planner import fallback_plan, inspect_artifacts


PAPER_ROW = {
    "title": "Vision Transformers",
    "authors": ["Researcher One"],
    "abstract": "Vision transformers process image patches with self-attention.",
    "arxiv_id": "2101.00001",
    "pdf_url": "",
    "published": "2021-01-01",
    "categories": ["cs.CV"],
}

SUMMARY_ROW = {
    **PAPER_ROW,
    "summary": {
        "arxiv_id": "2101.00001",
        "core_contribution": "Introduces image patch self-attention.",
        "methodology": "Uses transformer encoders over patch embeddings.",
        "key_result": "Improves image recognition with enough data.",
        "limitation": "Requires large-scale pretraining.",
        "raw_abstract": PAPER_ROW["abstract"],
    },
}

VALID_HYPOTHESES = {
    "detected_context": "vision transformers",
    "source_artifacts": ["summarize", "insights", "gaps"],
    "candidate_hypotheses": [
        {
            "title": "Data quality bottleneck",
            "claim": "Noisy pretraining data is the dominant bottleneck for smaller vision transformers.",
            "supporting_evidence": ["Recurring limitation: data quality"],
            "confidence": 0.78,
            "priority": "high",
            "next_actions": ["Search robustness papers"],
        }
    ],
    "notes": "Needs validation.",
}


def _state() -> AgentState:
    """Build a minimal agent state for planner tests."""
    return AgentState(
        topic="vision transformers",
        iteration=0,
        max_iterations=6,
        history=[],
        memory={},
        current_goal="test pipeline progression",
    )


def _write_json(path: str, payload: object) -> None:
    """Write JSON test artifacts."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _write_upstream_through_gaps() -> None:
    """Create valid artifacts through the gaps stage."""
    _write_json("output/results.json", [PAPER_ROW])
    _write_json("output/papers_with_pdf.json", [PAPER_ROW])
    _write_json("output/summaries.json", [SUMMARY_ROW])
    Path("output/report.md").write_text("# Research Report\n", encoding="utf-8")
    _write_json("output/insights.json", {"major_themes": ["attention"], "common_methodologies": ["patches"]})
    _write_json(
        "output/gaps.json",
        {
            "topic": "vision transformers",
            "paper_count": 1,
            "contradictions": [],
            "recurring_limitations": ["data hunger"],
            "underexplored_directions": ["small-data robustness"],
            "explicit_research_gaps": ["limited robustness evidence"],
        },
    )


class PlannerProgressionTests(unittest.TestCase):
    """Verify planner progression around the hypotheses stage."""

    def setUp(self) -> None:
        """Create an isolated artifact workspace."""
        self.tempdir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tempdir.name)
        Path("output").mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Restore the original workspace."""
        os.chdir(self.cwd)
        self.tempdir.cleanup()

    def test_valid_gaps_to_hypotheses_progression(self) -> None:
        """When gaps exist and hypotheses is missing, planner selects hypotheses."""
        _write_upstream_through_gaps()

        plan = fallback_plan(_state())

        self.assertEqual(plan["action"], "hypotheses")
        self.assertIn("hypotheses", " ".join(plan.get("debug_skips", [])))

    def test_missing_hypotheses_artifact_is_reported(self) -> None:
        """Missing hypotheses artifact is distinct from a valid artifact."""
        _write_upstream_through_gaps()

        inspection = inspect_artifacts(_state())

        self.assertFalse(inspection["hypotheses"]["ready"])
        self.assertEqual(inspection["hypotheses"]["detail"], "file missing")

    def test_malformed_hypotheses_artifact_is_rejected(self) -> None:
        """Malformed hypotheses artifact keeps planner on the hypotheses step."""
        _write_upstream_through_gaps()
        malformed = {**VALID_HYPOTHESES, "candidate_hypotheses": []}
        _write_json("output/hypotheses.json", malformed)

        inspection = inspect_artifacts(_state())
        plan = fallback_plan(_state(), inspection=inspection)

        self.assertFalse(inspection["hypotheses"]["ready"])
        self.assertIn("candidate_hypotheses must be a non-empty list", inspection["hypotheses"]["detail"])
        self.assertEqual(plan["action"], "hypotheses")

    def test_hypotheses_is_skipped_when_upstream_evidence_is_insufficient(self) -> None:
        """Planner does not force hypotheses before gaps are available."""
        _write_json("output/results.json", [PAPER_ROW])
        _write_json("output/papers_with_pdf.json", [PAPER_ROW])
        _write_json("output/summaries.json", [SUMMARY_ROW])
        Path("output/report.md").write_text("# Research Report\n", encoding="utf-8")
        _write_json("output/insights.json", {"major_themes": ["attention"]})

        plan = fallback_plan(_state())

        self.assertEqual(plan["action"], "gaps")


if __name__ == "__main__":
    unittest.main()
