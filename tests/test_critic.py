"""Tests for artifact-aware critic behavior."""

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
from research_copilot.critic import evaluate_state


SUMMARY_ROW = {
    "title": "Vision Transformers",
    "authors": ["Researcher One"],
    "abstract": "Vision transformers process image patches with self-attention.",
    "arxiv_id": "2101.00001",
    "pdf_url": "",
    "published": "2021-01-01",
    "categories": ["cs.CV"],
    "summary": {
        "arxiv_id": "2101.00001",
        "core_contribution": "Introduces image patch self-attention.",
        "methodology": "Uses transformer encoders over patch embeddings.",
        "key_result": "Improves image recognition with enough data.",
        "limitation": "Requires large-scale pretraining.",
        "raw_abstract": "Vision transformers process image patches with self-attention.",
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


def _state(iteration: int = 1, max_iterations: int = 6) -> AgentState:
    """Build a minimal agent state for critic tests."""
    return AgentState(
        topic="vision transformers",
        iteration=iteration,
        max_iterations=max_iterations,
        history=[],
        memory={},
        current_goal="test critic state",
    )


def _write_json(path: str, payload: object) -> None:
    """Write JSON test artifacts."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _write_downstream_through_gaps_without_fetch() -> None:
    """Create downstream artifacts that imply earlier steps are complete."""
    Path("output").mkdir(exist_ok=True)
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


class CriticArtifactStateTests(unittest.TestCase):
    """Verify critic decisions are based on current artifacts."""

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

    def test_critic_does_not_regress_to_stale_fetch_missing_logic(self) -> None:
        """Downstream artifacts imply fetch is satisfied even if results.json is absent."""
        _write_downstream_through_gaps_without_fetch()

        decision = evaluate_state(_state())

        self.assertIn("fetch", decision["completed_steps"])
        self.assertIn("gaps", decision["completed_steps"])
        self.assertEqual(decision["next_missing_step"], "hypotheses")
        self.assertNotEqual(decision["reason"], "Missing required step: fetch.")

    def test_missing_hypotheses_artifact_is_reported_by_critic(self) -> None:
        """Critic distinguishes missing hypotheses from malformed or valid."""
        _write_downstream_through_gaps_without_fetch()

        decision = evaluate_state(_state())

        self.assertEqual(decision["hypotheses_status"], "missing")
        self.assertEqual(decision["artifact_details"]["hypotheses"], "file missing")
        self.assertEqual(decision["next_missing_step"], "hypotheses")

    def test_malformed_hypotheses_artifact_is_reported_by_critic(self) -> None:
        """Critic rejects malformed hypotheses with a schema-specific reason."""
        _write_downstream_through_gaps_without_fetch()
        malformed = {**VALID_HYPOTHESES, "candidate_hypotheses": []}
        _write_json("output/hypotheses.json", malformed)

        decision = evaluate_state(_state())

        self.assertEqual(decision["hypotheses_status"], "malformed")
        self.assertIn("candidate_hypotheses must be a non-empty list", decision["hypotheses_detail"])
        self.assertEqual(decision["next_missing_step"], "hypotheses")

    def test_valid_hypotheses_artifact_advances_critic_to_experiment(self) -> None:
        """A valid hypotheses artifact is accepted as complete."""
        _write_downstream_through_gaps_without_fetch()
        _write_json("output/hypotheses.json", VALID_HYPOTHESES)

        decision = evaluate_state(_state())

        self.assertEqual(decision["hypotheses_status"], "valid")
        self.assertIn("hypotheses", decision["completed_steps"])
        self.assertEqual(decision["next_missing_step"], "experiment")

    def test_bounded_iterations_terminate_cleanly(self) -> None:
        """Critic returns done when max iterations have been reached."""
        decision = evaluate_state(_state(iteration=1, max_iterations=1))

        self.assertEqual(decision["status"], "done")
        self.assertEqual(decision["reason"], "Maximum iterations reached.")


if __name__ == "__main__":
    unittest.main()
