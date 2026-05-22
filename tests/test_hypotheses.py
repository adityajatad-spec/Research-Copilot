"""Tests for structured hypothesis artifact generation and validation."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda *args, **kwargs: None)

from research_copilot.config import Config, OLLAMA_NOT_RUNNING_MESSAGE
from research_copilot.hypotheses import extract_hypotheses
from research_copilot.models import Paper, PaperSummary
from research_copilot.planner import validate_hypotheses_payload


def _valid_hypotheses_payload() -> dict:
    """Build a valid structured hypotheses payload."""
    return {
        "detected_context": "vision transformers",
        "source_artifacts": ["summarize", "insights", "gaps"],
        "candidate_hypotheses": [
            {
                "title": "Data quality bottleneck",
                "claim": "Noisy pretraining data is the dominant bottleneck for smaller vision transformers.",
                "supporting_evidence": ["Recurring limitation: data quality", "Gap: robustness under noisy data"],
                "confidence": 0.78,
                "priority": "high",
                "next_actions": ["Search robustness papers", "Compare data-cleaning baselines"],
            }
        ],
        "notes": "Needs validation against source diversity.",
    }


def _paper() -> Paper:
    """Build one paper with a structured summary for provider tests."""
    return Paper(
        title="Vision Transformers",
        authors=["Researcher One"],
        abstract="Vision transformers process image patches with self-attention.",
        arxiv_id="2101.00001",
        pdf_url="",
        published="2021-01-01",
        categories=["cs.CV"],
        summary=PaperSummary(
            arxiv_id="2101.00001",
            core_contribution="Introduces image patch self-attention.",
            methodology="Uses transformer encoders over patch embeddings.",
            key_result="Improves image recognition with enough data.",
            limitation="Requires large-scale pretraining.",
            raw_abstract="Vision transformers process image patches with self-attention.",
        ),
    )


class HypothesesValidationTests(unittest.TestCase):
    """Validate the hypotheses schema and provider failure path."""

    def test_valid_hypotheses_payload_is_accepted(self) -> None:
        """A structured hypotheses artifact passes deterministic validation."""
        ready, detail = validate_hypotheses_payload(_valid_hypotheses_payload())

        self.assertTrue(ready)
        self.assertIn("valid hypotheses artifact", detail)

    def test_malformed_hypotheses_payload_is_rejected_with_reason(self) -> None:
        """A malformed hypotheses artifact is rejected with a specific reason."""
        payload = _valid_hypotheses_payload()
        payload["candidate_hypotheses"][0]["confidence"] = "very high"

        ready, detail = validate_hypotheses_payload(payload)

        self.assertFalse(ready)
        self.assertIn("confidence must be numeric", detail)

    def test_provider_unavailable_during_hypothesis_generation_is_explicit(self) -> None:
        """Ollama connection failures surface as clear provider setup errors."""
        with tempfile.TemporaryDirectory():
            config = Config(
                provider="ollama",
                model="llama3.2",
                ollama_base_url="http://127.0.0.1:9/v1",
            )

            with self.assertRaisesRegex(ValueError, OLLAMA_NOT_RUNNING_MESSAGE):
                extract_hypotheses([_paper()], "vision transformers", config)


if __name__ == "__main__":
    unittest.main()
