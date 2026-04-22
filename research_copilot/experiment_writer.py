"""Deterministic experiment script generation from hypothesis metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    from .models import HypothesisItem
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import HypothesisItem


DEFAULT_OUTPUT_DIR = "results/"


def _python_list_literal(items: list[str]) -> str:
    """Render a list of strings as a Python list literal."""
    escaped_items = [item.replace("\\", "\\\\").replace('"', '\\"') for item in items]
    return "[" + ", ".join(f'"{item}"' for item in escaped_items) + "]"


def _comment_lines(items: list[str], fallback: str = "None provided.") -> str:
    """Render list items as commented lines."""
    if not items:
        return f"# - {fallback}"
    return "\n".join(f"# - {item}" for item in items)


def generate_experiment_script(hypothesis: HypothesisItem, topic: str) -> str:
    """Generate a ready-to-adapt Python experiment scaffold from one hypothesis."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    datasets_literal = _python_list_literal(hypothesis.experiment_plan.datasets)
    metrics_literal = _python_list_literal(hypothesis.experiment_plan.metrics)

    script = f'''"""Experiment scaffold for hypothesis-driven research.

Hypothesis Title: {hypothesis.title}
Hypothesis: {hypothesis.hypothesis}
Topic: {topic}
Generated: {timestamp}
"""

import argparse
import json
import os
from pathlib import Path


SUGGESTED_DATASETS = {datasets_literal}
SUGGESTED_METRICS = {metrics_literal}


# === DATASETS ===
{_comment_lines(hypothesis.experiment_plan.datasets)}

# === BASELINES ===
{_comment_lines(hypothesis.experiment_plan.baselines)}

# === METRICS ===
{_comment_lines(hypothesis.experiment_plan.metrics)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Hypothesis experiment scaffold")
    parser.add_argument("--dataset", type=str, required=True, help="Path or name of the training dataset")
    parser.add_argument("--output_dir", type=str, default="{DEFAULT_OUTPUT_DIR}", help="Directory for run outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data(args) -> dict:
    """Load and return dataset artifacts."""
    # TODO: implement data loading for {", ".join(hypothesis.experiment_plan.datasets) or "the selected dataset"}
    return {{"dataset": args.dataset, "train": None, "validation": None, "test": None}}


def build_model(args) -> object:
    """Build and return the model object."""
    # TODO: implement model for objective: {hypothesis.experiment_plan.objective}
    return None


def train(model, data, args) -> dict:
    """Run training and return training artifacts."""
    # TODO: implement training loop
    return {{"epochs": args.epochs, "results": []}}


def evaluate(model, data, args) -> dict:
    """Evaluate the model and return metric outputs."""
    # TODO: implement evaluation
    metrics_list = SUGGESTED_METRICS
    return {{"metrics": {{m: None for m in metrics_list}}}}


def main() -> None:
    """Run the experiment scaffold."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args)
    model = build_model(args)
    train_output = train(model, data, args)
    eval_output = evaluate(model, data, args)

    result_payload = {{
        "hypothesis_title": "{hypothesis.title}",
        "topic": "{topic}",
        "objective": "{hypothesis.experiment_plan.objective}",
        "train": train_output,
        "evaluation": eval_output,
    }}

    output_path = Path(args.output_dir) / "results.json"
    output_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(f"Saved results to {{output_path}}")


if __name__ == "__main__":
    main()


# === IMPLEMENTATION NOTES ===
{_comment_lines(hypothesis.experiment_plan.implementation_notes)}
'''
    return script


def save_experiment_script(script: str, filepath: str) -> None:
    """Save a generated experiment scaffold script to disk."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")
