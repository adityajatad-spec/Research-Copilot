"""Helpers for loading and summarizing experiment execution outputs."""

from __future__ import annotations

import json
from pathlib import Path


def load_experiment_results(results_path: str) -> dict | None:
    """Load experiment results JSON when available and valid."""
    path = Path(results_path)
    if not path.exists():
        return None

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def extract_result_signals(parsed_results: dict | None) -> dict:
    """Extract lightweight, best-effort signals from parsed experiment results."""
    if not isinstance(parsed_results, dict):
        return {
            "has_results": False,
            "metric_keys": [],
            "epochs": None,
            "result_count": 0,
        }

    metrics = parsed_results.get("evaluation", {}).get("metrics", {})
    metric_keys = list(metrics.keys()) if isinstance(metrics, dict) else []

    epochs = parsed_results.get("train", {}).get("epochs")
    if epochs is None:
        epochs = parsed_results.get("epochs")

    results_list = parsed_results.get("train", {}).get("results", [])
    result_count = len(results_list) if isinstance(results_list, list) else 0

    return {
        "has_results": True,
        "metric_keys": metric_keys,
        "epochs": epochs,
        "result_count": result_count,
    }


def summarize_experiment_result(run_result: dict, parsed_results: dict | None) -> str:
    """Build a compact memory-friendly summary for one experiment run."""
    success = bool(run_result.get("success"))
    error = str(run_result.get("error") or "").strip()
    return_code = run_result.get("returncode")
    stderr = str(run_result.get("stderr") or "")

    if error == "timeout":
        timeout_value = run_result.get("timeout")
        if isinstance(timeout_value, int):
            return f"Experiment timed out after {timeout_value} seconds."
        return "Experiment timed out."

    if success:
        if parsed_results is None:
            return "Experiment ran successfully; no valid results.json was parsed."
        signals = extract_result_signals(parsed_results)
        metric_keys = signals.get("metric_keys", [])
        if metric_keys:
            keys_text = ", ".join(str(key) for key in metric_keys)
            return f"Experiment ran successfully; results.json found; metrics keys: {keys_text}"
        return "Experiment ran successfully; results.json found; no metric keys detected."

    if return_code is not None:
        if "ModuleNotFoundError" in stderr:
            return f"Experiment failed with return code {return_code}; stderr contains ModuleNotFoundError."
        return f"Experiment failed with return code {return_code}."

    if error:
        return f"Experiment failed: {error}"

    return "Experiment failed with unknown error."
