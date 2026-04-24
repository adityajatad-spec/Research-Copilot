"""Deterministic scoring helpers for benchmarked agent runs."""

from __future__ import annotations

from pathlib import Path


def _resolve_run_path(path_value: str, task_output_dir: Path | None) -> Path:
    """Resolve a possibly-relative run path against task output directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if task_output_dir is not None:
        candidate = task_output_dir / path
        if candidate.exists():
            return candidate
    return path


def _artifact_present(required_name: str, run_data: dict) -> bool:
    """Check whether a required artifact exists in run outputs or task directory."""
    final_output_paths = run_data.get("final_output_paths", {})
    task_output_dir_value = run_data.get("task_output_dir")
    task_output_dir = Path(task_output_dir_value) if isinstance(task_output_dir_value, str) else None

    if isinstance(final_output_paths, dict):
        for path_value in final_output_paths.values():
            if not isinstance(path_value, str):
                continue
            resolved = _resolve_run_path(path_value, task_output_dir)
            if resolved.name == required_name and resolved.exists():
                return True

    candidates: list[Path] = []
    if task_output_dir is not None:
        candidates.append(task_output_dir / "output" / required_name)
        candidates.append(task_output_dir / required_name)
    else:
        candidates.append(Path("output") / required_name)
        candidates.append(Path(required_name))

    return any(candidate.exists() for candidate in candidates)


def _iteration_score(iterations: int) -> float:
    """Return the deterministic iteration score."""
    if iterations <= 4:
        return 1.0
    if iterations <= 6:
        return 0.7
    if iterations <= 8:
        return 0.4
    return 0.0


def _experiment_score(run_data: dict) -> float:
    """Return experiment evidence score from final output paths."""
    final_output_paths = run_data.get("final_output_paths", {})
    if not isinstance(final_output_paths, dict):
        return 0.0

    experiment_results = final_output_paths.get("experiment_results")
    if isinstance(experiment_results, str):
        return 1.0

    experiment_script = final_output_paths.get("experiment_script")
    if isinstance(experiment_script, str):
        return 0.5

    return 0.0


def score_agent_run(task: dict, run_data: dict) -> dict:
    """Score one benchmark task run using deterministic file/state checks."""
    required_artifacts = task.get("required_artifacts", [])
    required_list = [str(item) for item in required_artifacts if str(item).strip()]
    present_count = sum(1 for artifact_name in required_list if _artifact_present(artifact_name, run_data))
    total_required = len(required_list) or 1
    artifact_score = present_count / total_required

    iterations = int(run_data.get("iterations", 0) or 0)
    iteration_score = _iteration_score(iterations)
    experiment_score = _experiment_score(run_data)

    total_score = 0.6 * artifact_score + 0.25 * iteration_score + 0.15 * experiment_score
    completed = bool(run_data.get("done")) and artifact_score >= 1.0

    notes: list[str] = []
    if not bool(run_data.get("done")):
        notes.append("Run did not reach done=True.")
    if artifact_score < 1.0:
        notes.append(f"Missing required artifacts: {len(required_list) - present_count}.")
    if experiment_score == 0.0:
        notes.append("No experiment artifact evidence.")

    return {
        "task_id": str(task.get("id", "")),
        "topic": str(task.get("topic", "")),
        "completed": completed,
        "artifact_score": round(artifact_score, 4),
        "iteration_score": round(iteration_score, 4),
        "experiment_score": round(experiment_score, 4),
        "total_score": round(total_score, 4),
        "iterations": iterations,
        "notes": notes,
    }


def aggregate_scores(results: list[dict]) -> dict:
    """Aggregate benchmark score rows into summary metrics."""
    if not results:
        return {
            "task_count": 0,
            "completed_count": 0,
            "average_total_score": 0.0,
            "average_artifact_score": 0.0,
            "average_iteration_score": 0.0,
            "average_experiment_score": 0.0,
            "best_task": "",
            "worst_task": "",
        }

    task_count = len(results)
    completed_count = sum(1 for row in results if bool(row.get("completed")))
    average_total_score = sum(float(row.get("total_score", 0.0)) for row in results) / task_count
    average_artifact_score = sum(float(row.get("artifact_score", 0.0)) for row in results) / task_count
    average_iteration_score = sum(float(row.get("iteration_score", 0.0)) for row in results) / task_count
    average_experiment_score = sum(float(row.get("experiment_score", 0.0)) for row in results) / task_count

    best_row = max(results, key=lambda row: float(row.get("total_score", 0.0)))
    worst_row = min(results, key=lambda row: float(row.get("total_score", 0.0)))

    return {
        "task_count": task_count,
        "completed_count": completed_count,
        "average_total_score": round(average_total_score, 4),
        "average_artifact_score": round(average_artifact_score, 4),
        "average_iteration_score": round(average_iteration_score, 4),
        "average_experiment_score": round(average_experiment_score, 4),
        "best_task": str(best_row.get("task_id", "")),
        "worst_task": str(worst_row.get("task_id", "")),
    }
