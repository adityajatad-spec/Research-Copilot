"""Lightweight benchmark harness for autonomous agent evaluation."""

from __future__ import annotations

import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path

try:
    from .agent_loop import run_agent
    from .benchmark_tasks import get_benchmark_tasks
    from .scoring import aggregate_scores, score_agent_run
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_loop import run_agent
    from benchmark_tasks import get_benchmark_tasks
    from scoring import aggregate_scores, score_agent_run


@contextmanager
def _working_directory(path: Path):
    """Temporarily switch working directory for isolated task execution."""
    previous_path = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_path)


def write_scores_csv(results: list[dict], filepath: str) -> None:
    """Write benchmark score rows to a CSV file."""
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task_id",
        "topic",
        "completed",
        "total_score",
        "artifact_score",
        "iteration_score",
        "experiment_score",
        "iterations",
        "partial_result",
        "failure_count",
        "notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            notes_value = row.get("notes", [])
            if isinstance(notes_value, list):
                notes_text = " | ".join(str(item) for item in notes_value)
            else:
                notes_text = str(notes_value)

            writer.writerow(
                {
                    "task_id": row.get("task_id", ""),
                    "topic": row.get("topic", ""),
                    "completed": row.get("completed", False),
                    "total_score": row.get("total_score", 0.0),
                    "artifact_score": row.get("artifact_score", 0.0),
                    "iteration_score": row.get("iteration_score", 0.0),
                    "experiment_score": row.get("experiment_score", 0.0),
                    "iterations": row.get("iterations", 0),
                    "partial_result": row.get("partial_result", False),
                    "failure_count": row.get("failure_count", 0),
                    "notes": notes_text,
                }
            )


def _history_failure_count(run_data: dict) -> int:
    """Return failed action count from run history."""
    history = run_data.get("history", [])
    if not isinstance(history, list):
        return 0
    return sum(1 for item in history if isinstance(item, dict) and str(item.get("status", "")) == "failed")


def _failure_prone_tasks(score_rows: list[dict]) -> list[str]:
    """Return top failure-prone task ids based on failures and score."""
    ranked = sorted(
        score_rows,
        key=lambda row: (
            -int(row.get("failure_count", 0) or 0),
            float(row.get("total_score", 0.0)),
        ),
    )
    picked: list[str] = []
    for row in ranked:
        task_id = str(row.get("task_id", "")).strip()
        failure_count = int(row.get("failure_count", 0) or 0)
        total_score = float(row.get("total_score", 0.0) or 0.0)
        if not task_id:
            continue
        if failure_count <= 0 and total_score >= 0.6:
            continue
        picked.append(task_id)
        if len(picked) >= 3:
            break
    return picked


def run_benchmark(
    provider: str,
    model: str,
    max_iterations: int = 6,
    limit: int | None = None,
    output_dir: str = "output/evals",
) -> dict:
    """Run the autonomous agent across benchmark tasks and score each run."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = get_benchmark_tasks(limit=limit)

    run_rows: list[dict] = []
    score_rows: list[dict] = []

    for task in tasks:
        task_id = str(task.get("id", "unknown_task"))
        topic = str(task.get("topic", "")).strip()
        task_output_dir = output_root / task_id
        run_output_path = output_root / f"{task_id}_run.json"

        run_data: dict
        try:
            with _working_directory(task_output_dir):
                run_data = run_agent(
                    topic=topic,
                    provider=provider,
                    model=model,
                    max_iterations=max_iterations,
                )
        except Exception as error:
            run_data = {
                "topic": topic,
                "done": False,
                "iterations": 0,
                "history": [],
                "memory_summary": "",
                "final_output_paths": {},
                "error": str(error),
            }

        run_data["task_id"] = task_id
        run_data["task_topic"] = topic
        run_data["task_output_dir"] = str(task_output_dir)
        run_output_path.write_text(json.dumps(run_data, indent=2), encoding="utf-8")

        score_row = score_agent_run(task, run_data)
        score_row["failure_summary"] = str(run_data.get("failure_summary", ""))
        score_row["repair_summary"] = str(run_data.get("repair_summary", ""))
        score_row["partial_result"] = bool(run_data.get("partial_result", False))
        score_row["failure_count"] = _history_failure_count(run_data)
        if score_row["partial_result"]:
            notes = score_row.get("notes", [])
            if isinstance(notes, list):
                notes.append("Run ended with partial_result=True.")
                score_row["notes"] = notes
        score_rows.append(score_row)
        run_rows.append(
            {
                "task": task,
                "run": run_data,
                "score": score_row,
                "run_path": str(run_output_path),
            }
        )

    aggregate = aggregate_scores(score_rows)
    aggregate["partial_result_count"] = sum(1 for row in score_rows if bool(row.get("partial_result")))
    aggregate["failure_prone_tasks"] = _failure_prone_tasks(score_rows)

    benchmark_results_path = output_root / "benchmark_results.json"
    benchmark_scores_csv_path = output_root / "benchmark_scores.csv"

    benchmark_payload = {
        "provider": provider,
        "model": model,
        "max_iterations": max_iterations,
        "limit": limit,
        "runs": run_rows,
        "scores": score_rows,
        "aggregate": aggregate,
        "output_paths": {
            "results_json": str(benchmark_results_path),
            "scores_csv": str(benchmark_scores_csv_path),
            "runs_dir": str(output_root),
        },
    }

    benchmark_results_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")
    write_scores_csv(score_rows, str(benchmark_scores_csv_path))
    return benchmark_payload


def safe_run_benchmark(
    provider: str,
    model: str,
    max_iterations: int = 6,
    limit: int | None = None,
    output_dir: str = "output/evals",
) -> dict:
    """Run benchmark safely and return structured failure output on errors."""
    try:
        result = run_benchmark(
            provider=provider,
            model=model,
            max_iterations=max_iterations,
            limit=limit,
            output_dir=output_dir,
        )
        result["success"] = True
        result["error"] = None
        return result
    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "runs": [],
            "scores": [],
            "aggregate": {
                "task_count": 0,
                "completed_count": 0,
                "average_total_score": 0.0,
                "average_artifact_score": 0.0,
                "average_iteration_score": 0.0,
                "average_experiment_score": 0.0,
                "best_task": "",
                "worst_task": "",
                "partial_result_count": 0,
                "failure_prone_tasks": [],
            },
            "output_paths": {},
        }
