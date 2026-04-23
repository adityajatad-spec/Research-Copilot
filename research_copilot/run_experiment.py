"""Helpers for safely running generated experiment scripts locally."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_python_experiment(
    script_path: str,
    dataset: str = "demo-dataset",
    output_dir: str = "output/experiment_run",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 42,
    timeout: int = 120,
) -> dict:
    """Run a generated Python experiment script and capture structured outputs."""
    script_file = Path(script_path)
    if not script_file.exists():
        raise ValueError(f"Experiment script not found: {script_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        "python3",
        str(script_file),
        "--dataset",
        str(dataset),
        "--output_dir",
        str(output_path),
        "--epochs",
        str(epochs),
        "--learning_rate",
        str(learning_rate),
        "--seed",
        str(seed),
    ]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        results_file = output_path / "results.json"
        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout or "",
            "stderr": process.stderr or "",
            "command": " ".join(command),
            "script_path": str(script_file),
            "output_dir": str(output_path),
            "results_path": str(results_file) if results_file.exists() else None,
            "timeout": timeout,
            "error": None,
        }
    except subprocess.TimeoutExpired as error:
        partial_stdout = error.stdout if isinstance(error.stdout, str) else ""
        partial_stderr = error.stderr if isinstance(error.stderr, str) else ""
        return {
            "success": False,
            "returncode": None,
            "stdout": partial_stdout,
            "stderr": partial_stderr,
            "command": " ".join(command),
            "script_path": str(script_file),
            "output_dir": str(output_path),
            "results_path": None,
            "timeout": timeout,
            "error": "timeout",
        }


def safe_run_experiment(
    script_path: str,
    dataset: str = "demo-dataset",
    output_dir: str = "output/experiment_run",
    epochs: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 42,
    timeout: int = 120,
) -> dict:
    """Run an experiment safely and always return a structured result dictionary."""
    try:
        return run_python_experiment(
            script_path=script_path,
            dataset=dataset,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            timeout=timeout,
        )
    except Exception as error:
        return {
            "success": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "command": "",
            "script_path": script_path,
            "output_dir": output_dir,
            "results_path": None,
            "timeout": timeout,
            "error": str(error),
        }
