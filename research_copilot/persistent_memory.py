"""Simple file-based persistent memory and run-history helpers."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LESSONS_PATH = str(PROJECT_ROOT / "output" / "persistent_lessons.json")
DEFAULT_RUN_HISTORY_PATH = str(PROJECT_ROOT / "output" / "agent_run_history.jsonl")


def _read_json_file(filepath: str) -> object | None:
    """Read one JSON file safely and return parsed content."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_file(filepath: str, payload: object) -> None:
    """Write one JSON payload to disk with UTF-8 encoding."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_lessons(limit: int | None = None, filepath: str = DEFAULT_LESSONS_PATH) -> list[dict]:
    """Load persistent lessons, optionally returning only the latest entries."""
    payload = _read_json_file(filepath)
    if not isinstance(payload, list):
        return []

    lessons = [item for item in payload if isinstance(item, dict)]
    if limit is None or limit <= 0:
        return lessons
    return lessons[-limit:]


def append_lesson(lesson: dict, filepath: str = DEFAULT_LESSONS_PATH) -> None:
    """Append one lesson record to persistent lesson storage."""
    lessons = load_lessons(filepath=filepath)
    lessons.append(lesson)
    _write_json_file(filepath, lessons)


def append_run_history(run_payload: dict, filepath: str = DEFAULT_RUN_HISTORY_PATH) -> None:
    """Append one run payload to line-delimited run history."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(run_payload, ensure_ascii=False))
        file_handle.write("\n")


def load_run_history(limit: int | None = 10, filepath: str = DEFAULT_RUN_HISTORY_PATH) -> list[dict]:
    """Load recent run history records from line-delimited JSON."""
    path = Path(filepath)
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)

    if limit is None or limit <= 0:
        return rows
    return rows[-limit:]
