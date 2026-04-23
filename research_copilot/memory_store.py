"""In-memory storage helpers for agent state."""

from __future__ import annotations

from typing import Any

try:
    from .agent_state import AgentState
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentState


def store_memory(state: AgentState, key: str, value: Any) -> None:
    """Store a key-value pair in agent memory."""
    state.memory[key] = value


def load_memory(state: AgentState, key: str, default: Any = None) -> Any:
    """Load a value from agent memory with a default fallback."""
    return state.memory.get(key, default)


def summarize_memory(state: AgentState) -> str:
    """Build a short human-readable summary of memory contents."""
    if not state.memory:
        return "No memory stored yet."

    lines: list[str] = []
    for key, value in state.memory.items():
        if isinstance(value, list):
            lines.append(f"{key}: {len(value)} item(s)")
        elif isinstance(value, dict):
            lines.append(f"{key}: {len(value)} key(s)")
        else:
            lines.append(f"{key}: {value}")
    return "; ".join(lines)


def _tail_text(value: object, limit: int = 500) -> str:
    """Return the trailing text segment for compact memory storage."""
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]


def record_experiment_memory(state: AgentState, run_result: dict, parsed_results: dict | None) -> None:
    """Store structured experiment execution evidence in agent memory."""
    try:
        from .result_parser import extract_result_signals, summarize_experiment_result
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from result_parser import extract_result_signals, summarize_experiment_result

    signals = extract_result_signals(parsed_results)
    summary = summarize_experiment_result(run_result, parsed_results)

    store_memory(state, "experiment_success", bool(run_result.get("success")))
    store_memory(state, "experiment_has_run", True)
    store_memory(state, "experiment_results_path", run_result.get("results_path"))
    store_memory(state, "experiment_signal_summary", summary)
    store_memory(state, "experiment_stdout_tail", _tail_text(run_result.get("stdout")))
    store_memory(state, "experiment_stderr_tail", _tail_text(run_result.get("stderr")))
    store_memory(state, "experiment_metric_keys", signals.get("metric_keys", []))
    store_memory(state, "experiment_last_run", run_result)

    if run_result.get("results_path"):
        store_memory(state, "run_experiment_output_path", run_result.get("results_path"))
