"""In-memory storage helpers for agent state."""

from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    from .agent_state import AgentState
    from .failure_taxonomy import summarize_failure_pattern
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentState
    from failure_taxonomy import summarize_failure_pattern


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


def store_circuit_breaker_state(state: AgentState, action: str, breaker_state: dict) -> None:
    """Store latest circuit breaker state for one action."""
    states = load_memory(state, "circuit_breakers", {})
    if not isinstance(states, dict):
        states = {}
    states[action] = breaker_state
    store_memory(state, "circuit_breakers", states)


def record_repair_decision(state: AgentState, failed_action: str, failure_info: dict, strategy: dict) -> None:
    """Store a structured repair decision with failure taxonomy context."""
    taxonomy = load_memory(state, "failure_taxonomy_by_action", {})
    if not isinstance(taxonomy, dict):
        taxonomy = {}

    action_failures = taxonomy.get(failed_action, [])
    if not isinstance(action_failures, list):
        action_failures = []
    action_failures.append(failure_info)
    taxonomy[failed_action] = action_failures[-10:]
    store_memory(state, "failure_taxonomy_by_action", taxonomy)

    recent_failures = load_memory(state, "recent_failures", [])
    if not isinstance(recent_failures, list):
        recent_failures = []
    recent_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": failed_action,
        **failure_info,
    }
    recent_failures.append(recent_event)
    recent_failures = recent_failures[-20:]
    store_memory(state, "recent_failures", recent_failures)
    store_memory(state, "recent_failure_summary", summarize_failure_pattern(recent_failures))

    repair_history = load_memory(state, "repair_history", [])
    if not isinstance(repair_history, list):
        repair_history = []
    decision = {
        "timestamp": datetime.utcnow().isoformat(),
        "failed_action": failed_action,
        "failure_info": failure_info,
        "strategy": strategy,
    }
    repair_history.append(decision)
    store_memory(state, "repair_history", repair_history[-20:])
    store_memory(state, "last_repair_decision", decision)
    store_memory(state, "last_repair_strategy", strategy.get("strategy"))

    if str(strategy.get("strategy")) == "stop_early":
        store_memory(state, "partial_result", True)
        store_memory(state, "stop_early_requested", True)

    if str(strategy.get("strategy")) in {"reroute", "replan"}:
        known_bad = load_memory(state, "known_bad_actions", {})
        if not isinstance(known_bad, dict):
            known_bad = {}
        known_bad[failed_action] = {
            "reason": failure_info.get("reason", ""),
            "category": failure_info.get("category", "unknown"),
        }
        store_memory(state, "known_bad_actions", known_bad)


def record_planner_snapshot(
    state: AgentState,
    planned_action: str,
    artifacts: dict[str, bool],
    skipped_reasons: list[str],
) -> None:
    """Store planner artifact inspection and progression rationale in memory."""
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "planned_action": planned_action,
        "artifacts": artifacts,
        "skipped_reasons": skipped_reasons,
    }

    history = load_memory(state, "planner_snapshots", [])
    if not isinstance(history, list):
        history = []
    history.append(snapshot)
    store_memory(state, "planner_snapshots", history[-25:])

    store_memory(state, "planner_artifacts", artifacts)
    store_memory(state, "planner_skipped_reasons", skipped_reasons[-10:])
