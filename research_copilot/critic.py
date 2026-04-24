"""Critic utilities for evaluating autonomous agent progress."""

from __future__ import annotations

from typing import Any

try:
    from .agent_state import AgentState
    from .memory_store import summarize_memory
    from .planner import inspect_artifacts
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentState
    from memory_store import summarize_memory
    from planner import inspect_artifacts


ACTION_ORDER = ["fetch", "pdf", "summarize", "report", "insights", "gaps", "hypotheses", "experiment", "run_experiment"]
CRITICAL_ACTIONS = ["fetch", "summarize", "insights", "gaps", "hypotheses"]
MAX_CONSECUTIVE_FAILURES = 3
REPEATED_FAILURE_THRESHOLD = 4


def _failed_action_count(state: AgentState, action: str) -> int:
    """Return failed attempts for one action from memory/history."""
    counts = state.memory.get("failed_action_counts", {})
    if isinstance(counts, dict):
        value = counts.get(action, 0)
        if isinstance(value, int):
            return value
    return sum(1 for item in state.history if item.action == action and item.status == "failed")


def _consecutive_failures(state: AgentState) -> int:
    """Count how many failed actions appear consecutively at the end of history."""
    count = 0
    for item in reversed(state.history):
        if item.status != "failed":
            break
        count += 1
    return count


def _total_failure_count(state: AgentState) -> int:
    """Count failed actions across the whole run history."""
    return sum(1 for item in state.history if item.status == "failed")


def _experiment_has_run(state: AgentState) -> bool:
    """Return whether experiment execution has been attempted in this run."""
    if bool(state.memory.get("experiment_has_run")):
        return True
    return any(item.action == "run_experiment" for item in state.history)


def _artifact_status_map(inspection: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Convert inspection payload into a compact readiness map."""
    status: dict[str, bool] = {}
    for action in ACTION_ORDER:
        info = inspection.get(action, {})
        status[action] = bool(info.get("ready", False)) if isinstance(info, dict) else False
    return status


def _compute_completed_steps(status: dict[str, bool]) -> tuple[list[str], set[str]]:
    """Infer completed steps; downstream artifacts imply upstream completion."""
    completed_set: set[str] = set()
    for index, action in enumerate(ACTION_ORDER):
        if status.get(action, False):
            completed_set.update(ACTION_ORDER[: index + 1])

    completed_steps = [action for action in ACTION_ORDER if action in completed_set]
    return completed_steps, completed_set


def _next_missing_step(state: AgentState, completed_set: set[str], status: dict[str, bool]) -> str | None:
    """Return next missing step under progression-aware rules."""
    main_order = ACTION_ORDER[:-1]  # exclude run_experiment for primary progression
    for action in main_order:
        if action not in completed_set:
            return action

    experiment_ready = status.get("experiment", False)
    run_ready = status.get("run_experiment", False)
    if experiment_ready and not run_ready and not _experiment_has_run(state):
        return "run_experiment"
    return None


def _build_critic_snapshot(state: AgentState) -> dict[str, Any]:
    """Build current artifact and progression state for critic decisions."""
    inspection = inspect_artifacts(state)
    status = _artifact_status_map(inspection)
    completed_steps, completed_set = _compute_completed_steps(status)
    next_missing = _next_missing_step(state, completed_set, status)
    return {
        "inspection": inspection,
        "status": status,
        "completed_steps": completed_steps,
        "completed_set": completed_set,
        "next_missing_step": next_missing,
    }


def _has_useful_partial_artifacts(snapshot: dict[str, Any]) -> bool:
    """Return whether enough critical artifacts exist to allow bounded completion."""
    completed_set = snapshot.get("completed_set", set())
    if not isinstance(completed_set, set):
        return False
    available = sum(1 for action in CRITICAL_ACTIONS if action in completed_set)
    return available >= 3


def _circuit_open_for(state: AgentState, action: str) -> bool:
    """Return whether circuit breaker is open for one action."""
    breakers = state.memory.get("circuit_breakers", {})
    if not isinstance(breakers, dict):
        return False
    row = breakers.get(action, {})
    if not isinstance(row, dict):
        return False
    return str(row.get("state", "closed")).strip().lower() == "open"


def _critical_path_blocked(state: AgentState, snapshot: dict[str, Any]) -> bool:
    """Return whether next required step is blocked by an open circuit."""
    missing_action = snapshot.get("next_missing_step")
    if not isinstance(missing_action, str) or not missing_action:
        return False
    return _circuit_open_for(state, missing_action)


def _has_unblocked_missing_action(state: AgentState, snapshot: dict[str, Any]) -> bool:
    """Return whether any missing step can still execute."""
    completed_set = snapshot.get("completed_set", set())
    if not isinstance(completed_set, set):
        return False

    for action in ACTION_ORDER:
        if action in completed_set:
            continue
        if action == "run_experiment" and _experiment_has_run(state):
            continue
        if not _circuit_open_for(state, action):
            return True
    return False


def _format_decision(snapshot: dict[str, Any], status: str, reason: str) -> dict:
    """Return critic decision with debug-friendly artifact context."""
    detected = snapshot.get("status", {})
    completed_steps = snapshot.get("completed_steps", [])
    next_missing = snapshot.get("next_missing_step")
    return {
        "status": status,
        "reason": reason,
        "detected_artifacts": detected if isinstance(detected, dict) else {},
        "completed_steps": completed_steps if isinstance(completed_steps, list) else [],
        "next_missing_step": next_missing if isinstance(next_missing, str) else None,
    }


def generate_critic_prompt(state: AgentState) -> tuple[str, str]:
    """Generate critic system and user prompts with JSON-only instructions."""
    snapshot = _build_critic_snapshot(state)
    system_prompt = (
        "You are a critic module for an autonomous research pipeline.\n"
        "Return ONLY JSON with keys: status, reason.\n"
        "status must be either 'continue' or 'done'.\n"
        "Use real artifact state, repeated-failure signals, and circuit breaker hints."
    )
    user_prompt = (
        f"Topic: {state.topic}\n"
        f"Iteration: {state.iteration}/{state.max_iterations}\n"
        f"Done flag: {state.done}\n"
        f"Last result: {state.last_result}\n"
        f"Detected artifacts: {snapshot.get('status')}\n"
        f"Completed steps: {snapshot.get('completed_steps')}\n"
        f"Next missing step: {snapshot.get('next_missing_step')}\n"
        f"Recent failure summary: {state.memory.get('recent_failure_summary')}\n"
        f"Last repair strategy: {state.memory.get('last_repair_strategy')}\n"
        f"Circuit breakers: {state.memory.get('circuit_breakers')}\n"
        f"Memory summary: {summarize_memory(state)}\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


def fallback_critic(state: AgentState) -> dict:
    """Return a conservative critic decision."""
    snapshot = _build_critic_snapshot(state)

    if state.iteration >= state.max_iterations:
        return _format_decision(snapshot, "done", "Maximum iterations reached.")

    if bool(state.memory.get("stop_early_requested")):
        return _format_decision(snapshot, "done", "Repair policy requested bounded early stop.")

    if snapshot.get("status", {}).get("run_experiment", False):
        return _format_decision(snapshot, "done", "Experiment execution succeeded.")

    next_missing = snapshot.get("next_missing_step")
    if isinstance(next_missing, str):
        return _format_decision(snapshot, "continue", f"Missing required step: {next_missing}.")

    return _format_decision(snapshot, "done", "All expected artifacts are present.")


def evaluate_state(state: AgentState) -> dict:
    """Evaluate whether the agent should continue or stop using current artifact state."""
    snapshot = _build_critic_snapshot(state)

    if state.iteration >= state.max_iterations:
        return _format_decision(snapshot, "done", "Maximum iterations reached.")

    if bool(state.memory.get("stop_early_requested")):
        return _format_decision(snapshot, "done", "Repair strategy requested early stop with partial result.")

    if snapshot.get("status", {}).get("run_experiment", False):
        return _format_decision(snapshot, "done", "Experiment execution succeeded with usable evidence.")

    if _consecutive_failures(state) >= MAX_CONSECUTIVE_FAILURES:
        if _has_useful_partial_artifacts(snapshot):
            return _format_decision(
                snapshot,
                "done",
                "Stopping after repeated failures because useful partial artifacts already exist.",
            )
        return _format_decision(snapshot, "continue", "Consecutive failures detected; continue with corrective strategy.")

    if _total_failure_count(state) >= REPEATED_FAILURE_THRESHOLD and _has_useful_partial_artifacts(snapshot):
        return _format_decision(snapshot, "done", "Repeated failures exceeded threshold; returning bounded partial result.")

    experiment_script_exists = bool(snapshot.get("status", {}).get("experiment", False))
    experiment_has_run = _experiment_has_run(state)
    run_failures = _failed_action_count(state, "run_experiment")
    experiment_success = bool(state.memory.get("experiment_success"))

    if experiment_script_exists and not experiment_has_run and snapshot.get("next_missing_step") == "run_experiment":
        return _format_decision(snapshot, "continue", "Experiment script exists but has not been executed yet.")

    if experiment_script_exists and experiment_has_run and not experiment_success:
        if run_failures >= 2:
            if _has_useful_partial_artifacts(snapshot):
                return _format_decision(
                    snapshot,
                    "done",
                    "Experiment execution failed multiple times; analytical outputs are already available.",
                )
            return _format_decision(
                snapshot,
                "continue",
                "Experiment execution failed repeatedly; continue only if critical outputs are still missing.",
            )
        return _format_decision(snapshot, "continue", "Experiment execution failed; one bounded retry is allowed.")

    next_missing = snapshot.get("next_missing_step")
    if next_missing is None:
        return _format_decision(snapshot, "done", "All expected artifacts are present.")

    if _critical_path_blocked(state, snapshot):
        if not _has_unblocked_missing_action(state, snapshot):
            if _has_useful_partial_artifacts(snapshot):
                return _format_decision(
                    snapshot,
                    "done",
                    "No unblocked remaining actions; returning bounded partial result.",
                )
            return _format_decision(
                snapshot,
                "done",
                "No unblocked remaining actions due to open circuit breakers.",
            )
        if _has_useful_partial_artifacts(snapshot):
            return _format_decision(
                snapshot,
                "done",
                "Critical next action is blocked by open circuit; returning bounded partial result.",
            )
        return _format_decision(
            snapshot,
            "continue",
            "Critical action is blocked; wait for alternate plan or circuit recovery.",
        )

    if state.history:
        last_action = state.history[-1]
        if last_action.status == "failed":
            return _format_decision(
                snapshot,
                "continue",
                f"Last action '{last_action.action}' failed; retry with repair strategy.",
            )

    return _format_decision(snapshot, "continue", f"Missing required step: {next_missing}.")
