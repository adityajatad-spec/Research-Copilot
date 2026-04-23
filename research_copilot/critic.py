"""Critic utilities for evaluating autonomous agent progress."""

from __future__ import annotations

from pathlib import Path

try:
    from .agent_state import AgentState
    from .memory_store import summarize_memory
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentState
    from memory_store import summarize_memory


DEFAULT_OUTPUT_PATHS = {
    "fetch": "output/results.json",
    "pdf": "output/papers_with_pdf.json",
    "summarize": "output/summaries.json",
    "report": "output/report.md",
    "insights": "output/insights.json",
    "gaps": "output/gaps.json",
    "hypotheses": "output/hypotheses.json",
    "experiment": "output/experiment.py",
    "run_experiment": "output/experiment_run/results.json",
}
ACTION_ORDER = ["fetch", "pdf", "summarize", "report", "insights", "gaps", "hypotheses", "experiment"]
MAX_CONSECUTIVE_FAILURES = 3


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


def _artifact_exists(state: AgentState, action: str) -> bool:
    """Check whether a required artifact exists for the current run."""
    if action == "run_experiment":
        return bool(state.memory.get("experiment_success"))

    if action not in DEFAULT_OUTPUT_PATHS:
        return False

    memory_key = f"{action}_output_path"
    memory_path = state.memory.get(memory_key)
    if isinstance(memory_path, str) and Path(memory_path).exists():
        return True

    completed_in_run = any(
        item.action == action and item.status == "completed"
        for item in state.history
    )
    if completed_in_run and Path(DEFAULT_OUTPUT_PATHS[action]).exists():
        return True

    return False


def _next_missing_action(state: AgentState) -> str | None:
    """Return the next missing action, if any."""
    for action in ACTION_ORDER:
        if not _artifact_exists(state, action):
            return action
    return None


def _experiment_has_run(state: AgentState) -> bool:
    """Return whether experiment execution has been attempted in this run."""
    if bool(state.memory.get("experiment_has_run")):
        return True
    return any(item.action == "run_experiment" for item in state.history)


def generate_critic_prompt(state: AgentState) -> tuple[str, str]:
    """Generate critic system and user prompts with JSON-only instructions."""
    system_prompt = (
        "You are a critic module for an autonomous research pipeline.\n"
        "Return ONLY JSON with keys: status, reason.\n"
        "status must be either 'continue' or 'done'.\n"
        "Use experiment execution signals when available and avoid unbounded retries."
    )
    user_prompt = (
        f"Topic: {state.topic}\n"
        f"Iteration: {state.iteration}/{state.max_iterations}\n"
        f"Done flag: {state.done}\n"
        f"Last result: {state.last_result}\n"
        f"Experiment success: {state.memory.get('experiment_success')}\n"
        f"Experiment has run: {state.memory.get('experiment_has_run')}\n"
        f"Experiment summary: {state.memory.get('experiment_signal_summary')}\n"
        f"Memory summary: {summarize_memory(state)}\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


def fallback_critic(state: AgentState) -> dict:
    """Return a conservative critic decision."""
    if state.iteration >= state.max_iterations:
        return {"status": "done", "reason": "Maximum iterations reached."}

    if _artifact_exists(state, "run_experiment"):
        return {"status": "done", "reason": "Experiment execution succeeded."}

    if _artifact_exists(state, "experiment") and not _experiment_has_run(state):
        return {"status": "continue", "reason": "Experiment script exists but has not been executed yet."}

    return {"status": "continue", "reason": "Pipeline is not complete yet."}


def evaluate_state(state: AgentState) -> dict:
    """Evaluate whether the agent should continue or stop."""
    if state.iteration >= state.max_iterations:
        return {"status": "done", "reason": "Maximum iterations reached."}

    if _consecutive_failures(state) >= MAX_CONSECUTIVE_FAILURES:
        return {
            "status": "done",
            "reason": f"Stopping after {MAX_CONSECUTIVE_FAILURES} consecutive failed actions.",
        }

    if _artifact_exists(state, "run_experiment"):
        return {"status": "done", "reason": "Experiment execution succeeded with usable evidence."}

    experiment_script_exists = _artifact_exists(state, "experiment")
    experiment_has_run = _experiment_has_run(state)
    run_failures = _failed_action_count(state, "run_experiment")
    experiment_success = bool(state.memory.get("experiment_success"))

    if experiment_script_exists and not experiment_has_run:
        return {"status": "continue", "reason": "Experiment script exists but has not been executed yet."}

    if experiment_script_exists and experiment_has_run and not experiment_success:
        if run_failures >= 2:
            return {
                "status": "done",
                "reason": "Experiment execution failed multiple times; analytical outputs are already available.",
            }
        return {
            "status": "continue",
            "reason": "Experiment execution failed; one bounded retry is allowed.",
        }

    missing_action = _next_missing_action(state)
    if missing_action is None:
        return {"status": "done", "reason": "All expected artifacts are present."}

    if state.history:
        last_action = state.history[-1]
        if last_action.status == "failed":
            return {
                "status": "continue",
                "reason": f"Last action '{last_action.action}' failed; retry with a different strategy.",
            }

    return {"status": "continue", "reason": f"Missing required step: {missing_action}."}
