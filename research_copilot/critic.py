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
}
ACTION_ORDER = ["fetch", "pdf", "summarize", "report", "insights", "gaps", "hypotheses", "experiment"]


def _artifact_exists(state: AgentState, action: str) -> bool:
    """Check whether a required artifact exists for the current run."""
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


def generate_critic_prompt(state: AgentState) -> tuple[str, str]:
    """Generate critic system and user prompts with JSON-only instructions."""
    system_prompt = (
        "You are a critic module for an autonomous research pipeline.\n"
        "Return ONLY JSON with keys: status, reason.\n"
        "status must be either 'continue' or 'done'.\n"
        "Be conservative and ensure the pipeline has enough evidence before 'done'."
    )
    user_prompt = (
        f"Topic: {state.topic}\n"
        f"Iteration: {state.iteration}/{state.max_iterations}\n"
        f"Done flag: {state.done}\n"
        f"Last result: {state.last_result}\n"
        f"Memory summary: {summarize_memory(state)}\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


def fallback_critic(state: AgentState) -> dict:
    """Return a conservative critic decision."""
    if state.iteration >= state.max_iterations:
        return {"status": "done", "reason": "Maximum iterations reached."}

    if _artifact_exists(state, "experiment"):
        return {"status": "done", "reason": "Experiment scaffold exists."}

    return {"status": "continue", "reason": "Pipeline is not complete yet."}


def evaluate_state(state: AgentState) -> dict:
    """Evaluate whether the agent should continue or stop."""
    if state.iteration >= state.max_iterations:
        return {"status": "done", "reason": "Maximum iterations reached."}

    if _artifact_exists(state, "experiment"):
        return {"status": "done", "reason": "Experiment scaffold generated successfully."}

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
