"""Planning utilities for the autonomous research agent."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .agent_state import AgentState
    from .config import Config, get_client, validate_provider_setup
    from .memory_store import summarize_memory
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentState
    from config import Config, get_client, validate_provider_setup
    from memory_store import summarize_memory


PLANNER_ACTIONS = [
    "fetch",
    "pdf",
    "summarize",
    "report",
    "insights",
    "gaps",
    "hypotheses",
    "experiment",
    "run_experiment",
    "finish",
]
ACTION_ORDER = ["fetch", "pdf", "summarize", "report", "insights", "gaps", "hypotheses", "experiment", "run_experiment"]
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


def _failed_counts(state: AgentState) -> dict[str, int]:
    """Count failed actions from state history."""
    counts: dict[str, int] = {}
    for item in state.history:
        if item.status != "failed":
            continue
        counts[item.action] = counts.get(item.action, 0) + 1
    return counts


def _artifact_exists(state: AgentState, action: str) -> bool:
    """Check if an action artifact exists for the current run."""
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


def _next_missing_action(state: AgentState) -> str:
    """Return the next missing action in preferred pipeline order."""
    for action in ACTION_ORDER:
        if not _artifact_exists(state, action):
            return action
    return "finish"


def _default_input_for(action: str, state: AgentState) -> str:
    """Return default input text for a given action."""
    if action == "fetch":
        return f"topic={state.topic} source=hybrid max=12"
    if action == "pdf":
        return "input=output/results.json output=output/papers_with_pdf.json"
    if action == "summarize":
        return "input=output/papers_with_pdf.json output=output/summaries.json"
    if action == "report":
        return f"input=output/summaries.json output=output/report.md topic={state.topic}"
    if action == "insights":
        return f"input=output/summaries.json output=output/insights.json topic={state.topic}"
    if action == "gaps":
        return f"input=output/summaries.json output=output/gaps.json topic={state.topic}"
    if action == "hypotheses":
        return (
            "papers=output/papers_with_pdf.json insights=output/insights.json "
            f"gaps=output/gaps.json output=output/hypotheses.json topic={state.topic}"
        )
    if action == "experiment":
        return f"hypotheses=output/hypotheses.json output=output/experiment.py topic={state.topic}"
    if action == "run_experiment":
        return (
            "script=output/experiment.py dataset=demo-dataset output_dir=output/experiment_run "
            "epochs=1 learning_rate=1e-4 seed=42 timeout=120"
        )
    return ""


def _strategy_change_action(action: str) -> str:
    """Choose an alternate action when repeated failures occur."""
    if action == "fetch":
        return "pdf"
    if action == "pdf":
        return "summarize"
    if action == "summarize":
        return "report"
    if action == "report":
        return "insights"
    if action == "insights":
        return "gaps"
    if action == "gaps":
        return "hypotheses"
    if action == "hypotheses":
        return "experiment"
    if action == "experiment":
        return "run_experiment"
    if action == "run_experiment":
        return "finish"
    return "finish"


def _apply_failure_guard(plan: dict, state: AgentState) -> dict:
    """Adjust planner output when repeated failures suggest a strategy change."""
    failed_counts = _failed_counts(state)
    action = str(plan.get("action", "")).strip()
    if action not in PLANNER_ACTIONS:
        return fallback_plan(state)

    last_action = state.history[-1] if state.history else None
    failed_twice = failed_counts.get(action, 0) >= 2
    immediate_repeat_failure = bool(
        last_action and last_action.action == action and last_action.status == "failed" and failed_counts.get(action, 0) >= 2
    )

    if failed_twice or immediate_repeat_failure:
        alternative = _strategy_change_action(action)
        return {
            "thought": f"Changing strategy because '{action}' failed repeatedly.",
            "action": alternative,
            "input": _default_input_for(alternative, state),
        }

    return plan


def generate_planner_prompt(state: AgentState) -> tuple[str, str]:
    """Generate planner system and user prompts with JSON-only instructions."""
    system_prompt = (
        "You are a planning module for an autonomous research pipeline.\n"
        "Return ONLY valid JSON with keys: thought, action, input.\n"
        "Allowed actions: fetch, pdf, summarize, report, insights, gaps, hypotheses, experiment, run_experiment, finish.\n"
        "Choose the minimum next step needed. Avoid repeating failed actions."
    )

    history_lines = [
        f"- step={item.step} action={item.action} status={item.status} reason={item.reason}" for item in state.history[-8:]
    ]
    history_text = "\n".join(history_lines) if history_lines else "- none"

    user_prompt = (
        f"Topic: {state.topic}\n"
        f"Iteration: {state.iteration}/{state.max_iterations}\n"
        f"Current goal: {state.current_goal}\n"
        f"Recent history:\n{history_text}\n"
        f"Memory summary: {summarize_memory(state)}\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


def fallback_plan(state: AgentState) -> dict:
    """Return a safe deterministic plan when planner output is unavailable."""
    next_action = _next_missing_action(state)
    if next_action == "finish":
        return {"thought": "All expected artifacts are present.", "action": "finish", "input": ""}

    failed_counts = _failed_counts(state)
    if failed_counts.get(next_action, 0) >= 2:
        alternative = _strategy_change_action(next_action)
        return {
            "thought": f"Switching from '{next_action}' after repeated failures.",
            "action": alternative,
            "input": _default_input_for(alternative, state),
        }

    return {
        "thought": f"Next missing artifact requires '{next_action}'.",
        "action": next_action,
        "input": _default_input_for(next_action, state),
    }


def plan_next_step(state: AgentState, config: Config) -> dict:
    """Plan the next agent action using LLM output with a deterministic fallback."""
    baseline_plan = fallback_plan(state)

    try:
        validate_provider_setup(config)
        client = get_client(config)
        system_prompt, user_prompt = generate_planner_prompt(state)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        action = str(parsed.get("action", "")).strip().lower()
        thought = str(parsed.get("thought", "")).strip() or baseline_plan["thought"]
        input_text = str(parsed.get("input", "")).strip() or _default_input_for(action, state)

        if action not in PLANNER_ACTIONS:
            return baseline_plan

        candidate = {"thought": thought, "action": action, "input": input_text}
        return _apply_failure_guard(candidate, state)
    except Exception:
        return baseline_plan
