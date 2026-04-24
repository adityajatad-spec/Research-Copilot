"""Deterministic self-repair policy helpers for autonomous runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    from .agent_state import AgentState
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentState


OUTPUT_PATHS = {
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
ACTION_ORDER = ["fetch", "pdf", "summarize", "report", "insights", "gaps", "hypotheses", "experiment", "run_experiment"]
PREREQUISITES = {
    "pdf": "fetch",
    "summarize": "pdf",
    "report": "summarize",
    "insights": "summarize",
    "gaps": "insights",
    "hypotheses": "gaps",
    "experiment": "hypotheses",
    "run_experiment": "experiment",
}
CORE_ARTIFACT_ACTIONS = ["fetch", "summarize", "insights", "gaps", "hypotheses"]


def _failed_count(state: AgentState, action: str) -> int:
    """Return failed count for one action."""
    counts = state.memory.get("failed_action_counts", {})
    if isinstance(counts, dict):
        raw_value = counts.get(action, 0)
        if isinstance(raw_value, int):
            return raw_value
    return sum(1 for item in state.history if item.action == action and item.status == "failed")


def _artifact_exists(state: AgentState, action: str) -> bool:
    """Return whether an artifact for the action is currently available."""
    if action == "run_experiment":
        return bool(state.memory.get("experiment_success"))

    memory_path = state.memory.get(f"{action}_output_path")
    if isinstance(memory_path, str) and Path(memory_path).exists():
        return True

    default_path = OUTPUT_PATHS.get(action)
    if isinstance(default_path, str) and Path(default_path).exists():
        return True
    return False


def _has_useful_partial_results(state: AgentState) -> bool:
    """Return whether enough core artifacts exist for a bounded partial completion."""
    available = sum(1 for action in CORE_ARTIFACT_ACTIONS if _artifact_exists(state, action))
    return available >= 3


def _next_missing_action(state: AgentState) -> str:
    """Return the next missing action in preferred order."""
    for action in ACTION_ORDER:
        if not _artifact_exists(state, action):
            return action
    return "finish"


def _strategy_change_action(action: str) -> str:
    """Return an alternate action when one action is failing repeatedly."""
    if action not in ACTION_ORDER:
        return "finish"

    index = ACTION_ORDER.index(action)
    for candidate in ACTION_ORDER[index + 1 :]:
        if candidate != action:
            return candidate
    return "finish"


def choose_repair_strategy(
    state: AgentState,
    failed_action: str,
    error_text: str,
    failure_info: dict,
) -> dict:
    """Choose a bounded repair strategy based on failure taxonomy and history."""
    category = str(failure_info.get("category", "unknown"))
    repeated_failures = _failed_count(state, failed_action) >= 2
    has_partial = _has_useful_partial_results(state)

    if repeated_failures and has_partial:
        return {
            "strategy": "stop_early",
            "next_action": "finish",
            "notes": "Repeated failures detected and partial evidence is already useful.",
        }

    if category == "provider_unavailable":
        if has_partial:
            return {
                "strategy": "stop_early",
                "next_action": "finish",
                "notes": "Provider unavailable; returning partial result to avoid useless retries.",
            }
        return {
            "strategy": "reroute",
            "next_action": _next_missing_action(state),
            "notes": "Provider unavailable; rerouting to non-LLM or prerequisite steps.",
        }

    if category == "missing_input":
        prerequisite = PREREQUISITES.get(failed_action, _next_missing_action(state))
        return {
            "strategy": "replan",
            "next_action": prerequisite,
            "notes": "Missing input detected; rebuilding prerequisite artifact first.",
        }

    if category == "parse_error":
        if not repeated_failures:
            return {
                "strategy": "modify_prompt",
                "next_action": failed_action,
                "notes": "Parsing failed; retrying once with stricter output guidance.",
            }
        return {
            "strategy": "replan",
            "next_action": _strategy_change_action(failed_action),
            "notes": "Repeated parsing failures; shifting strategy to next pipeline step.",
        }

    if category == "tool_failure":
        return {
            "strategy": "reroute",
            "next_action": _strategy_change_action(failed_action),
            "notes": "Tool compatibility issue detected; using an alternate route.",
        }

    if category == "execution_failure":
        if not repeated_failures:
            return {
                "strategy": "retry_same",
                "next_action": failed_action,
                "notes": "Likely transient execution issue; bounded retry allowed.",
            }
        return {
            "strategy": "reroute",
            "next_action": _strategy_change_action(failed_action),
            "notes": "Execution keeps failing; rerouting instead of repeating.",
        }

    if repeated_failures:
        if has_partial:
            return {
                "strategy": "stop_early",
                "next_action": "finish",
                "notes": "Repeated unknown failures with useful partial outputs available.",
            }
        return {
            "strategy": "replan",
            "next_action": _next_missing_action(state),
            "notes": "Repeated unknown failures; replanning from the next required artifact.",
        }

    _ = error_text
    return {
        "strategy": "retry_same",
        "next_action": failed_action,
        "notes": "Default bounded retry strategy.",
    }


def build_repair_lesson(topic: str, failed_action: str, strategy: dict, failure_info: dict) -> dict:
    """Build a compact repair lesson suitable for persistent memory."""
    strategy_name = str(strategy.get("strategy", "retry_same"))
    category = str(failure_info.get("category", "unknown"))
    reason = str(failure_info.get("reason", "No reason provided."))
    notes = str(strategy.get("notes", ""))
    content = (
        f"For topic '{topic}', action '{failed_action}' failed with '{category}'. "
        f"Applied '{strategy_name}' strategy. {reason} {notes}".strip()
    )
    return {
        "topic": topic,
        "type": "repair_lesson",
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }
