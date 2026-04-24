"""Planning utilities for the autonomous research agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from .agent_state import AgentState
    from .config import Config, get_client, validate_provider_setup
    from .memory_store import get_last_successful_action, load_memory, summarize_memory
    from .persistent_memory import load_lessons
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentState
    from config import Config, get_client, validate_provider_setup
    from memory_store import get_last_successful_action, load_memory, summarize_memory
    from persistent_memory import load_lessons


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
PIPELINE_ORDER = ["fetch", "pdf", "summarize", "report", "insights", "gaps", "hypotheses", "experiment"]
PLANNER_LLM_TIMEOUT_SECONDS = 20


def _failed_counts(state: AgentState) -> dict[str, int]:
    """Count failed actions from state history."""
    counts: dict[str, int] = {}
    for item in state.history:
        if item.status != "failed":
            continue
        counts[item.action] = counts.get(item.action, 0) + 1
    return counts


def _safe_load_json(path: Path) -> object | None:
    """Load JSON from disk safely."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _artifact_candidates(state: AgentState, action: str) -> list[Path]:
    """Return candidate file paths for one artifact action."""
    candidates: list[Path] = []

    memory_path = load_memory(state, f"{action}_output_path")
    if isinstance(memory_path, str) and memory_path.strip():
        candidates.append(Path(memory_path))

    if action == "experiment":
        for key in ("experiment_script_path", "experiment_output_path"):
            value = load_memory(state, key)
            if isinstance(value, str) and value.strip():
                candidates.append(Path(value))

    if action == "run_experiment":
        for key in ("experiment_results_path", "run_experiment_output_path"):
            value = load_memory(state, key)
            if isinstance(value, str) and value.strip():
                candidates.append(Path(value))

    default_path = DEFAULT_OUTPUT_PATHS.get(action)
    if isinstance(default_path, str):
        candidates.append(Path(default_path))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _validate_artifact_file(action: str, path: Path) -> tuple[bool, str]:
    """Validate one artifact file for planner progression checks."""
    if not path.exists():
        return False, "file missing"

    if path.stat().st_size <= 0:
        return False, "file is empty"

    if action in {"report", "experiment"}:
        return True, "text artifact exists"

    payload = _safe_load_json(path)
    if payload is None:
        return False, "invalid JSON"

    if action in {"fetch", "pdf", "summarize"}:
        if isinstance(payload, list) and len(payload) > 0:
            return True, f"JSON list with {len(payload)} item(s)"
        return False, "JSON list missing or empty"

    if action in {"insights", "gaps", "hypotheses", "run_experiment"}:
        if isinstance(payload, dict) and len(payload) > 0:
            return True, f"JSON object with {len(payload)} key(s)"
        if isinstance(payload, list) and len(payload) > 0:
            return True, f"JSON list with {len(payload)} item(s)"
        return False, "JSON payload empty or unexpected shape"

    return True, "artifact exists"


def _inspect_artifacts(state: AgentState) -> dict[str, dict[str, Any]]:
    """Inspect all known artifacts from memory and disk."""
    inspection: dict[str, dict[str, Any]] = {}

    for action in ACTION_ORDER:
        info: dict[str, Any] = {
            "ready": False,
            "path": "",
            "source": "none",
            "detail": "missing",
        }
        last_detail = "file missing"
        for candidate in _artifact_candidates(state, action):
            ready, detail = _validate_artifact_file(action, candidate)
            if ready:
                info = {
                    "ready": True,
                    "path": str(candidate),
                    "source": "disk",
                    "detail": detail,
                }
                break
            last_detail = detail

        if not info["ready"] and action == "fetch":
            paper_count = load_memory(state, "fetched_paper_count", 0)
            if isinstance(paper_count, int) and paper_count > 0:
                info = {
                    "ready": True,
                    "path": "",
                    "source": "memory",
                    "detail": f"memory has fetched_paper_count={paper_count}",
                }

        if not info["ready"] and action == "summarize":
            summary_count = load_memory(state, "summary_count", 0)
            if isinstance(summary_count, int) and summary_count > 0:
                info = {
                    "ready": True,
                    "path": "",
                    "source": "memory",
                    "detail": f"memory has summary_count={summary_count}",
                }

        if not info["ready"] and action == "run_experiment":
            if bool(load_memory(state, "experiment_success", False)):
                info = {
                    "ready": True,
                    "path": "",
                    "source": "memory",
                    "detail": "memory indicates experiment_success=True",
                }

        if not info["ready"]:
            info["detail"] = last_detail

        inspection[action] = info

    return inspection


def inspect_artifacts(state: AgentState) -> dict[str, dict[str, Any]]:
    """Return artifact inspection for external modules such as critic."""
    return _inspect_artifacts(state)


def _artifact_bool_map(inspection: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Return a compact artifact readiness map."""
    return {action: bool(data.get("ready", False)) for action, data in inspection.items()}


def _artifact_debug_labels(inspection: dict[str, dict[str, Any]]) -> list[str]:
    """Return short labels for ready artifacts."""
    labels: list[str] = []
    for action in ACTION_ORDER:
        info = inspection.get(action, {})
        if not bool(info.get("ready", False)):
            continue
        source = str(info.get("source", "none"))
        labels.append(f"{action}({source})")
    return labels


def _circuit_state(state: AgentState, action: str) -> str:
    """Return circuit breaker state for one action."""
    breakers = load_memory(state, "circuit_breakers", {})
    if isinstance(breakers, dict):
        row = breakers.get(action, {})
        if isinstance(row, dict):
            value = str(row.get("state", "closed")).strip().lower()
            if value in {"closed", "open", "half_open"}:
                return value
    return "closed"


def _action_is_blocked(state: AgentState, action: str) -> bool:
    """Return whether planner should avoid an action right now."""
    if _circuit_state(state, action) == "open":
        return True

    known_bad = load_memory(state, "known_bad_actions", {})
    if not isinstance(known_bad, dict):
        return False
    if action not in known_bad:
        return False

    return _circuit_state(state, action) == "open"


def _allow_retry_same(state: AgentState, action: str) -> bool:
    """Return whether retry_same explicitly allows repeating an action."""
    if not state.history:
        return False
    last_item = state.history[-1]
    if last_item.action != action or last_item.status != "failed":
        return False

    decision = load_memory(state, "last_repair_decision", {})
    if not isinstance(decision, dict):
        return False
    strategy = decision.get("strategy", {})
    if not isinstance(strategy, dict):
        return False
    return str(strategy.get("strategy", "")).strip() == "retry_same"


def _last_successful_action(state: AgentState) -> str | None:
    """Return the latest successful pipeline action."""
    action = get_last_successful_action(state)
    if isinstance(action, str) and action in ACTION_ORDER:
        return action
    return None


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


def _next_progression_action(state: AgentState, inspection: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Select next action from strict pipeline progression."""
    skipped_reasons: list[str] = []
    selected_action: str | None = None

    for action in PIPELINE_ORDER:
        info = inspection.get(action, {})
        if bool(info.get("ready", False)):
            skipped_reasons.append(f"Skipped {action}: artifact ready ({info.get('detail', 'ready')}).")
            continue
        selected_action = action
        skipped_reasons.append(f"Selected {action}: artifact missing/corrupt ({info.get('detail', 'missing')}).")
        break

    if selected_action is None:
        experiment_ready = bool(inspection.get("experiment", {}).get("ready", False))
        run_ready = bool(inspection.get("run_experiment", {}).get("ready", False))
        experiment_has_run = bool(load_memory(state, "experiment_has_run", False))

        if experiment_ready and not experiment_has_run and not run_ready:
            selected_action = "run_experiment"
            skipped_reasons.append("Selected run_experiment: experiment script exists and has not been run.")
        elif run_ready:
            selected_action = "finish"
            skipped_reasons.append("Selected finish: experiment results artifact already exists.")
        elif experiment_ready and experiment_has_run and not run_ready:
            selected_action = "finish"
            skipped_reasons.append("Selected finish: experiment already ran without a new results artifact.")
        else:
            selected_action = "finish"
            skipped_reasons.append("Selected finish: no missing progression artifacts.")

    if selected_action != "finish" and _action_is_blocked(state, selected_action):
        skipped_reasons.append(f"Skipped {selected_action}: circuit breaker is open.")
        selected_action = "finish"
        skipped_reasons.append("Selected finish: no safe unblocked progression step available.")

    return selected_action, skipped_reasons


def _build_progression_plan(state: AgentState, inspection: dict[str, dict[str, Any]]) -> dict:
    """Build deterministic progression plan with debug context."""
    action, skipped = _next_progression_action(state, inspection)
    thought = f"Pipeline progression selected '{action}' from artifact state."
    return {
        "thought": thought,
        "action": action,
        "input": _default_input_for(action, state),
        "debug_artifacts": _artifact_debug_labels(inspection),
        "debug_skips": skipped,
        "debug_artifact_status": _artifact_bool_map(inspection),
    }


def _fetch_repeat_allowed(state: AgentState, inspection: dict[str, dict[str, Any]]) -> bool:
    """Return whether selecting fetch again is strongly justified."""
    fetch_ready = bool(inspection.get("fetch", {}).get("ready", False))
    if not fetch_ready:
        return True

    paper_count = load_memory(state, "fetched_paper_count", 0)
    if isinstance(paper_count, int) and paper_count <= 0:
        return True

    return _allow_retry_same(state, "fetch")


def _repair_override_action(state: AgentState, inspection: dict[str, dict[str, Any]]) -> str | None:
    """Return a repair-policy override action when safe."""
    decision = load_memory(state, "last_repair_decision", {})
    if not isinstance(decision, dict):
        return None

    strategy = decision.get("strategy", {})
    if not isinstance(strategy, dict):
        return None

    strategy_name = str(strategy.get("strategy", "")).strip()
    suggested = str(strategy.get("next_action", "")).strip().lower()

    if strategy_name == "stop_early":
        return "finish"

    if strategy_name == "retry_same" and suggested in PLANNER_ACTIONS and _allow_retry_same(state, suggested):
        return suggested

    if suggested in PLANNER_ACTIONS and suggested != "fetch" and not _action_is_blocked(state, suggested):
        return suggested

    if suggested == "fetch" and _fetch_repeat_allowed(state, inspection):
        return suggested

    return None


def _apply_progression_guard(plan: dict, state: AgentState, inspection: dict[str, dict[str, Any]]) -> dict:
    """Guard planner output so it follows progression and avoids repeats."""
    progression_plan = _build_progression_plan(state, inspection)
    base_skips = progression_plan.get("debug_skips", [])
    debug_skips: list[str] = [str(item) for item in base_skips] if isinstance(base_skips, list) else []

    action = str(plan.get("action", "")).strip().lower()
    if action not in PLANNER_ACTIONS:
        debug_skips.append("Planner output invalid; switched to progression fallback.")
        progression_plan["debug_skips"] = debug_skips
        return progression_plan

    thought = str(plan.get("thought", "")).strip() or progression_plan["thought"]
    input_text = str(plan.get("input", "")).strip() or _default_input_for(action, state)
    preferred_action = str(progression_plan.get("action", "finish"))

    if action == "fetch" and not _fetch_repeat_allowed(state, inspection):
        debug_skips.append("Skipped fetch: fetch already succeeded and papers exist.")
        action = preferred_action
        input_text = _default_input_for(action, state)

    last_success_action = _last_successful_action(state)
    if (
        action != "finish"
        and action == last_success_action
        and not _allow_retry_same(state, action)
        and preferred_action != action
    ):
        debug_skips.append(
            f"Skipped {action}: same as previous successful action; advancing to {preferred_action}."
        )
        action = preferred_action
        input_text = _default_input_for(action, state)

    if action != "finish" and _action_is_blocked(state, action):
        debug_skips.append(f"Skipped {action}: blocked by circuit breaker.")
        action = preferred_action
        input_text = _default_input_for(action, state)

    if action != preferred_action and not _allow_retry_same(state, action):
        debug_skips.append(f"Adjusted action to progression step '{preferred_action}'.")
        action = preferred_action
        input_text = _default_input_for(action, state)

    return {
        "thought": thought,
        "action": action,
        "input": input_text,
        "debug_artifacts": _artifact_debug_labels(inspection),
        "debug_skips": debug_skips,
        "debug_artifact_status": _artifact_bool_map(inspection),
    }


def _recent_lessons(topic: str, limit: int = 3) -> list[dict]:
    """Load recent persistent lessons relevant to a topic."""
    lessons = load_lessons(limit=limit * 3)
    if not lessons:
        return []

    topic_lower = topic.strip().lower()
    filtered: list[dict] = []
    for lesson in lessons:
        lesson_topic = str(lesson.get("topic", "")).strip().lower()
        if topic_lower and lesson_topic and lesson_topic != topic_lower:
            continue
        filtered.append(lesson)
    return filtered[-limit:]


def generate_planner_prompt(state: AgentState, inspection: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """Generate planner system and user prompts with JSON-only instructions."""
    system_prompt = (
        "You are a planning module for an autonomous research pipeline.\n"
        "Return ONLY valid JSON with keys: thought, action, input.\n"
        "Allowed actions: fetch, pdf, summarize, report, insights, gaps, hypotheses, experiment, run_experiment, finish.\n"
        "Prioritize the next missing pipeline artifact and avoid repeating successful actions."
    )

    history_lines = [
        f"- step={item.step} action={item.action} status={item.status} reason={item.reason}" for item in state.history[-8:]
    ]
    history_text = "\n".join(history_lines) if history_lines else "- none"

    recent_failures = str(load_memory(state, "recent_failure_summary", "No recent failures."))
    repair_advice = load_memory(state, "last_repair_decision", {})
    repair_text = json.dumps(repair_advice) if isinstance(repair_advice, dict) else "none"
    circuit_hints = json.dumps(load_memory(state, "circuit_breakers", {}))
    lesson_text = json.dumps(_recent_lessons(state.topic))
    artifact_text = json.dumps(
        {
            action: {"ready": bool(data.get("ready", False)), "detail": str(data.get("detail", ""))}
            for action, data in inspection.items()
        }
    )

    user_prompt = (
        f"Topic: {state.topic}\n"
        f"Iteration: {state.iteration}/{state.max_iterations}\n"
        f"Current goal: {state.current_goal}\n"
        f"Recent history:\n{history_text}\n"
        f"Artifact status: {artifact_text}\n"
        f"Recent failures: {recent_failures}\n"
        f"Repair advice: {repair_text}\n"
        f"Circuit hints: {circuit_hints}\n"
        f"Relevant persistent lessons: {lesson_text}\n"
        f"Memory summary: {summarize_memory(state)}\n"
        "Return JSON only."
    )
    return system_prompt, user_prompt


def fallback_plan(state: AgentState, inspection: dict[str, dict[str, Any]] | None = None) -> dict:
    """Return a safe deterministic plan when planner output is unavailable."""
    if inspection is None:
        inspection = _inspect_artifacts(state)

    if bool(load_memory(state, "stop_early_requested", False)):
        return {
            "thought": "Repair strategy requested early stop with partial outputs.",
            "action": "finish",
            "input": "",
            "debug_artifacts": _artifact_debug_labels(inspection),
            "debug_skips": ["Selected finish because stop_early_requested=True."],
            "debug_artifact_status": _artifact_bool_map(inspection),
        }

    repair_action = _repair_override_action(state, inspection)
    if repair_action is not None:
        candidate = {
            "thought": "Following latest repair strategy recommendation.",
            "action": repair_action,
            "input": _default_input_for(repair_action, state),
        }
        return _apply_progression_guard(candidate, state, inspection)

    progression_plan = _build_progression_plan(state, inspection)
    return _apply_progression_guard(progression_plan, state, inspection)


def _normalize_planner_output(
    parsed: object,
    state: AgentState,
    baseline_plan: dict,
    inspection: dict[str, dict[str, Any]],
) -> dict:
    """Normalize and validate planner output with immediate fallback behavior."""
    if not isinstance(parsed, dict):
        return baseline_plan

    action = str(parsed.get("action", "")).strip().lower()
    if action not in PLANNER_ACTIONS:
        return baseline_plan

    thought = str(parsed.get("thought", "")).strip() or str(baseline_plan.get("thought", ""))
    input_text = str(parsed.get("input", "")).strip() or _default_input_for(action, state)
    candidate = {"thought": thought, "action": action, "input": input_text}
    return _apply_progression_guard(candidate, state, inspection)


def plan_next_step(state: AgentState, config: Config) -> dict:
    """Plan the next agent action using LLM output with deterministic progression guards."""
    inspection = _inspect_artifacts(state)
    baseline_plan = fallback_plan(state, inspection=inspection)

    try:
        validate_provider_setup(config)
        client = get_client(config)
        system_prompt, user_prompt = generate_planner_prompt(state, inspection)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
            timeout=PLANNER_LLM_TIMEOUT_SECONDS,
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return baseline_plan

        return _normalize_planner_output(parsed, state, baseline_plan, inspection)
    except Exception:
        return baseline_plan
