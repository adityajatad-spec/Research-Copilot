"""Bounded autonomous loop for end-to-end research pipeline execution."""

from __future__ import annotations

from pathlib import Path
import re

try:
    from .agent_state import AgentAction, AgentState
    from .circuit_breaker import CircuitBreaker
    from .config import Config
    from .critic import evaluate_state
    from .failure_taxonomy import classify_failure, summarize_failure_pattern
    from .memory_store import (
        load_memory,
        record_experiment_memory,
        record_repair_decision,
        store_circuit_breaker_state,
        store_memory,
        summarize_memory,
    )
    from .persistent_memory import append_lesson, append_run_history, load_lessons
    from .planner import plan_next_step
    from .repair_policy import build_repair_lesson, choose_repair_strategy
    from .utils import load_from_json, save_json_data, save_report, save_to_json
except ImportError:  # pragma: no cover - fallback for direct script execution
    from agent_state import AgentAction, AgentState
    from circuit_breaker import CircuitBreaker
    from config import Config
    from critic import evaluate_state
    from failure_taxonomy import classify_failure, summarize_failure_pattern
    from memory_store import (
        load_memory,
        record_experiment_memory,
        record_repair_decision,
        store_circuit_breaker_state,
        store_memory,
        summarize_memory,
    )
    from persistent_memory import append_lesson, append_run_history, load_lessons
    from planner import plan_next_step
    from repair_policy import build_repair_lesson, choose_repair_strategy
    from utils import load_from_json, save_json_data, save_report, save_to_json


DEFAULT_MAX_ITERATIONS = 6
HARD_MAX_ITERATIONS = 8
DEFAULT_FETCH_SOURCE = "hybrid"
DEFAULT_FETCH_MAX_RESULTS = 12
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
PARAM_PATTERN = re.compile(r"(?P<key>[a-zA-Z0-9_-]+)=(?P<value>[^ ]+)")
DEFAULT_EXPERIMENT_DATASET = "demo-dataset"
DEFAULT_EXPERIMENT_OUTPUT_DIR = "output/experiment_run"
DEFAULT_EXPERIMENT_EPOCHS = 1
DEFAULT_EXPERIMENT_LEARNING_RATE = 1e-4
DEFAULT_EXPERIMENT_SEED = 42
DEFAULT_EXPERIMENT_TIMEOUT = 120


def _parse_params(input_text: str) -> dict[str, str]:
    """Parse key=value input text into a dictionary."""
    params: dict[str, str] = {}
    for match in PARAM_PATTERN.finditer(input_text):
        params[match.group("key")] = match.group("value")
    return params


def _safe_int(value: str | None, default: int) -> int:
    """Parse an integer safely with a fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    """Parse a float safely with a fallback."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_circuit_breaker(state: AgentState, action: str) -> CircuitBreaker:
    """Load one action circuit breaker from memory."""
    states = load_memory(state, "circuit_breakers", {})
    if not isinstance(states, dict):
        return CircuitBreaker()

    row = states.get(action, {})
    if not isinstance(row, dict):
        return CircuitBreaker()

    return CircuitBreaker(
        failure_threshold=int(row.get("failure_threshold", 3) or 3),
        recovery_timeout_seconds=int(row.get("recovery_timeout_seconds", 300) or 300),
        failure_count=int(row.get("failure_count", 0) or 0),
        last_failure_ts=row.get("last_failure_ts"),
        state=str(row.get("state", "closed") or "closed"),
    )


def _save_circuit_breaker(state: AgentState, action: str, breaker: CircuitBreaker) -> None:
    """Persist one action circuit breaker state to memory."""
    store_circuit_breaker_state(state, action, breaker.to_dict())


def _increment_failed_count(state: AgentState, action: str) -> int:
    """Increase and return failed attempt count for an action."""
    failed_counts = load_memory(state, "failed_action_counts", {})
    if not isinstance(failed_counts, dict):
        failed_counts = {}
    current_value = int(failed_counts.get(action, 0) or 0) + 1
    failed_counts[action] = current_value
    store_memory(state, "failed_action_counts", failed_counts)
    return current_value


def _clear_failed_count(state: AgentState, action: str) -> None:
    """Reset failed attempt count for an action after success."""
    failed_counts = load_memory(state, "failed_action_counts", {})
    if not isinstance(failed_counts, dict):
        failed_counts = {}
    failed_counts[action] = 0
    store_memory(state, "failed_action_counts", failed_counts)


def _record_failure_and_repair(
    state: AgentState,
    action: str,
    input_text: str,
    reason: str,
    error_message: str,
    breaker: CircuitBreaker,
) -> str:
    """Record one failure, classify it, and persist repair strategy decisions."""
    record_action(state, action, input_text, reason, "failed")
    state.last_result = error_message
    store_memory(state, "last_error", error_message)
    _increment_failed_count(state, action)

    failure_info = classify_failure(error_message, action)
    strategy = choose_repair_strategy(state, action, error_message, failure_info)
    record_repair_decision(state, action, failure_info, strategy)

    breaker.record_failure()
    _save_circuit_breaker(state, action, breaker)

    try:
        lesson = build_repair_lesson(state.topic, action, strategy, failure_info)
        append_lesson(lesson)
        store_memory(state, "last_repair_lesson", lesson)
    except Exception:
        # Persistent lesson write failures should never stop the agent.
        pass

    recent_failures = load_memory(state, "recent_failures", [])
    if isinstance(recent_failures, list):
        store_memory(state, "failure_summary", summarize_failure_pattern(recent_failures))

    repair_message = (
        f"{error_message} "
        f"[category={failure_info.get('category', 'unknown')}, strategy={strategy.get('strategy', 'retry_same')}]"
    )
    state.last_result = repair_message
    return repair_message


def _fetch_with_source(topic: str, source: str, max_results: int) -> tuple[list, str]:
    """Fetch papers from the requested source, with graceful source fallback."""
    try:
        from .fetcher import fetch_papers
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from fetcher import fetch_papers

    normalized_source = source.strip().lower() or DEFAULT_FETCH_SOURCE
    if normalized_source == "arxiv":
        return fetch_papers(topic, max_results), "arxiv"

    try:
        try:
            from .scholar_fetcher import fetch_hybrid_papers, fetch_semantic_scholar_papers
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from scholar_fetcher import fetch_hybrid_papers, fetch_semantic_scholar_papers

        if normalized_source == "semanticscholar":
            return fetch_semantic_scholar_papers(topic, max_results), "semanticscholar"
        return fetch_hybrid_papers(topic, max_results), "hybrid"
    except Exception:
        # If optional source tooling is unavailable, fall back to pure arXiv.
        return fetch_papers(topic, max_results), "arxiv"


def _ensure_results(state: AgentState, topic: str, source: str, max_results: int) -> list:
    """Ensure fetched paper metadata exists and return the paper list."""
    results_path = OUTPUT_PATHS["fetch"]
    if Path(results_path).exists():
        papers = load_from_json(results_path)
    else:
        papers, source_used = _fetch_with_source(topic, source, max_results)
        if not papers:
            raise ValueError("No papers were fetched.")
        save_to_json(papers, results_path)
        store_memory(state, "fetch_source", source_used)

    if not papers:
        raise ValueError("No papers are available in fetch results.")

    store_memory(state, "fetch_output_path", results_path)
    store_memory(state, "fetched_paper_count", len(papers))
    return papers


def _ensure_pdf(state: AgentState, topic: str, source: str, max_results: int) -> list:
    """Ensure PDF-enriched paper output exists and return papers."""
    pdf_output_path = OUTPUT_PATHS["pdf"]
    if Path(pdf_output_path).exists():
        papers = load_from_json(pdf_output_path)
    else:
        try:
            from .pdf_parser import build_pdf_stats, enrich_papers_with_pdf_text
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from pdf_parser import build_pdf_stats, enrich_papers_with_pdf_text

        papers = _ensure_results(state, topic, source, max_results)
        papers = enrich_papers_with_pdf_text(papers, pdf_dir="output/pdfs", max_pages=10, verbose=False)
        save_to_json(papers, pdf_output_path)
        stats = build_pdf_stats(papers)
        store_memory(state, "pdf_stats", stats)

    store_memory(state, "pdf_output_path", pdf_output_path)
    return papers


def _ensure_summaries(state: AgentState, topic: str, provider: str, model: str, source: str, max_results: int) -> list:
    """Ensure summarized paper output exists and return papers."""
    summary_path = OUTPUT_PATHS["summarize"]
    if Path(summary_path).exists():
        papers = load_from_json(summary_path)
    else:
        try:
            from .summarizer import summarize_all
            from .config import Config
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from summarizer import summarize_all
            from config import Config

        # Prefer PDF-enriched papers when available, fall back to fetched metadata.
        if Path(OUTPUT_PATHS["pdf"]).exists():
            papers = load_from_json(OUTPUT_PATHS["pdf"])
        else:
            papers = _ensure_results(state, topic, source, max_results)

        config = Config(provider=provider, model=model)
        papers = summarize_all(papers, config, verbose=False)
        save_to_json(papers, summary_path)

    summary_count = sum(1 for item in papers if item.summary is not None)
    store_memory(state, "summarize_output_path", summary_path)
    store_memory(state, "summary_count", summary_count)
    return papers


def _hypothesis_item_from_dict(data: dict):
    """Deserialize one hypothesis item payload."""
    try:
        from .models import ExperimentPlan, HypothesisItem
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from models import ExperimentPlan, HypothesisItem

    plan_data = data.get("experiment_plan", {}) if isinstance(data, dict) else {}
    plan = ExperimentPlan(
        objective=str(plan_data.get("objective", "")).strip(),
        datasets=[str(item) for item in plan_data.get("datasets", []) if str(item).strip()],
        baselines=[str(item) for item in plan_data.get("baselines", []) if str(item).strip()],
        metrics=[str(item) for item in plan_data.get("metrics", []) if str(item).strip()],
        implementation_notes=[str(item) for item in plan_data.get("implementation_notes", []) if str(item).strip()],
    )
    return HypothesisItem(
        title=str(data.get("title", "")).strip(),
        hypothesis=str(data.get("hypothesis", "")).strip(),
        novelty_rationale=str(data.get("novelty_rationale", "")).strip(),
        feasibility_rationale=str(data.get("feasibility_rationale", "")).strip(),
        experiment_plan=plan,
    )


def execute_action(state: AgentState, action: str, input_text: str, provider: str, model: str) -> str:
    """Execute an agent action by dispatching to existing project modules."""
    params = _parse_params(input_text)
    topic = state.topic
    source = params.get("source", DEFAULT_FETCH_SOURCE)
    max_results = _safe_int(params.get("max"), DEFAULT_FETCH_MAX_RESULTS)

    if action == "fetch":
        papers, source_used = _fetch_with_source(topic, source, max_results)
        if not papers:
            fallback_path = Path(OUTPUT_PATHS["fetch"])
            if fallback_path.exists():
                papers = load_from_json(str(fallback_path))
                source_used = "cached"
            else:
                raise ValueError("No papers were fetched.")

        if not papers:
            raise ValueError("No papers are available for downstream steps.")

        save_to_json(papers, OUTPUT_PATHS["fetch"])
        store_memory(state, "fetch_output_path", OUTPUT_PATHS["fetch"])
        store_memory(state, "fetched_paper_count", len(papers))
        store_memory(state, "fetch_source", source_used)
        return f"Fetched {len(papers)} papers from {source_used}."

    if action == "pdf":
        papers = _ensure_pdf(state, topic, source, max_results)
        store_memory(state, "pdf_output_path", OUTPUT_PATHS["pdf"])
        return f"PDF enrichment complete for {len(papers)} papers."

    if action == "summarize":
        papers = _ensure_summaries(state, topic, provider, model, source, max_results)
        return f"Summaries available for {len(papers)} papers."

    if action == "report":
        try:
            from .reporter import generate_report
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from reporter import generate_report

        papers = _ensure_summaries(state, topic, provider, model, source, max_results)
        report_text = generate_report(papers, topic)
        save_report(report_text, OUTPUT_PATHS["report"])
        store_memory(state, "report_output_path", OUTPUT_PATHS["report"])
        return "Markdown report generated."

    if action == "insights":
        try:
            from .config import Config
            from .insights import extract_insights
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from config import Config
            from insights import extract_insights

        papers = _ensure_summaries(state, topic, provider, model, source, max_results)
        config = Config(provider=provider, model=model)
        report = extract_insights(papers, topic, config)
        save_json_data(report.to_dict(), OUTPUT_PATHS["insights"])
        store_memory(state, "insights_output_path", OUTPUT_PATHS["insights"])
        store_memory(state, "insight_topics", report.major_themes)
        return "Cross-paper insights extracted."

    if action == "gaps":
        try:
            from .config import Config
            from .gaps import extract_gaps_and_contradictions, save_gap_report
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from config import Config
            from gaps import extract_gaps_and_contradictions, save_gap_report

        papers = _ensure_summaries(state, topic, provider, model, source, max_results)
        config = Config(provider=provider, model=model)
        report = extract_gaps_and_contradictions(papers, topic, config)
        save_gap_report(report, OUTPUT_PATHS["gaps"])
        store_memory(state, "gaps_output_path", OUTPUT_PATHS["gaps"])
        store_memory(state, "gap_count", len(report.explicit_research_gaps))
        return "Gap and contradiction report extracted."

    if action == "hypotheses":
        try:
            from .config import Config
            from .hypotheses import extract_hypotheses, load_optional_json, save_hypothesis_report
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from config import Config
            from hypotheses import extract_hypotheses, load_optional_json, save_hypothesis_report

        papers = _ensure_pdf(state, topic, source, max_results)
        insights_data = load_optional_json(OUTPUT_PATHS["insights"])
        gaps_data = load_optional_json(OUTPUT_PATHS["gaps"])
        config = Config(provider=provider, model=model)
        report = extract_hypotheses(
            papers=papers,
            topic=topic,
            config=config,
            insights_data=insights_data,
            gap_data=gaps_data,
        )
        save_hypothesis_report(report, OUTPUT_PATHS["hypotheses"])
        store_memory(state, "hypotheses_output_path", OUTPUT_PATHS["hypotheses"])
        store_memory(state, "hypothesis_titles", [item.title for item in report.hypotheses])
        return f"Hypothesis report generated with {len(report.hypotheses)} hypotheses."

    if action == "experiment":
        try:
            from .experiment_writer import generate_experiment_script, save_experiment_script
            from .hypotheses import load_optional_json
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from experiment_writer import generate_experiment_script, save_experiment_script
            from hypotheses import load_optional_json

        if not Path(OUTPUT_PATHS["hypotheses"]).exists():
            execute_action(state, "hypotheses", "", provider, model)

        report_data = load_optional_json(OUTPUT_PATHS["hypotheses"])
        if not isinstance(report_data, dict):
            raise ValueError("Hypothesis report is unavailable.")

        hypotheses = report_data.get("hypotheses", [])
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError("No hypotheses available for experiment scaffolding.")

        top_hypothesis = _hypothesis_item_from_dict(hypotheses[0])
        script = generate_experiment_script(top_hypothesis, report_data.get("topic", topic))
        save_experiment_script(script, OUTPUT_PATHS["experiment"])
        store_memory(state, "experiment_output_path", OUTPUT_PATHS["experiment"])
        store_memory(state, "experiment_script_path", OUTPUT_PATHS["experiment"])
        return "Experiment scaffold generated."

    if action == "run_experiment":
        try:
            from .result_parser import load_experiment_results, summarize_experiment_result
            from .run_experiment import safe_run_experiment
        except ImportError:  # pragma: no cover - fallback for direct script execution
            from result_parser import load_experiment_results, summarize_experiment_result
            from run_experiment import safe_run_experiment

        script_path = params.get("script") or load_memory(state, "experiment_script_path") or OUTPUT_PATHS["experiment"]
        dataset = params.get("dataset", DEFAULT_EXPERIMENT_DATASET)
        output_dir = params.get("output_dir", DEFAULT_EXPERIMENT_OUTPUT_DIR)
        epochs = _safe_int(params.get("epochs"), DEFAULT_EXPERIMENT_EPOCHS)
        learning_rate = _safe_float(params.get("learning_rate"), DEFAULT_EXPERIMENT_LEARNING_RATE)
        seed = _safe_int(params.get("seed"), DEFAULT_EXPERIMENT_SEED)
        timeout = _safe_int(params.get("timeout"), DEFAULT_EXPERIMENT_TIMEOUT)

        if not Path(script_path).exists():
            execute_action(state, "experiment", "", provider, model)
            script_path = load_memory(state, "experiment_script_path", OUTPUT_PATHS["experiment"])

        run_result = safe_run_experiment(
            script_path=script_path,
            dataset=dataset,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            timeout=timeout,
        )
        parsed_results = None
        results_path = run_result.get("results_path")
        if isinstance(results_path, str) and results_path:
            parsed_results = load_experiment_results(results_path)

        record_experiment_memory(state, run_result, parsed_results)
        store_memory(state, "experiment_script_path", script_path)
        store_memory(state, "run_experiment_output_dir", output_dir)

        return summarize_experiment_result(run_result, parsed_results)

    if action == "finish":
        return "Planner requested finish."

    raise ValueError(f"Unsupported action: {action}")


def record_action(state: AgentState, action: str, input_text: str, reason: str, status: str) -> None:
    """Append a new action record to state history."""
    state.history.append(
        AgentAction(
            step=state.iteration,
            action=action,
            input=input_text,
            reason=reason,
            status=status,
        )
    )


def safe_execute(state: AgentState, action: str, input_text: str, reason: str, provider: str, model: str) -> str:
    """Execute an action safely and record success/failure details."""
    breaker = _load_circuit_breaker(state, action)
    if not breaker.can_execute():
        _save_circuit_breaker(state, action, breaker)
        message = f"{action} skipped: circuit breaker is open."
        record_action(state, action, input_text, reason, "skipped")
        state.last_result = message
        store_memory(state, "last_error", message)
        return message

    _save_circuit_breaker(state, action, breaker)

    try:
        result = execute_action(state, action, input_text, provider, model)
        if action == "run_experiment" and not bool(load_memory(state, "experiment_success", False)):
            return _record_failure_and_repair(
                state=state,
                action=action,
                input_text=input_text,
                reason=reason,
                error_message=result,
                breaker=breaker,
            )

        record_action(state, action, input_text, reason, "completed")
        state.last_result = result
        store_memory(state, "last_success_action", action)
        breaker.record_success()
        _save_circuit_breaker(state, action, breaker)
        _clear_failed_count(state, action)

        known_bad = load_memory(state, "known_bad_actions", {})
        if isinstance(known_bad, dict) and action in known_bad:
            known_bad.pop(action, None)
            store_memory(state, "known_bad_actions", known_bad)

        return result
    except Exception as error:  # pragma: no cover - runtime safety path
        error_message = f"{action} failed: {error}"
        return _record_failure_and_repair(
            state=state,
            action=action,
            input_text=input_text,
            reason=reason,
            error_message=error_message,
            breaker=breaker,
        )


def _final_output_paths(state: AgentState) -> dict[str, str]:
    """Collect output paths produced or confirmed during this run."""
    paths: dict[str, str] = {}
    for action, default_path in OUTPUT_PATHS.items():
        memory_path = state.memory.get(f"{action}_output_path")
        if isinstance(memory_path, str) and Path(memory_path).exists():
            paths[action] = memory_path
            continue

        completed_in_run = any(
            item.action == action and item.status == "completed"
            for item in state.history
        )
        if completed_in_run and Path(default_path).exists():
            paths[action] = default_path

    experiment_script = state.memory.get("experiment_script_path") or state.memory.get("experiment_output_path")
    if isinstance(experiment_script, str) and Path(experiment_script).exists():
        paths["experiment_script"] = experiment_script
    elif Path(OUTPUT_PATHS["experiment"]).exists():
        paths["experiment_script"] = OUTPUT_PATHS["experiment"]

    experiment_results = state.memory.get("experiment_results_path") or state.memory.get("run_experiment_output_path")
    if isinstance(experiment_results, str) and Path(experiment_results).exists():
        paths["experiment_results"] = experiment_results
    elif Path(OUTPUT_PATHS["run_experiment"]).exists():
        paths["experiment_results"] = OUTPUT_PATHS["run_experiment"]

    return paths


def run_agent(topic: str, provider: str, model: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> dict:
    """Run the autonomous research loop and return a final run summary."""
    bounded_iterations = max(1, min(max_iterations, HARD_MAX_ITERATIONS))
    config = Config(provider=provider, model=model)

    state = AgentState(
        topic=topic.strip(),
        iteration=0,
        max_iterations=bounded_iterations,
        history=[],
        memory={},
        current_goal="Build complete research outputs through experiment scaffolding.",
    )
    store_memory(state, "provider", provider)
    store_memory(state, "model", model)
    store_memory(state, "persistent_lessons", load_lessons(limit=5))

    while state.iteration < state.max_iterations and not state.done:
        state.iteration += 1
        plan = plan_next_step(state, config)
        action = str(plan.get("action", "finish")).strip().lower()
        reason = str(plan.get("thought", "No planner rationale provided.")).strip()
        input_text = str(plan.get("input", "")).strip()

        store_memory(state, "last_plan", plan)

        if action == "finish":
            record_action(state, action, input_text, reason, "completed")
            state.last_result = "Planner requested completion."
        else:
            safe_execute(state, action, input_text, reason, provider, model)

        critic_decision = evaluate_state(state)
        store_memory(state, "last_critic_decision", critic_decision)

        if critic_decision.get("status") == "done":
            state.done = True
            state.last_result = str(critic_decision.get("reason", state.last_result))

    final_paths = _final_output_paths(state)
    store_memory(state, "final_output_paths", final_paths)
    recent_failures = load_memory(state, "recent_failures", [])
    failure_summary = summarize_failure_pattern(recent_failures) if isinstance(recent_failures, list) else "No failures."
    repair_history = load_memory(state, "repair_history", [])
    if isinstance(repair_history, list) and repair_history:
        recent_repairs = repair_history[-3:]
        repair_summary = "; ".join(
            f"{item.get('failed_action')}->{item.get('strategy', {}).get('strategy')}"
            for item in recent_repairs
            if isinstance(item, dict)
        )
    else:
        repair_summary = "No repair actions recorded."
    partial_result = bool(load_memory(state, "partial_result", False)) or (not state.done and bool(final_paths))
    circuit_breakers = load_memory(state, "circuit_breakers", {})
    recent_failures_payload = recent_failures[-20:] if isinstance(recent_failures, list) else []
    repair_history_payload = repair_history[-20:] if isinstance(repair_history, list) else []
    memory_summary = summarize_memory(state)

    final_payload = {
        "topic": state.topic,
        "done": state.done,
        "iterations": state.iteration,
        "history": [item.to_dict() for item in state.history],
        "memory_summary": memory_summary,
        "final_output_paths": final_paths,
        "repair_summary": repair_summary,
        "partial_result": partial_result,
        "failure_summary": failure_summary,
        "circuit_breakers": circuit_breakers if isinstance(circuit_breakers, dict) else {},
        "recent_failures": recent_failures_payload,
        "repair_history": repair_history_payload,
    }

    try:
        append_run_history(final_payload)
    except Exception:
        # Run history persistence should not break core execution.
        pass

    return final_payload
