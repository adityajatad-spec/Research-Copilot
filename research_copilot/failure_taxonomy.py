"""Heuristic failure taxonomy helpers for bounded agent self-repair."""

from __future__ import annotations


def classify_failure(error_text: str, action: str | None = None) -> dict:
    """Classify one failure into category, severity, and repair level."""
    text = (error_text or "").lower()
    action_name = (action or "").lower()

    if "ollama is not running" in text:
        return {
            "category": "provider_unavailable",
            "severity": "high",
            "repair_level": 2,
            "reason": "Configured LLM provider is unavailable.",
        }

    if "input file not found" in text or "no such file" in text:
        return {
            "category": "missing_input",
            "severity": "medium",
            "repair_level": 3,
            "reason": "A required prerequisite file is missing.",
        }

    if "timeout" in text:
        timeout_level = 1 if action_name == "run_experiment" else 2
        return {
            "category": "execution_failure",
            "severity": "medium",
            "repair_level": timeout_level,
            "reason": "Execution exceeded the allowed timeout window.",
        }

    if "json" in text and "parse" in text:
        return {
            "category": "parse_error",
            "severity": "low",
            "repair_level": 1,
            "reason": "Structured output parsing failed.",
        }

    if "unsupported fields" in text:
        return {
            "category": "tool_failure",
            "severity": "medium",
            "repair_level": 2,
            "reason": "Tool/API compatibility mismatch was detected.",
        }

    if "connection" in text or "resolve" in text or "network" in text:
        return {
            "category": "network",
            "severity": "medium",
            "repair_level": 0,
            "reason": "Network connectivity issue detected.",
        }

    return {
        "category": "unknown",
        "severity": "low",
        "repair_level": 1,
        "reason": "Failure did not match a known taxonomy rule.",
    }


def summarize_failure_pattern(failures: list[dict]) -> str:
    """Summarize repeated failures in one compact sentence."""
    if not failures:
        return "No failures recorded."

    category_counts: dict[str, int] = {}
    for item in failures:
        category = str(item.get("category", "unknown"))
        category_counts[category] = category_counts.get(category, 0) + 1

    top_items = sorted(category_counts.items(), key=lambda pair: pair[1], reverse=True)[:3]
    joined = ", ".join(f"{category} ({count})" for category, count in top_items)
    return f"Recent failures are mostly: {joined}."
