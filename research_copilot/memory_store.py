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
