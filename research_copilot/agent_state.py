"""State dataclasses for the autonomous research agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentAction:
    """Represent one action attempted by the agent."""

    step: int
    action: str
    input: str
    reason: str
    status: str = "pending"

    def to_dict(self) -> dict:
        """Return the action as a plain dictionary."""
        return {
            "step": self.step,
            "action": self.action,
            "input": self.input,
            "reason": self.reason,
            "status": self.status,
        }


@dataclass(slots=True)
class AgentState:
    """Track state across autonomous planning and execution iterations."""

    topic: str
    iteration: int
    max_iterations: int
    history: list[AgentAction] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    current_goal: str = ""
    last_result: str | None = None
    done: bool = False

    def to_dict(self) -> dict:
        """Return the agent state as a plain dictionary."""
        return {
            "topic": self.topic,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "history": [item.to_dict() for item in self.history],
            "memory": self.memory,
            "current_goal": self.current_goal,
            "last_result": self.last_result,
            "done": self.done,
        }
