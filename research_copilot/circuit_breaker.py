"""Lightweight circuit breaker utility for bounded action retries."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(slots=True)
class CircuitBreaker:
    """Track repeated failures and gate execution with cooldown recovery."""

    failure_threshold: int = 3
    recovery_timeout_seconds: int = 300
    failure_count: int = 0
    last_failure_ts: float | None = None
    state: str = "closed"

    def record_success(self) -> None:
        """Record a successful execution and close the circuit."""
        self.failure_count = 0
        self.last_failure_ts = None
        self.state = "closed"

    def record_failure(self) -> None:
        """Record a failed execution and open the circuit when threshold is reached."""
        self.failure_count += 1
        self.last_failure_ts = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
        elif self.state not in {"closed", "open", "half_open"}:
            self.state = "closed"

    def can_execute(self) -> bool:
        """Return whether execution is currently allowed."""
        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure_ts is None:
                return False
            elapsed = time.time() - self.last_failure_ts
            if elapsed >= self.recovery_timeout_seconds:
                self.state = "half_open"
                return True
            return False

        return self.state == "half_open"

    def to_dict(self) -> dict:
        """Return breaker state as a plain dictionary."""
        return {
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "failure_count": self.failure_count,
            "last_failure_ts": self.last_failure_ts,
            "state": self.state,
        }
