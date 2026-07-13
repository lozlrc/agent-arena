"""Fault isolation for untrusted agent code.

Every agent call goes through `GuardedAgent.call`, which enforces:
  * a hard per-move CPU-time limit (ITIMER_VIRTUAL / SIGVTALRM
    preemption, so even an infinite loop in agent code is
    interrupted). CPU time rather than wall clock keeps fault verdicts
    deterministic: whether a move times out depends on the work it
    does, not on machine load. An agent that blocks without consuming
    CPU is not preempted by this timer — the evaluation API's parent
    process covers that with a wall-clock kill;
  * exception containment (a crashing agent never takes down a match);
  * action validation (delegated to the caller via `validate`).

On any violation the referee substitutes a deterministic fallback
action and records a fault against the seat. A seat that exceeds
MAX_FAULTS is downgraded to fallback-only for the rest of the match.

This is in-process isolation, so the observation objects handed to
agents are immutable views (see the engines) but a hostile agent can
still fight the interpreter; the hard boundary for untrusted
submissions is the disposable worker process the evaluation API runs
them in.
"""

from __future__ import annotations

import signal
import time
from typing import Any, Callable

MOVE_TIMEOUT_S = 0.10   # per-move hard limit, CPU seconds
MAX_FAULTS = 3          # per seat per match before downgrade


class AgentTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise AgentTimeout()


class GuardedAgent:
    """Wraps one agent seat with timing, fault counting and fallbacks."""

    def __init__(self, agent: Any, seat: int):
        self.agent = agent
        self.seat = seat
        self.faults = 0
        self.time_spent = 0.0
        self.n_moves = 0

    @property
    def downgraded(self) -> bool:
        return self.faults >= MAX_FAULTS

    def call(
        self,
        method: str,
        obs: dict,
        validate: Callable[[Any], bool],
        fallback: Callable[[], Any],
    ) -> tuple[Any, str | None]:
        """Returns (action, fault_reason|None)."""
        self.n_moves += 1
        if self.downgraded:
            return fallback(), "downgraded"

        old = signal.signal(signal.SIGVTALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_VIRTUAL, MOVE_TIMEOUT_S)
        start = time.perf_counter()
        try:
            action = getattr(self.agent, method)(obs)
        except AgentTimeout:
            self.faults += 1
            return fallback(), "timeout"
        except Exception as exc:  # noqa: BLE001 — untrusted code
            self.faults += 1
            return fallback(), f"crash:{type(exc).__name__}"
        finally:
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            signal.signal(signal.SIGVTALRM, old)
            self.time_spent += time.perf_counter() - start

        if not validate(action):
            self.faults += 1
            return fallback(), "invalid_action"
        return action, None
