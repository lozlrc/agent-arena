"""Fault isolation for agent code.

Every agent call goes through `GuardedAgent.call`, which enforces:
  * a hard per-move timeout via signal preemption,
  * exception containment (a crashing agent never takes down a match),
  * action validation (delegated to the caller via `validate`).

Two timing modes:
  * CPU time (default, ITIMER_VIRTUAL / SIGVTALRM) for scripted agents.
    Whether a move times out depends on the work it does, not on
    machine load, so fault verdicts stay deterministic and reruns are
    byte-identical. An agent that blocks without consuming CPU is not
    preempted by this timer.
  * wall clock (ITIMER_REAL / SIGALRM) for I/O-bound agents (e.g. an
    LLM adapter making a network call, which consumes almost no CPU
    while waiting). The budget is much larger (seconds) and, because
    the wall time of a network call varies, matches that use it are
    only statistically — not byte-for-byte — reproducible.

On any violation the referee substitutes a deterministic fallback
action and records a fault against the seat. A seat that exceeds
`max_faults` is downgraded to fallback-only for the rest of the match.

This is in-process isolation, so the observation objects handed to
agents are immutable views (see the engines) but a hostile agent can
still fight the interpreter; the hard boundary for untrusted
submissions is the disposable worker process the evaluation API runs
them in. (LLM agents are trusted, run from the registry/CLI — they are
never reachable through the untrusted-submission path, which blocks
imports and network.)
"""

from __future__ import annotations

import signal
import time
from typing import Any, Callable

MOVE_TIMEOUT_S = 0.10   # per-move hard limit, CPU seconds (scripted agents)
LLM_TIMEOUT_S = 30.0    # per-move hard limit, wall seconds (I/O-bound agents)
MAX_FAULTS = 3          # per seat per match before downgrade


class AgentTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise AgentTimeout()


class GuardedAgent:
    """Wraps one agent seat with timing, fault counting and fallbacks.

    wall_clock=False (default) preempts on CPU time (deterministic,
    for scripted agents); wall_clock=True preempts on real time (for
    I/O-bound agents such as LLM adapters). `timeout_s` defaults to the
    matching per-mode limit; `max_faults` bounds downgrades.
    """

    def __init__(self, agent: Any, seat: int, wall_clock: bool = False,
                 timeout_s: float | None = None, max_faults: int = MAX_FAULTS):
        self.agent = agent
        self.seat = seat
        self.faults = 0
        self.time_spent = 0.0
        self.n_moves = 0
        self.wall_clock = wall_clock
        self.max_faults = max_faults
        if timeout_s is None:
            timeout_s = LLM_TIMEOUT_S if wall_clock else MOVE_TIMEOUT_S
        self.timeout_s = timeout_s
        self._sig = signal.SIGALRM if wall_clock else signal.SIGVTALRM
        self._itimer = signal.ITIMER_REAL if wall_clock else signal.ITIMER_VIRTUAL

    @property
    def downgraded(self) -> bool:
        return self.faults >= self.max_faults

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

        old = signal.signal(self._sig, _alarm_handler)
        signal.setitimer(self._itimer, self.timeout_s)
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
            signal.setitimer(self._itimer, 0)
            signal.signal(self._sig, old)
            self.time_spent += time.perf_counter() - start

        if not validate(action):
            self.faults += 1
            return fallback(), "invalid_action"
        return action, None
