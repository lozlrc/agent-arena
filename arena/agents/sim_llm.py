"""Simulated LLM agent — mimics a real API for stress-testing.

A real OpenAI PD agent (arena/agents/llm_pd.py) is I/O-bound: every
message/move waits ~1-2 s on the network and costs tokens. That makes
it impossible to benchmark the platform's concurrency, timeout, and
fault handling cheaply. This agent stands in for it: each call

  * sleeps a sampled latency (releasing the GIL, exactly like a real
    network wait, so the thread-based runner can overlap many games),
  * fails with a configurable probability (mimicking API 500s / rate
    limits), which exercises the referee's fault containment, and
  * plays a fixed persona so transcripts are meaningful.

Game decisions are seeded (reproducible); the latency/error stream is
seeded separately from the match seed, so a stress run is repeatable
while still exhibiting realistic per-call timing variation.
"""

from __future__ import annotations

import random
import time

from arena.game.dilemma import COOPERATE, DEFECT, PDAgent

# persona -> (message(obs) -> str, play(obs) -> bool)
PERSONAS = {
    "deceiver": (
        lambda obs: "Let's both cooperate this round — I promise I will.",
        lambda obs: DEFECT,
    ),
    "honest": (
        lambda obs: "Let's cooperate; I'll hold to my word.",
        lambda obs: COOPERATE,
    ),
    "tft": (
        lambda obs: "I'll mirror your last move — let's cooperate.",
        lambda obs: obs["history_opp"][-1] if obs["history_opp"] else COOPERATE,
    ),
}


class SimulatedLLMAgent(PDAgent):
    """Latency/error-injecting stand-in for a real LLM PD agent."""

    name = "llm_sim"

    def __init__(self, persona: str = "tft", mean_latency: float = 1.0,
                 jitter: float = 0.3, min_latency: float = 0.05,
                 error_rate: float = 0.0, name: str | None = None):
        assert persona in PERSONAS, persona
        self.persona = persona
        self.mean_latency = mean_latency
        self.jitter = jitter
        self.min_latency = min_latency
        self.error_rate = error_rate
        if name:
            self.name = name

    def reset(self, player_id: int, rng: random.Random) -> None:
        super().reset(player_id, rng)
        # Separate, seeded stream for timing/errors so it never perturbs
        # the (also seeded) game decisions.
        self._io = random.Random(rng.getrandbits(48))

    def _simulate_api(self) -> None:
        latency = max(self.min_latency,
                      self._io.gauss(self.mean_latency, self.jitter))
        time.sleep(latency)
        if self._io.random() < self.error_rate:
            raise RuntimeError("simulated API error (500)")

    def message(self, obs: dict) -> str:
        self._simulate_api()
        return PERSONAS[self.persona][0](obs)

    def play(self, obs: dict) -> bool:
        self._simulate_api()
        return PERSONAS[self.persona][1](obs)


class SimDeceiver(SimulatedLLMAgent):
    name = "sim_deceiver"

    def __init__(self):
        super().__init__(persona="deceiver", mean_latency=1.0)


class SimHonest(SimulatedLLMAgent):
    name = "sim_honest"

    def __init__(self):
        super().__init__(persona="honest", mean_latency=1.0)
