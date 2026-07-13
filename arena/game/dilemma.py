"""Iterated Prisoner's Dilemma: the second arena game.

Two players, a seeded-random number of rounds (100-200, hidden from
the agents so end-game defection can't be timed), and 2% seeded
execution noise (an intended action occasionally flips — both players
observe realized actions, which is what makes forgiveness matter).

Payoffs per round: both cooperate 3/3, both defect 1/1,
defect-vs-cooperate 5/0. Higher total score wins; equal is a draw.

Same platform guarantees as Saboteur: a match is a pure function of
(lineup, seed), every agent call goes through the referee sandbox
(SIGALRM preemption, exception containment, action validation), and
transcripts are canonical-JSON SHA-256 hashable.
"""

from __future__ import annotations

import random

from arena.game.saboteur import MatchResult
from arena.runner.sandbox import GuardedAgent

PAYOFF = {(True, True): (3, 3), (True, False): (0, 5),
          (False, True): (5, 0), (False, False): (1, 1)}
MIN_ROUNDS, MAX_ROUNDS = 100, 200
NOISE = 0.02

COOPERATE = True
DEFECT = False


class PDAgent:
    """Base class for Prisoner's Dilemma submissions.

    Override play(obs) -> bool (True = cooperate). obs contains the
    realized (post-noise) action histories for both players. Use only
    self.rng for randomness so matches stay deterministic.
    """

    name = "pd_base"

    def reset(self, player_id: int, rng: random.Random) -> None:
        self.me = player_id
        self.rng = rng

    def play(self, obs: dict) -> bool:
        return COOPERATE


class AlwaysCooperate(PDAgent):
    name = "all_c"


class AlwaysDefect(PDAgent):
    name = "all_d"

    def play(self, obs):
        return DEFECT


class RandomPD(PDAgent):
    name = "random_pd"

    def play(self, obs):
        return bool(self.rng.getrandbits(1))


class TitForTat(PDAgent):
    """Cooperate first, then mirror the opponent's last realized action."""
    name = "tit_for_tat"

    def play(self, obs):
        return obs["history_opp"][-1] if obs["history_opp"] else COOPERATE


class GenerousTitForTat(PDAgent):
    """Tit-for-tat that forgives a defection 10% of the time — the
    classic antidote to noise-induced retaliation spirals."""
    name = "generous_tft"

    def play(self, obs):
        if not obs["history_opp"] or obs["history_opp"][-1]:
            return COOPERATE
        return self.rng.random() < 0.10


class GrimTrigger(PDAgent):
    """Cooperate until the opponent's first defection, then never again."""
    name = "grim_trigger"

    def play(self, obs):
        return all(obs["history_opp"])


class Pavlov(PDAgent):
    """Win-stay lose-shift: cooperate iff both played the same last round."""
    name = "pavlov"

    def play(self, obs):
        if not obs["history_opp"]:
            return COOPERATE
        return obs["history_self"][-1] == obs["history_opp"][-1]


PD_REGISTRY: dict[str, type[PDAgent]] = {
    a.name: a
    for a in [AlwaysCooperate, AlwaysDefect, RandomPD, TitForTat,
              GenerousTitForTat, GrimTrigger, Pavlov]
}
PD_RANKED = ["all_c", "all_d", "random_pd", "tit_for_tat", "generous_tft",
             "grim_trigger", "pavlov"]


def run_pd_match(agents: list, seed: int, noise: float = NOISE) -> MatchResult:
    """agents: 2 agent instances. Deterministic in (agents, seed, noise)."""
    assert len(agents) == 2
    rng = random.Random(seed)
    n_rounds = rng.randint(MIN_ROUNDS, MAX_ROUNDS)
    fallback_rng = random.Random(seed ^ 0xFA11BACC)
    guarded = [GuardedAgent(a, i) for i, a in enumerate(agents)]
    for i, a in enumerate(agents):
        a.reset(player_id=i, rng=random.Random((seed << 2) | i))

    history: list[list[bool]] = [[], []]  # realized actions per seat
    scores = [0, 0]
    events: list[dict] = []

    for r in range(n_rounds):
        realized = []
        flipped = []
        for p in (0, 1):
            # tuple views: agents can't corrupt the referee's history
            obs = {
                "player_id": p,
                "round": r,
                "history_self": tuple(history[p]),
                "history_opp": tuple(history[1 - p]),
                "score_self": scores[p],
                "score_opp": scores[1 - p],
            }
            action, fault = guarded[p].call(
                "play", obs,
                validate=lambda a: isinstance(a, bool),
                fallback=lambda: bool(fallback_rng.getrandbits(1)),
            )
            if fault:
                events.append({"t": "fault", "seat": p, "round": r,
                               "reason": fault})
            flip = rng.random() < noise
            realized.append(action ^ flip)
            flipped.append(flip)
        for p in (0, 1):
            history[p].append(realized[p])
        pay = PAYOFF[(realized[0], realized[1])]
        scores[0] += pay[0]
        scores[1] += pay[1]
        events.append({"t": "round", "r": r, "actions": realized,
                       "flipped": flipped})

    events.append({"t": "result", "scores": scores, "rounds": n_rounds})
    if scores[0] == scores[1]:
        winner = "draw"
    else:
        winner = f"seat{0 if scores[0] > scores[1] else 1}"
    return MatchResult(
        seed=seed,
        agent_names=[g.agent.name for g in guarded],
        roles=["p0", "p1"],
        winner=winner,
        round_wins={"seat0": scores[0], "seat1": scores[1]},
        events=events,
        faults={g.seat: g.faults for g in guarded if g.faults},
        move_time_s={g.seat: round(g.time_spent, 6) for g in guarded},
        n_moves=sum(g.n_moves for g in guarded),
    )


def coop_rate(events: list[dict], seat: int) -> float:
    rounds = [e for e in events if e["t"] == "round"]
    if not rounds:
        return 0.0
    return sum(e["actions"][seat] for e in rounds) / len(rounds)
