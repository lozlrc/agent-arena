"""Saboteur: a 5-player hidden-role game (Resistance-style).

Players 0-4. Two are secret SABOTEURS (they know each other), three are
LOYALISTS. The game is played over up to 5 rounds; first team to 3 round
wins takes the match.

Each round, with team sizes [2, 3, 2, 3, 3]:
  1. A rotating leader proposes a team of the required size.
  2. Discussion: each player broadcasts one structured message
     {"accuse": pid|None, "vouch": pid|None}.
  3. All players vote approve/reject (majority approves). Up to 5
     proposal attempts per round; the 5th is auto-approved.
  4. The approved team secretly plays SUCCESS or SABOTAGE cards.
     Loyalists always play SUCCESS. Any SABOTAGE card wins the round
     for the saboteurs; the public log shows only the sabotage count.

Everything is driven by a single match seed: role assignment, leader
order, and each agent's private RNG stream, so a (agents, seed) pair
fully determines the transcript.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

N_PLAYERS = 5
N_SABOTEURS = 2
TEAM_SIZES = [2, 3, 2, 3, 3]
ROUNDS_TO_WIN = 3
MAX_PROPOSALS = 5

LOYALIST = "loyalist"
SABOTEUR = "saboteur"


@dataclass
class MatchResult:
    seed: int
    agent_names: list[str]          # seat -> agent name
    roles: list[str]                # seat -> role
    winner: str                     # "loyalist" | "saboteur"
    round_wins: dict[str, int]
    events: list[dict[str, Any]]    # full public+secret transcript
    faults: dict[int, int]          # seat -> fault count (crash/timeout/invalid)
    move_time_s: dict[int, float]   # seat -> cumulative agent think time
    n_moves: int

    def transcript_hash(self) -> str:
        canonical = json.dumps(
            {
                "seed": self.seed,
                "agents": self.agent_names,
                "roles": self.roles,
                "winner": self.winner,
                "events": self.events,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class GameState:
    """Referee-side state; builds agent observations."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.seed = seed
        seats = list(range(N_PLAYERS))
        self.saboteurs = sorted(self.rng.sample(seats, N_SABOTEURS))
        self.roles = [
            SABOTEUR if p in self.saboteurs else LOYALIST for p in seats
        ]
        self.leader = self.rng.randrange(N_PLAYERS)
        self.round_num = 0
        self.round_wins = {LOYALIST: 0, SABOTEUR: 0}
        self.public_log: list[dict[str, Any]] = []

    def observation(self, player: int, phase: str, **extra: Any) -> dict:
        # public_log is handed out as a tuple so agents can't corrupt
        # the referee's transcript (inner events are still shared —
        # in-process isolation is defense-in-depth, the evaluation
        # API's worker process is the hard boundary for hostile code).
        obs = {
            "player_id": player,
            "role": self.roles[player],
            "n_players": N_PLAYERS,
            "round": self.round_num,
            "team_size": TEAM_SIZES[self.round_num],
            "leader": self.leader,
            "round_wins": dict(self.round_wins),
            "phase": phase,
            "public_log": tuple(self.public_log),
        }
        if self.roles[player] == SABOTEUR:
            obs["saboteurs"] = list(self.saboteurs)
        obs.update(extra)
        return obs
