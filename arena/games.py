"""Game registry: everything game-specific behind one interface.

Adding a game = one GameDef entry: how to build a lineup, how to run a
match, which agents are ranked, and how to score a finished match for
ratings. The orchestrator, CLI, evaluation API, and web UI are all
game-agnostic and dispatch through this table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from arena.agents.baselines import RANKED, REGISTRY, BaseAgent
from arena.game import dilemma
from arena.game.saboteur import SABOTEUR, MatchResult
from arena.runner.match import run_match


@dataclass(frozen=True)
class GameDef:
    name: str
    n_seats: int
    registry: dict[str, type]        # agent name -> class
    ranked: list[str]                # default ranked pool
    base_class: type                 # what submissions subclass
    base_class_name: str             # name exposed to submitted code
    run: Callable[[list, int], MatchResult]
    make_lineup: Callable[[random.Random, list[str]], list[str]]
    # rating_teams: result row -> (list of teams of seat indices, ranks)
    # fed to openskill.
    rating_teams: Callable[[dict], tuple[list[list[int]], list[int]]]
    # seat_won: did this seat win the match? (row = play() output dict)
    seat_won: Callable[[dict, int], bool]
    # summarize: cheap per-match extras merged into the play() row
    # (so ratings never need full transcripts).
    summarize: Callable[[MatchResult], dict]
    # per-agent stat accumulator for leaderboard extras:
    # tally_init() -> acc; tally_update(acc, row, seat); tally_final(acc) -> dict
    tally_init: Callable[[], dict]
    tally_update: Callable[[dict, dict, int], None]
    tally_final: Callable[[dict], dict]


def _saboteur_lineup(rng: random.Random, pool: list[str]) -> list[str]:
    lineup = list(pool) if len(pool) == 5 else rng.choices(pool, k=5)
    rng.shuffle(lineup)
    return lineup


def _saboteur_teams(r: dict) -> tuple[list[list[int]], list[int]]:
    loyal = [i for i, role in enumerate(r["roles"]) if role != SABOTEUR]
    evil = [i for i, role in enumerate(r["roles"]) if role == SABOTEUR]
    ranks = [0, 1] if r["winner"] != SABOTEUR else [1, 0]
    return [loyal, evil], ranks


def _saboteur_seat_won(r: dict, seat: int) -> bool:
    return (r["roles"][seat] == SABOTEUR) == (r["winner"] == SABOTEUR)


def _pd_lineup(rng: random.Random, pool: list[str]) -> list[str]:
    return rng.sample(pool, 2)


def _pd_teams(r: dict) -> tuple[list[list[int]], list[int]]:
    if r["winner"] == "draw":
        return [[0], [1]], [0, 0]
    w = int(r["winner"][-1])
    return [[0], [1]], [0, 1] if w == 0 else [1, 0]


def _pd_seat_won(r: dict, seat: int) -> bool:
    return r["winner"] == f"seat{seat}"


def _saboteur_tally_init() -> dict:
    return {"lw": 0, "lm": 0, "sw": 0, "sm": 0}


def _saboteur_tally_update(acc: dict, r: dict, seat: int) -> None:
    won = _saboteur_seat_won(r, seat)
    if r["roles"][seat] == SABOTEUR:
        acc["sm"] += 1
        acc["sw"] += won
    else:
        acc["lm"] += 1
        acc["lw"] += won


def _saboteur_tally_final(acc: dict) -> dict:
    return {
        "loyalist_win_pct": round(100 * acc["lw"] / max(1, acc["lm"]), 1),
        "saboteur_win_pct": round(100 * acc["sw"] / max(1, acc["sm"]), 1),
    }


def _pd_summarize(r: MatchResult) -> dict:
    return {
        "rounds": next(e["rounds"] for e in reversed(r.events)
                       if e["t"] == "result"),
        "coop": [dilemma.coop_rate(r.events, 0), dilemma.coop_rate(r.events, 1)],
    }


def _pd_tally_init() -> dict:
    return {"pts": 0, "rounds": 0, "coop_rounds": 0.0, "draws": 0, "n": 0}


def _pd_tally_update(acc: dict, r: dict, seat: int) -> None:
    acc["n"] += 1
    acc["pts"] += r["round_wins"][f"seat{seat}"]
    acc["rounds"] += r["rounds"]
    acc["coop_rounds"] += r["coop"][seat] * r["rounds"]
    acc["draws"] += r["winner"] == "draw"


def _pd_tally_final(acc: dict) -> dict:
    return {
        "pts_per_round": round(acc["pts"] / max(1, acc["rounds"]), 3),
        "coop_pct": round(100 * acc["coop_rounds"] / max(1, acc["rounds"]), 1),
        "draw_pct": round(100 * acc["draws"] / max(1, acc["n"]), 1),
    }


GAMES: dict[str, GameDef] = {
    "saboteur": GameDef(
        name="saboteur",
        n_seats=5,
        registry=REGISTRY,
        ranked=RANKED,
        base_class=BaseAgent,
        base_class_name="BaseAgent",
        run=run_match,
        make_lineup=_saboteur_lineup,
        rating_teams=_saboteur_teams,
        seat_won=_saboteur_seat_won,
        summarize=lambda r: {},
        tally_init=_saboteur_tally_init,
        tally_update=_saboteur_tally_update,
        tally_final=_saboteur_tally_final,
    ),
    "dilemma": GameDef(
        name="dilemma",
        n_seats=2,
        registry=dilemma.PD_REGISTRY,
        ranked=dilemma.PD_RANKED,
        base_class=dilemma.PDAgent,
        base_class_name="PDAgent",
        run=dilemma.run_pd_match,
        make_lineup=_pd_lineup,
        rating_teams=_pd_teams,
        seat_won=_pd_seat_won,
        summarize=_pd_summarize,
        tally_init=_pd_tally_init,
        tally_update=_pd_tally_update,
        tally_final=_pd_tally_final,
    ),
}
