"""Parallel match orchestration — game-agnostic.

A match is fully determined by (game, lineup, seed), so scheduling is
embarrassingly parallel: matches fan out over a ProcessPoolExecutor
(one OS process per worker — crash isolation comes free) and results
stream back for collection. Lineup generation is itself seeded, so an
entire N-match tournament is reproducible from one integer.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from arena.games import GAMES


@dataclass(frozen=True)
class MatchSpec:
    lineup: tuple[str, ...]   # agent names, seat order
    seed: int
    game: str = "saboteur"


def tournament_specs(n_matches: int, master_seed: int,
                     pool: list[str] | None = None,
                     game: str = "saboteur") -> list[MatchSpec]:
    """Seeded schedule: every lineup and match seed derives from
    master_seed, so the whole tournament replays from one integer."""
    gd = GAMES[game]
    pool = pool or gd.ranked
    rng = random.Random(master_seed)
    return [
        MatchSpec(lineup=tuple(gd.make_lineup(rng, pool)),
                  seed=rng.getrandbits(48), game=game)
        for _ in range(n_matches)
    ]


def play_with_agents(game: str, agents: list, seed: int,
                     keep_transcript: bool = True) -> dict[str, Any]:
    """Run one match with pre-built agent instances (also used by the
    submission evaluator, where one seat is untrusted code)."""
    gd = GAMES[game]
    r = gd.run(agents, seed)
    out = {
        "game": game,
        "seed": r.seed,
        "lineup": r.agent_names,
        "roles": r.roles,
        "winner": r.winner,
        "round_wins": r.round_wins,
        "hash": r.transcript_hash(),
        "faults": r.faults,
        "move_time_s": r.move_time_s,
        "n_moves": r.n_moves,
        **gd.summarize(r),
    }
    if keep_transcript:
        out["events"] = r.events
    return out


def play(spec: MatchSpec, keep_transcript: bool = True) -> dict[str, Any]:
    gd = GAMES[spec.game]
    agents = [gd.registry[name]() for name in spec.lineup]
    return play_with_agents(spec.game, agents, spec.seed, keep_transcript)


def _play_light(spec: MatchSpec) -> dict[str, Any]:
    return play(spec, keep_transcript=False)


def run_serial(specs: list[MatchSpec],
               keep_transcripts: bool = True) -> list[dict]:
    return [play(s, keep_transcripts) for s in specs]


def run_parallel(specs: list[MatchSpec], workers: int,
                 keep_transcripts: bool = True,
                 chunksize: int = 64) -> list[dict]:
    fn = play if keep_transcripts else _play_light
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, specs, chunksize=chunksize))


def timed(fn, *args, **kw) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0
