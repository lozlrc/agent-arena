"""Arena CLI.

  uv run python -m arena.cli [--game saboteur|dilemma] <command>

  bench       --matches 10000   throughput: serial vs parallel
  verify      --matches 5000    reproducibility: transcript hash equality
  tournament  --matches 50000   ranked tournament -> SQLite + ratings
  faults      --matches 1000    fault injection (saboteur pool)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arena import db
from arena.games import GAMES
from arena.ratings import compute_ratings
from arena.runner.orchestrator import (
    run_parallel,
    run_serial,
    timed,
    tournament_specs,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
MASTER_SEED = 20260708


def _write(args, kind: str, out: dict) -> None:
    print(json.dumps(out, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{kind}_{args.game}.json").write_text(json.dumps(out, indent=2))
    con = db.connect(args.db)
    db.set_stat(con, f"{args.game}:{kind}", out)


def cmd_bench(args) -> None:
    specs = tournament_specs(args.matches, MASTER_SEED, game=args.game)
    warm = tournament_specs(200, MASTER_SEED + 1, game=args.game)
    # absorb first-run effects (pyc compilation, file cache) before timing
    run_parallel(warm, args.workers, keep_transcripts=False)

    _, t_serial = timed(run_serial, specs, False)
    _, t_par = timed(run_parallel, specs, args.workers, False)
    _write(args, "bench", {
        "game": args.game,
        "matches": args.matches,
        "workers": args.workers,
        "serial_s": round(t_serial, 3),
        "parallel_s": round(t_par, 3),
        "serial_gps": round(args.matches / t_serial, 1),
        "parallel_gps": round(args.matches / t_par, 1),
        "speedup": round(t_serial / t_par, 2),
    })


def cmd_verify(args) -> None:
    """Reproducibility: same specs, two runs, different parallelism."""
    specs = tournament_specs(args.matches, MASTER_SEED, game=args.game)
    a = run_parallel(specs, args.workers, keep_transcripts=False)
    b = run_serial(specs, keep_transcripts=False)
    same = sum(x["hash"] == y["hash"] for x, y in zip(a, b))
    _write(args, "verify", {
        "game": args.game,
        "matches": args.matches,
        "identical_transcripts": same,
        "reproducibility_pct": round(100.0 * same / args.matches, 2),
    })


def cmd_tournament(args) -> None:
    specs = tournament_specs(args.matches, MASTER_SEED, game=args.game)
    results, t = timed(run_parallel, specs, args.workers, False)

    # Ratings from summaries (no transcripts needed), then replay a
    # sample of matches with full transcripts for the match viewer —
    # determinism means the replays are the same matches.
    rows = compute_ratings(results, args.game)
    keep_every = max(1, args.matches // args.keep_transcripts)
    sample_idx = list(range(0, args.matches, keep_every))
    sampled = run_parallel([specs[i] for i in sample_idx], args.workers, True)
    for i, r in zip(sample_idx, sampled):
        assert r["hash"] == results[i]["hash"], "replay mismatch"
        results[i] = r

    con = db.connect(args.db)
    # delete + reinsert commit together (insert_matches commits)
    con.execute("DELETE FROM matches WHERE game = ?", (args.game,))
    db.insert_matches(con, results)
    db.save_ratings(con, args.game, rows)
    db.set_stat(con, f"{args.game}:tournament", {
        "matches": args.matches,
        "wall_s": round(t, 2),
        "games_per_s": round(args.matches / t, 1),
        "master_seed": MASTER_SEED,
    })
    print(f"[{args.game}] {args.matches} matches in {t:.1f}s "
          f"({args.matches / t:.0f} games/s) -> {args.db}")
    for r in rows:
        extra = " ".join(f"{k}={v}" for k, v in r["extra"].items())
        print(f"  {r['name']:16s} rating={r['mu'] - 3 * r['sigma']:6.2f} "
              f"(mu={r['mu']:6.2f} sigma={r['sigma']:4.2f}) "
              f"win%={100 * r['wins'] / r['matches']:5.1f} {extra}")


def cmd_faults(args) -> None:
    """Inject crashing + infinite-looping agents; prove the platform
    survives. Uses the saboteur pool (fault-injection bots live there)."""
    if args.game != "saboteur":
        raise SystemExit("faults injection pool is defined for --game saboteur")
    pool = ["crash_test", "slow_test", "suspicion", "random", "naive_truster"]
    specs = tournament_specs(args.matches, MASTER_SEED + 2, pool=pool,
                             game="saboteur")
    results, t = timed(run_parallel, specs, args.workers, False)
    _write(args, "faults", {
        "game": args.game,
        "matches_scheduled": args.matches,
        "matches_completed": len(results),
        "completion_pct": round(100.0 * len(results) / args.matches, 2),
        "matches_with_faults": sum(1 for r in results if r["faults"]),
        "total_faults_contained": sum(sum(r["faults"].values())
                                      for r in results),
        "wall_s": round(t, 2),
    })


def main() -> None:
    p = argparse.ArgumentParser(prog="arena")
    p.add_argument("--db", default=str(db.DEFAULT_DB))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--game", choices=sorted(GAMES), default="saboteur")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, default_n in [
        ("bench", cmd_bench, 10000),
        ("verify", cmd_verify, 5000),
        ("tournament", cmd_tournament, 50000),
        ("faults", cmd_faults, 1000),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("--matches", type=int, default=default_n)
        if name == "tournament":
            sp.add_argument("--keep-transcripts", type=int, default=200)
        sp.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
