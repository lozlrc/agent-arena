"""Research runner for the dilemma_comms variant.

Run a series of matches between two agents (baseline, cheap-talk mock,
or LLM) and collect cooperation / deception statistics. LLM agents make
real API calls, so this runs from the CLI (never the public submission
endpoint) and you supply the key via the environment.

  # deterministic mock sanity check (no key, no cost):
  uv run python -m arena.research series --a deceiver --b honest_coop --matches 20

  # LLM vs baseline (needs `uv sync --extra llm` and OPENAI_API_KEY):
  OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o-mini \
    uv run python -m arena.research series --a llm_openai --b tit_for_tat --matches 10

  # LLM vs LLM, different models head-to-head:
  OPENAI_API_KEY=sk-... uv run python -m arena.research series \
    --a llm_openai --a-model gpt-4o --b llm_openai --b-model gpt-4o-mini --matches 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arena.agents.llm_pd import OpenAIPDAgent
from arena.game import dilemma
from arena.games import GAMES

RESULTS = Path(__file__).resolve().parent.parent / "results"
LLM_NAMES = {"llm_openai", "llm"}


def _factory(spec: str, model: str | None, label: str, api_key: str | None):
    """Return a no-arg callable building an agent for one side."""
    reg = GAMES["dilemma_comms"].registry
    if spec in LLM_NAMES:
        return lambda: OpenAIPDAgent(model=model, name=label, api_key=api_key)
    if spec not in reg:
        raise SystemExit(
            f"unknown agent {spec!r}; choose from {sorted(reg)} or llm_openai")
    cls = reg[spec]
    return lambda: cls()


def _is_llm(spec: str) -> bool:
    return spec in LLM_NAMES


def cmd_series(args) -> None:
    # distinct labels so transcripts/metrics tell the two sides apart
    a_label = args.a if not _is_llm(args.a) else f"A:{args.a_model or 'env'}"
    b_label = args.b if not _is_llm(args.b) else f"B:{args.b_model or 'env'}"
    if a_label == b_label:                    # self-play: keep the buckets apart
        a_label, b_label = a_label + "#A", b_label + "#B"
    # per-side keys fall back to the shared env var
    a_key = args.a_key or os.environ.get("OPENAI_API_KEY")
    b_key = args.b_key or os.environ.get("OPENAI_API_KEY")
    if _is_llm(args.a) or _is_llm(args.b):
        # preflight so we fail before spending money on a misconfigured run
        try:
            import openai  # noqa: F401
        except ImportError:
            raise SystemExit("openai SDK not installed — run `uv sync --extra llm`")
        if (_is_llm(args.a) and not a_key) or (_is_llm(args.b) and not b_key):
            raise SystemExit(
                "OPENAI_API_KEY is not set (env var or --a-key/--b-key)")
        n_llm = _is_llm(args.a) + _is_llm(args.b)   # 2 calls/round per LLM side
        calls = args.matches * dilemma.COMMS_MAX_ROUNDS * 2 * n_llm
        print(f"[cost] LLM run: up to ~{calls} API calls "
              f"({args.matches} matches x {dilemma.COMMS_MAX_ROUNDS} rounds "
              f"x 2 calls x {n_llm} LLM side(s))")

    make_a = _factory(args.a, args.a_model, a_label, a_key)
    make_b = _factory(args.b, args.b_model, b_label, b_key)
    run = GAMES["dilemma_comms"].run

    agg = {a_label: _blank(), b_label: _blank()}
    wins = {a_label: 0, b_label: 0, "draw": 0}
    transcripts = []
    for i in range(args.matches):
        r = run([make_a(), make_b()], args.seed + i)
        labels = [a_label, b_label]
        for seat, lab in enumerate(labels):
            broken, promises = dilemma.promise_breaks(r.events, seat)
            a = agg[lab]
            a["matches"] += 1
            a["score"] += r.round_wins[f"seat{seat}"]
            a["rounds"] += next(e["rounds"] for e in r.events
                                if e["t"] == "result")
            a["coop_sum"] += dilemma.coop_rate(r.events, seat)
            a["broken"] += broken
            a["promises"] += promises
            a["faults"] += r.faults.get(seat, 0)
        wins[r.winner if r.winner == "draw" else labels[int(r.winner[-1])]] += 1
        transcripts.append({"seed": r.seed, "lineup": [a_label, b_label],
                            "winner": r.winner, "scores": r.round_wins,
                            "events": r.events})
        print(f"  match {i + 1}/{args.matches}: {a_label} {r.round_wins['seat0']}"
              f" - {r.round_wins['seat1']} {b_label} ({r.winner})")

    print("\n=== summary ===")
    for lab in (a_label, b_label):
        a = agg[lab]
        print(f"  {lab:16s} avg_score={a['score'] / a['matches']:6.1f} "
              f"coop={100 * a['coop_sum'] / a['matches']:5.1f}% "
              f"betrayal={100 * a['broken'] / max(1, a['promises']):5.1f}% "
              f"(broke {a['broken']}/{a['promises']} coop-promises) "
              f"faults={a['faults']}")
    print(f"  record: {a_label} {wins[a_label]} / draw {wins['draw']} / "
          f"{wins[b_label]} {b_label}")

    if args.out:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / args.out
        out.write_text("\n".join(json.dumps(t) for t in transcripts))
        print(f"\ntranscripts -> {out}")


def _blank() -> dict:
    return {"matches": 0, "score": 0, "rounds": 0, "coop_sum": 0.0,
            "broken": 0, "promises": 0, "faults": 0}


def main() -> None:
    p = argparse.ArgumentParser(prog="arena.research")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("series", help="run A vs B comms matches")
    sp.add_argument("--a", required=True, help="agent name or llm_openai")
    sp.add_argument("--b", required=True, help="agent name or llm_openai")
    sp.add_argument("--a-model", default=None, help="OpenAI model for side A")
    sp.add_argument("--b-model", default=None, help="OpenAI model for side B")
    sp.add_argument("--a-key", default=None)
    sp.add_argument("--b-key", default=None)
    sp.add_argument("--matches", type=int, default=10)
    sp.add_argument("--seed", type=int, default=1000)
    sp.add_argument("--out", default=None, help="save transcripts to results/<name>")
    sp.set_defaults(fn=cmd_series)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
