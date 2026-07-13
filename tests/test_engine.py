import pytest

from arena.agents.baselines import REGISTRY
from arena.runner.match import run_match
from arena.runner.orchestrator import play, tournament_specs


def make(names):
    return [REGISTRY[n]() for n in names]


LINEUP = ["suspicion", "random", "naive_truster", "aggressive_sab",
          "cautious_voter"]


def test_match_completes_with_winner():
    r = run_match(make(LINEUP), seed=42)
    assert r.winner in ("loyalist", "saboteur")
    assert max(r.round_wins.values()) == 3
    assert not r.faults  # well-behaved agents produce no faults


def test_determinism_same_seed_same_hash():
    a = run_match(make(LINEUP), seed=123)
    b = run_match(make(LINEUP), seed=123)
    assert a.transcript_hash() == b.transcript_hash()
    assert a.events == b.events


def test_different_seeds_differ():
    hashes = {run_match(make(LINEUP), seed=s).transcript_hash()
              for s in range(20)}
    assert len(hashes) > 1


def test_crash_agent_is_contained():
    r = run_match(make(["crash_test"] + LINEUP[1:]), seed=7)
    assert r.winner in ("loyalist", "saboteur")  # match still completes


def test_slow_agent_times_out_but_match_completes():
    r = run_match(make(["slow_test"] + LINEUP[1:]), seed=7)
    assert r.winner in ("loyalist", "saboteur")
    assert r.faults.get(0, 0) >= 1
    fault_events = [e for e in r.events if e["t"] == "fault" and e["seat"] == 0]
    assert any(e["reason"] == "timeout" for e in fault_events)


def test_loyalists_cannot_sabotage():
    # aggressive_sab in every seat: sabotage every mission when evil.
    r = run_match(make(["aggressive_sab"] * 5), seed=1)
    for e in r.events:
        if e["t"] == "mission":
            n_evil_on_team = sum(1 for p in e["team"]
                                 if r.roles[p] == "saboteur")
            assert e["sabotages"] <= n_evil_on_team


def test_tournament_specs_reproducible():
    assert tournament_specs(50, 99) == tournament_specs(50, 99)
    hashes_a = [play(s)["hash"] for s in tournament_specs(10, 99)]
    hashes_b = [play(s)["hash"] for s in tournament_specs(10, 99)]
    assert hashes_a == hashes_b
