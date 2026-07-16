"""Tests for the simulated-API stress path: the latency/error-injecting
agent, the thread-based concurrent runner, and the `stress` CLI command."""

import argparse
import json

from arena.agents.sim_llm import PERSONAS, SimulatedLLMAgent
from arena.game.dilemma import coop_rate, run_pd_match
from arena.runner.orchestrator import MatchSpec, run_serial, run_threaded


def sim(persona, **kw):
    kw.setdefault("mean_latency", 0.0)
    kw.setdefault("jitter", 0.0)
    return SimulatedLLMAgent(persona=persona, **kw)


def test_sim_personas_play_as_advertised():
    # deceiver always defects, honest always cooperates
    r = run_pd_match([sim("deceiver"), sim("honest")], seed=1,
                     communication=True)
    assert coop_rate(r.events, 0) == 0.0     # deceiver
    assert coop_rate(r.events, 1) == 1.0     # honest
    assert not r.faults                      # no errors injected


def test_sim_game_decisions_reproducible_despite_io_stream():
    # latency/errors use a separate seeded stream, so game outcomes are
    # identical across runs even though timing is not part of them
    def run():
        return run_pd_match([sim("deceiver"), sim("tft")], seed=21,
                            communication=True).round_wins
    assert run() == run()


def test_sim_errors_are_contained_as_faults():
    agents = [sim("tft", error_rate=1.0), sim("honest", error_rate=1.0)]
    r = run_pd_match(agents, seed=4, communication=True)
    assert r.winner in ("seat0", "seat1", "draw")   # completed anyway
    assert sum(r.faults.values()) > 0               # every call faulted


def test_sim_latency_is_actually_spent():
    import time
    a = sim("tft", mean_latency=0.02, jitter=0.0)
    a.reset(0, __import__("random").Random(0))
    t0 = time.perf_counter()
    a.play({"history_self": (), "history_opp": ()})
    assert time.perf_counter() - t0 >= 0.015        # it really slept


def test_all_personas_defined():
    assert set(PERSONAS) == {"deceiver", "honest", "tft"}


def test_threaded_runner_equals_serial_for_deterministic_agents():
    specs = [MatchSpec(("deceiver", "honest_coop"), seed=s,
                       game="dilemma_comms") for s in range(8)]
    serial = run_serial(specs, keep_transcripts=False)
    threaded = run_threaded(specs, workers=4, keep_transcripts=False)
    assert [x["hash"] for x in serial] == [x["hash"] for x in threaded]


def test_threaded_runner_agent_factory_is_used():
    calls = []

    def factory(name, seat):
        calls.append((name, seat))
        return sim("tft")
    specs = [MatchSpec(("x", "y"), seed=1, game="dilemma_comms")]
    run_threaded(specs, workers=2, keep_transcripts=False, agent_factory=factory)
    assert calls == [("x", 0), ("y", 1)]


def test_stress_command_reports_speedup_and_contained_errors(capsys, tmp_path):
    from arena.cli import cmd_stress
    cmd_stress(argparse.Namespace(
        matches=12, concurrency=8, latency=0.02, error_rate=0.3,
        serial_sample=2, db=str(tmp_path / "s.db")))
    out = json.loads(capsys.readouterr().out)
    assert out["game"] == "dilemma_comms"
    assert out["matches"] == 12
    assert out["concurrency_speedup"] > 1.0            # overlap beat serial
    assert out["simulated_api_errors_contained"] > 0   # errors were injected
    assert out["total_agent_calls"] > 0


def test_stress_serial_sample_zero_does_not_crash(capsys, tmp_path):
    # --serial-sample 0 must clamp to a valid baseline, not ZeroDivisionError
    from arena.cli import cmd_stress
    cmd_stress(argparse.Namespace(
        matches=4, concurrency=4, latency=0.01, error_rate=0.0,
        serial_sample=0, db=str(tmp_path / "s.db")))
    out = json.loads(capsys.readouterr().out)
    assert out["serial_baseline_matches"] >= 1
    assert out["concurrency_speedup"] > 0
