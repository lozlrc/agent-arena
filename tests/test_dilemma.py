from arena.game.dilemma import PD_REGISTRY, run_pd_match
from arena.runner.orchestrator import play, tournament_specs


def make(names):
    return [PD_REGISTRY[n]() for n in names]


def test_match_completes_and_scores_add_up():
    r = run_pd_match(make(["tit_for_tat", "pavlov"]), seed=42)
    assert r.winner in ("seat0", "seat1", "draw")
    res = [e for e in r.events if e["t"] == "result"][0]
    assert 100 <= res["rounds"] <= 200
    assert res["scores"] == [r.round_wins["seat0"], r.round_wins["seat1"]]
    assert not r.faults


def test_determinism_same_seed_same_hash():
    a = run_pd_match(make(["generous_tft", "all_d"]), seed=123)
    b = run_pd_match(make(["generous_tft", "all_d"]), seed=123)
    assert a.transcript_hash() == b.transcript_hash()
    assert a.events == b.events


def test_all_d_exploits_all_c():
    r = run_pd_match(make(["all_c", "all_d"]), seed=5, noise=0.0)
    res = [e for e in r.events if e["t"] == "result"][0]
    assert r.winner == "seat1"
    assert res["scores"][0] == 0                      # sucker every round
    assert res["scores"][1] == 5 * res["rounds"]      # temptation every round


def test_tit_for_tat_retaliates_without_noise():
    r = run_pd_match(make(["tit_for_tat", "all_d"]), seed=5, noise=0.0)
    res = [e for e in r.events if e["t"] == "result"][0]
    n = res["rounds"]
    # TFT loses only the first round, then mutual defection.
    assert res["scores"] == [n - 1, 5 + (n - 1)]


def test_pavlov_self_play_is_all_cooperation_draw():
    r = run_pd_match(make(["pavlov", "pavlov"]), seed=9, noise=0.0)
    assert r.winner == "draw"
    rounds = [e for e in r.events if e["t"] == "round"]
    assert all(e["actions"] == [True, True] for e in rounds)


def test_noise_flips_are_recorded_and_observed():
    r = run_pd_match(make(["all_c", "all_c"]), seed=1)  # default 2% noise
    rounds = [e for e in r.events if e["t"] == "round"]
    flips = [(e["actions"][i], i) for e in rounds
             for i in (0, 1) if e["flipped"][i]]
    # all_c always intends C, so every flipped action must be a D.
    assert flips and all(a is False for a, _ in flips)


def test_pd_fault_containment():
    class Boom(PD_REGISTRY["all_c"]):
        name = "boom"

        def play(self, obs):
            raise RuntimeError("boom")

    r = run_pd_match([Boom(), PD_REGISTRY["tit_for_tat"]()], seed=3)
    assert r.winner in ("seat0", "seat1", "draw")
    assert r.faults[0] >= 1


def test_pd_tournament_specs_reproducible_and_distinct_pairs():
    specs = tournament_specs(50, 77, game="dilemma")
    assert specs == tournament_specs(50, 77, game="dilemma")
    assert all(len(set(s.lineup)) == 2 for s in specs)
    a = [play(s)["hash"] for s in specs[:10]]
    b = [play(s)["hash"] for s in specs[:10]]
    assert a == b
