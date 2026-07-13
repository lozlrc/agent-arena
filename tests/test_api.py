from fastapi.testclient import TestClient

from arena.submissions import validate_code
from arena.web.app import app

client = TestClient(app)

GOOD_AGENT = '''
class MyAgent(BaseAgent):
    name = "tester"
    def vote(self, obs):
        return self.me in obs["team"] or obs["attempt"] >= 2
    def mission(self, obs):
        return self.role == "loyalist"
'''


def test_evaluate_good_agent():
    r = client.post("/api/evaluate", json={
        "code": GOOD_AGENT, "name": "tester", "matches": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matches"] == 30
    assert 0 <= body["win_pct"] <= 100
    assert body["faults"] == 0
    assert len(body["eval_hash"]) == 64
    assert body["sample_match"]["events"]
    # appears on the submissions leaderboard
    subs = client.get("/api/submissions").json()
    assert any(s["name"] == "tester" for s in subs)


def test_evaluate_is_deterministic():
    a = client.post("/api/evaluate", json={
        "code": GOOD_AGENT, "matches": 20, "seed": 7}).json()
    b = client.post("/api/evaluate", json={
        "code": GOOD_AGENT, "matches": 20, "seed": 7}).json()
    assert a["eval_hash"] == b["eval_hash"]
    assert a["win_pct"] == b["win_pct"]


def test_malicious_imports_rejected():
    for snippet in [
        "import os",
        "from subprocess import run",
        "x = ().__class__",
        "open('/etc/passwd')",
        "getattr(int, 'x')",
    ]:
        code = snippet + "\nclass A(BaseAgent):\n    pass"
        assert validate_code(code), snippet
        r = client.post("/api/evaluate", json={"code": code})
        assert r.status_code == 422, snippet


def test_crashing_agent_reports_faults_not_500():
    code = '''
class Boom(BaseAgent):
    name = "boom"
    def vote(self, obs):
        raise ValueError("boom")
'''
    r = client.post("/api/evaluate", json={"code": code, "matches": 10})
    assert r.status_code == 200
    assert r.json()["faults"] > 0


def test_infinite_loop_agent_is_contained():
    code = '''
class Spin(BaseAgent):
    name = "spin"
    def discuss(self, obs):
        while True:
            pass
'''
    r = client.post("/api/evaluate", json={"code": code, "matches": 3})
    assert r.status_code == 200
    assert r.json()["faults"] > 0


def test_no_agent_class_is_422():
    r = client.post("/api/evaluate", json={"code": "x = 1"})
    assert r.status_code == 422


PD_AGENT = '''
class TitForTwoTats(PDAgent):
    def play(self, obs):
        h = obs["history_opp"]
        return len(h) < 2 or h[-1] or h[-2]
'''


def test_evaluate_dilemma_agent():
    r = client.post("/api/evaluate", json={
        "code": PD_AGENT, "name": "tf2t", "game": "dilemma", "matches": 30})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["game"] == "dilemma"
    assert 0 <= body["win_pct"] <= 100
    assert 0 <= body["coop_pct"] <= 100
    assert body["pts_per_round"] > 0
    assert body["faults"] == 0
    subs = client.get("/api/submissions?game=dilemma").json()
    assert any(s["name"] == "tf2t" for s in subs)


def test_wrong_base_class_for_game_is_422():
    # saboteur-style code submitted to the dilemma game
    r = client.post("/api/evaluate", json={
        "code": GOOD_AGENT, "game": "dilemma", "matches": 5})
    assert r.status_code == 422


def test_unknown_game_is_404():
    r = client.post("/api/evaluate", json={"code": PD_AGENT, "game": "chess"})
    assert r.status_code == 404
