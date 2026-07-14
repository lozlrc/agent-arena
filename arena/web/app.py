"""Agent Arena web API + UI: per-game leaderboards, match replays, and
remote agent evaluation (POST /api/evaluate)."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from arena import db as adb
from arena.games import GAMES
from arena.submissions import clean_name, evaluate_code, validate_code

DB_PATH = Path(os.environ.get("ARENA_DB", adb.DEFAULT_DB))
EVAL_TIMEOUT_S = 60
MAX_EVAL_MATCHES = 1000
DEFAULT_EVAL_SEED = 1000003

app = FastAPI(title="Agent Arena")

# At most 2 concurrent evaluations (protects the small VM); ephemeral
# store of evaluated submissions for the live submissions leaderboard.
_eval_slots = threading.Semaphore(2)
_submissions: dict[tuple[str, str], dict] = {}
_submissions_lock = threading.Lock()


def q(sql: str, *params) -> list[sqlite3.Row]:
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return []  # no results DB yet — serve an empty (not broken) UI
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def _check_game(game: str) -> str:
    if game not in GAMES:
        raise HTTPException(404, f"unknown game {game!r}; "
                            f"available: {sorted(GAMES)}")
    return game


@app.get("/api/games")
def api_games():
    return sorted(GAMES)


@app.get("/api/leaderboard")
def api_leaderboard(game: str = "saboteur"):
    _check_game(game)
    rows = [dict(r) for r in q("SELECT * FROM ratings WHERE game = ?", game)]
    for r in rows:
        r.update(json.loads(r.pop("extra")))
        r["rating"] = round(r["mu"] - 3 * r["sigma"], 2)
        r["win_pct"] = round(100 * r["wins"] / max(1, r["matches"]), 1)
    rows.sort(key=lambda r: r["rating"], reverse=True)
    return rows


@app.get("/api/stats")
def api_stats():
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return {}
    try:
        return adb.get_stats(con)
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


@app.get("/api/matches")
def api_matches(game: str | None = None, limit: int = 50):
    where = "WHERE transcript IS NOT NULL"
    params: list = []
    if game:
        where += " AND game = ?"
        params.append(_check_game(game))
    rows = q(
        f"SELECT id, game, seed, lineup, roles, winner, round_wins, hash,"
        f" faults FROM matches {where} ORDER BY id LIMIT ?",
        *params, max(1, min(limit, 500)))
    return [
        {**dict(r), "lineup": json.loads(r["lineup"]),
         "roles": json.loads(r["roles"]),
         "round_wins": json.loads(r["round_wins"]),
         "faults": json.loads(r["faults"])}
        for r in rows
    ]


@app.get("/api/matches/{match_id}")
def api_match(match_id: int):
    rows = q("SELECT * FROM matches WHERE id = ?", match_id)
    if not rows:
        raise HTTPException(404)
    r = dict(rows[0])
    for k in ("lineup", "roles", "round_wins", "faults", "transcript"):
        r[k] = json.loads(r[k]) if r[k] else None
    return r


def _eval_worker(out: mp.Queue, code: str, name: str,
                 n_matches: int, seed: int, game: str) -> None:
    """Runs in a disposable child process — the only place submitted
    code executes. Main thread of the child, so the referee sandbox's
    signal-based per-move preemption works."""
    try:
        out.put(("ok", evaluate_code(code, name, n_matches, seed, game)))
    except Exception as exc:  # noqa: BLE001 — untrusted code
        out.put(("error", f"{type(exc).__name__}: {exc}"))


class EvalRequest(BaseModel):
    code: str
    name: str | None = None
    game: str = "saboteur"
    matches: int = Field(default=200, ge=1, le=MAX_EVAL_MATCHES)
    seed: int = Field(default=DEFAULT_EVAL_SEED, ge=0)


@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest, x_api_key: str | None = Header(None)):
    required = os.environ.get("ARENA_API_KEY")
    if required and x_api_key != required:
        raise HTTPException(401, "missing or invalid X-API-Key")
    _check_game(req.game)
    errs = validate_code(req.code)
    if errs:
        raise HTTPException(422, {"validation_errors": errs})
    if not _eval_slots.acquire(blocking=False):
        raise HTTPException(429, "evaluation slots busy, retry shortly")
    try:
        name = clean_name(req.name)
        ctx = mp.get_context("spawn")
        chan: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_eval_worker,
            args=(chan, req.code, name, req.matches, req.seed, req.game),
        )
        proc.start()
        deadline = time.monotonic() + EVAL_TIMEOUT_S
        while True:
            try:
                status, payload = chan.get(timeout=0.5)
                break
            except queue.Empty:
                if not proc.is_alive():
                    proc.join()
                    raise HTTPException(
                        500, "evaluation worker died unexpectedly")
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.join()
                    raise HTTPException(
                        408,
                        f"evaluation exceeded {EVAL_TIMEOUT_S}s and was killed")
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        if status != "ok":
            raise HTTPException(422, {"agent_error": payload})
        summary = {k: v for k, v in payload.items() if k != "sample_match"}
        with _submissions_lock:
            _submissions[(req.game, name)] = summary
        return payload
    finally:
        _eval_slots.release()


@app.get("/api/submissions")
def api_submissions(game: str | None = None):
    with _submissions_lock:
        rows = [r for (g, _), r in _submissions.items()
                if game is None or g == game]
    rows.sort(key=lambda r: r["win_pct"], reverse=True)
    return rows


@app.get("/healthz")
def healthz():
    return {"ok": True}


STYLE = """
:root { --bg:#0d1117; --card:#161b22; --border:#30363d; --fg:#e6edf3;
        --dim:#8b949e; --accent:#58a6ff; --good:#3fb950; --bad:#f85149; }
* { box-sizing: border-box; margin: 0; }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 ui-monospace,
       SFMono-Regular, Menlo, monospace; padding: 2rem 1rem; }
main { max-width: 1020px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: .25rem; }
h2 { font-size: 1.05rem; margin: 1.75rem 0 .6rem; color: var(--accent); }
h3 { font-size: .9rem; margin: 1.1rem 0 .4rem; color: var(--dim);
     text-transform: uppercase; letter-spacing: .06em; }
p.sub { color: var(--dim); margin-bottom: 1rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr));
         gap: .75rem; }
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 8px; padding: .9rem 1rem; }
.card .num { font-size: 1.35rem; font-weight: 700; color: var(--accent); }
.card .lbl { color: var(--dim); font-size: .78rem; }
table { width: 100%; border-collapse: collapse; background: var(--card);
        border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
        margin-bottom: 1rem; }
th, td { padding: .5rem .7rem; text-align: right; border-top: 1px solid var(--border); }
th { background: #1c2129; color: var(--dim); font-size: .78rem; }
td:first-child, th:first-child { text-align: left; }
a { color: var(--accent); text-decoration: none; }
.win-loyalist, .win-seat0 { color: var(--good); }
.win-saboteur, .win-seat1 { color: var(--bad); }
.win-draw { color: var(--dim); }
.ev { padding: .15rem .5rem; border-left: 2px solid var(--border); margin: .15rem 0;
      color: var(--dim); }
.ev b { color: var(--fg); }
pre { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      padding: .9rem 1rem; overflow-x: auto; font-size: .78rem; line-height: 1.45;
      color: var(--fg); margin-bottom: 1rem; }
footer { color: var(--dim); font-size: .8rem; margin-top: 2.5rem; }
.tag { display: inline-block; padding: 0 .45rem; border: 1px solid var(--border);
       border-radius: 10px; font-size: .75rem; color: var(--dim); }
"""

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Arena</title><style>{style}</style></head><body><main>
<h1>Agent Arena</h1>
<p class="sub">Evaluation platform for AI agents in multi-agent games &mdash;
parallel seeded simulation, fault-isolated agent execution, reproducible
tournaments. Games: <b>Saboteur</b> (5-player hidden-role: deception,
communication, coordination) and <b>Prisoner's Dilemma</b> (iterated 1v1
with execution noise: cooperation, forgiveness, exploitation).</p>
<div id="stats"></div>
<h2>Leaderboards &mdash; TrueSkill-style ratings (&mu; &minus; 3&sigma;)</h2>
<h3>Saboteur</h3>
<table id="lb-saboteur"><thead><tr><th>agent</th><th>rating</th><th>&mu;</th>
<th>&sigma;</th><th>matches</th><th>win %</th><th>as loyalist</th>
<th>as saboteur</th></tr></thead><tbody></tbody></table>
<h3>Prisoner's Dilemma</h3>
<table id="lb-dilemma"><thead><tr><th>agent</th><th>rating</th><th>&mu;</th>
<th>&sigma;</th><th>matches</th><th>win %</th><th>draw %</th>
<th>pts/round</th><th>coop %</th></tr></thead><tbody></tbody></table>
<h2>Run your own agent &mdash; POST /api/evaluate</h2>
<p class="sub">Submit agent code and get measured results back: seeded
evaluation vs the baseline pool, win rates, fault count, move latency,
and a reproducible eval hash. Code is AST-screened, exec'd with
restricted builtins, and run in a disposable worker process with
per-move preemption &mdash; it never executes in the web process.</p>
<pre>curl -s <span class="origin"></span>/api/evaluate -H 'Content-Type: application/json' -d '{
  "game": "saboteur", "name": "greedy_accuser", "matches": 300,
  "code": "class MyAgent(BaseAgent):\\n    def vote(self, obs):\\n        return self.me in obs[\\"team\\"] or obs[\\"attempt\\"] >= 2\\n    def mission(self, obs):\\n        return self.role == \\"loyalist\\""
}'</pre>
<pre>curl -s <span class="origin"></span>/api/evaluate -H 'Content-Type: application/json' -d '{
  "game": "dilemma", "name": "tit_for_two_tats", "matches": 300,
  "code": "class MyAgent(PDAgent):\\n    def play(self, obs):\\n        h = obs[\\"history_opp\\"]\\n        return len(h) < 2 or h[-1] or h[-2]"
}'</pre>
<p class="sub">Interface: subclass <b>BaseAgent</b> (saboteur:
propose / discuss / vote / mission) or <b>PDAgent</b> (dilemma: play)
&mdash; see the README for observation schemas. Same (code, seed)
&rarr; same eval_hash. Allowed imports: math, random, itertools,
collections, statistics, functools, heapq, bisect.</p>
<div id="subsWrap" style="display:none"><h2>Submitted agents (this
session)</h2><table id="subs"><thead><tr><th>agent</th><th>game</th>
<th>matches</th><th>win %</th><th>details</th><th>faults</th>
<th>avg move ms</th></tr></thead><tbody></tbody></table></div>
<h2>Sample match replays</h2>
<table id="mt"><thead><tr><th>match</th><th>game</th><th>seed</th>
<th>winner</th><th>score</th><th>transcript hash</th></tr></thead>
<tbody></tbody></table>
<div id="replay"></div>
<footer>Every match is deterministic in (game, lineup, seed):
transcripts are SHA-256 hashed and re-verifiable. Faulting agents
(crash / timeout / invalid action) are contained per-seat and
recorded, never crash the platform. &mdash;
<a href="/api/stats">stats</a> &middot;
<a href="/api/leaderboard?game=saboteur">saboteur api</a> &middot;
<a href="/api/leaderboard?game=dilemma">dilemma api</a></footer>
</main><script>
async function j(u) { return (await fetch(u)).json(); }
function esc(s) { return String(s).replace(/[&<>]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
const GAME_LABEL = {saboteur: 'Saboteur', dilemma: "Prisoner's Dilemma"};

function statCards(s, game) {
  const cards = [];
  const t = s[game + ':tournament'], b = s[game + ':bench'],
        v = s[game + ':verify'], f = s[game + ':faults'];
  if (t) cards.push([t.matches.toLocaleString(), 'ranked matches'],
    [t.games_per_s.toLocaleString() + '/s', 'tournament throughput']);
  if (b) cards.push([b.parallel_gps.toLocaleString() + '/s',
    'peak games/s (' + b.workers + ' workers)'],
    [b.speedup + 'x', 'parallel speedup vs serial']);
  if (v) cards.push([v.reproducibility_pct + '%', 'reproducibility (' +
    v.matches.toLocaleString() + ' reruns)']);
  if (f) cards.push([f.completion_pct + '%', 'completion, ' +
    f.total_faults_contained.toLocaleString() + ' faults injected']);
  if (!cards.length) return '';
  return `<h3>${GAME_LABEL[game]}</h3><div class="cards">` +
    cards.map(([n, l]) =>
      `<div class="card"><div class="num">${n}</div><div class="lbl">${l}</div></div>`
    ).join('') + '</div>';
}

// isolate sections: one failed fetch must not blank the whole page
async function sect(fn) { try { await fn(); } catch (e) { console.error(e); } }

(async () => {
  document.querySelectorAll('.origin').forEach(el =>
    el.textContent = location.origin);
  await sect(async () => {
  const s = await j('/api/stats');
  document.getElementById('stats').innerHTML =
    statCards(s, 'saboteur') + statCards(s, 'dilemma');
  });

  await sect(async () => {
  for (const game of ['saboteur', 'dilemma']) {
    const lb = await j('/api/leaderboard?game=' + game);
    document.querySelector('#lb-' + game + ' tbody').innerHTML = lb.map(r => {
      const common = `<td>${esc(r.name)}</td><td><b>${r.rating}</b></td>
        <td>${r.mu.toFixed(2)}</td><td>${r.sigma.toFixed(2)}</td>
        <td>${r.matches.toLocaleString()}</td><td>${r.win_pct}%</td>`;
      const extra = game === 'saboteur'
        ? `<td>${r.loyalist_win_pct}%</td><td>${r.saboteur_win_pct}%</td>`
        : `<td>${r.draw_pct}%</td><td>${r.pts_per_round}</td><td>${r.coop_pct}%</td>`;
      return `<tr>${common}${extra}</tr>`;
    }).join('');
  }
  });

  await sect(async () => {
  const subs = await j('/api/submissions');
  if (subs.length) {
    document.getElementById('subsWrap').style.display = '';
    document.querySelector('#subs tbody').innerHTML = subs.map(r => {
      const det = r.game === 'saboteur'
        ? `loyal ${r.loyalist_win_pct}% / sab ${r.saboteur_win_pct}%`
        : `${r.pts_per_round} pts/rd, coop ${r.coop_pct}%`;
      return `<tr><td>${esc(r.name)}</td><td>${esc(r.game)}</td>
        <td>${r.matches}</td><td><b>${r.win_pct}%</b></td><td>${det}</td>
        <td>${r.faults}</td><td>${r.avg_move_ms}</td></tr>`;
    }).join('');
  }
  });

  await sect(async () => {
  // fetch per game so both games' replays are always reachable
  const ms = (await Promise.all(['saboteur', 'dilemma'].map(g =>
    j('/api/matches?game=' + g + '&limit=10')))).flat();
  document.querySelector('#mt tbody').innerHTML = ms.map(m => {
    const score = m.game === 'saboteur'
      ? `${m.round_wins.loyalist}&ndash;${m.round_wins.saboteur}`
      : `${m.round_wins.seat0}&ndash;${m.round_wins.seat1}`;
    return `<tr><td><a href="#" onclick="showMatch(${m.id});return false">#${m.id}</a>
      &nbsp;${m.lineup.map(esc).join(', ')}</td>
      <td><span class="tag">${esc(m.game)}</span></td><td>${m.seed}</td>
      <td class="win-${m.winner}">${m.winner}</td><td>${score}</td>
      <td>${m.hash.slice(0, 16)}&hellip;</td></tr>`;
  }).join('');
  });
})();

function renderSaboteur(m) {
  const name = p => `${esc(m.lineup[p])}[${p}${m.roles[p]=='saboteur'?'*':''}]`;
  return m.transcript.map(e => {
    if (e.t == 'propose') return `<b>R${e.round}.${e.attempt}</b> leader ${name(e.leader)} proposes team [${e.team.map(name).join(', ')}]`;
    if (e.t == 'discuss') { const parts = []; if (e.accuse != null) parts.push('accuses ' + name(e.accuse)); if (e.vouch != null) parts.push('vouches ' + name(e.vouch)); return `${name(e.player)} ${parts.length ? parts.join(', ') : 'says nothing'}`; }
    if (e.t == 'vote') return `vote: ${e.votes.map((v, i) => name(i) + (v ? '+' : '-')).join(' ')}`;
    if (e.t == 'auto_approve') return `<b>5th proposal auto-approved</b>`;
    if (e.t == 'mission') return `<b>mission R${e.round}</b>: ${e.sabotages} sabotage card(s) &rarr; <span class="win-${e.round_winner}">${e.round_winner}s win round</span>`;
    if (e.t == 'fault') return `<b>FAULT</b> seat ${e.seat} in ${e.phase}: ${esc(e.reason)} (contained)`;
    return esc(JSON.stringify(e));
  });
}

function renderDilemma(m) {
  const lines = [];
  const rounds = m.transcript.filter(e => e.t == 'round');
  const res = m.transcript.find(e => e.t == 'result');
  lines.push(`<b>${rounds.length} rounds</b> &mdash; C = cooperate, ` +
    `D = defect, * = noise flip; each cell is seat0/seat1`);
  for (let i = 0; i < rounds.length; i += 25) {
    lines.push(rounds.slice(i, i + 25).map(e =>
      (e.actions[0] ? 'C' : 'D') + (e.flipped[0] ? '*' : '') +
      (e.actions[1] ? 'c' : 'd') + (e.flipped[1] ? '*' : '')
    ).join(' '));
  }
  for (const e of m.transcript.filter(e => e.t == 'fault'))
    lines.push(`<b>FAULT</b> seat ${e.seat} round ${e.round}: ${esc(e.reason)} (contained)`);
  if (res) lines.push(`<b>final</b>: ${esc(m.lineup[0])} ${res.scores[0]} ` +
    `&mdash; ${res.scores[1]} ${esc(m.lineup[1])}`);
  return lines;
}

async function showMatch(id) {
  const m = await j('/api/matches/' + id);
  const lines = m.game == 'saboteur' ? renderSaboteur(m) : renderDilemma(m);
  const who = m.winner == 'draw' ? 'draw'
    : m.game == 'saboteur' ? m.winner + 's'
    : esc(m.lineup[Number(m.winner.slice(-1))]);
  document.getElementById('replay').innerHTML =
    `<h2>Match #${id} <span class="tag">${esc(m.game)}</span> &mdash;
     winner: <span class="win-${m.winner}">${who}</span></h2>` +
    lines.map(l => `<div class="ev">${l}</div>`).join('') +
    `<p class="sub">sha256: ${m.hash}</p>`;
  document.getElementById('replay').scrollIntoView({behavior: 'smooth'});
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("{style}", STYLE)
