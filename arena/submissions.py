"""Remote agent submissions: validation, loading, and evaluation.

Untrusted code goes through three layers before and during execution:

  1. AST screening (parent process): imports outside a small stdlib
     allowlist, dunder attribute access, and dangerous builtins
     (exec/eval/open/getattr/...) are rejected before anything runs.
  2. Restricted execution environment: the code is exec'd with an
     allowlisted __builtins__ and a guarded __import__.
  3. Process isolation (see web/app.py): compilation + matches happen
     in a disposable worker process — per-move CPU-time preemption via
     the referee sandbox, and the parent hard-kills the whole worker
     on wall-clock overrun. The web process never runs submitted code.

This is defense-in-depth for a demo, not a hard security boundary;
a production deployment would run each submission in its own
container/microVM.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import random
import re

from arena.games import GAMES
from arena.runner.orchestrator import play_with_agents

ALLOWED_IMPORTS = {"math", "random", "itertools", "collections",
                   "statistics", "functools", "heapq", "bisect"}
BANNED_NAMES = {
    "exec", "eval", "open", "compile", "input", "breakpoint", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "type",
    "super", "object", "memoryview", "classmethod", "staticmethod", "help",
    "exit", "quit",
}
SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "hash", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "pow", "print", "range",
    "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
    "zip", "ValueError", "TypeError", "KeyError", "IndexError",
    "StopIteration", "Exception", "True", "False", "None",
]
MAX_CODE_BYTES = 20_000


def _guarded_import(name, *args, **kwargs):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, *args, **kwargs)


def validate_code(code: str) -> list[str]:
    """Static screen. Returns a list of violations (empty = pass)."""
    if len(code.encode()) > MAX_CODE_BYTES:
        return [f"code exceeds {MAX_CODE_BYTES} bytes"]
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    errs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (node.module if isinstance(node, ast.ImportFrom)
                   else node.names[0].name) or ""
            if mod.split(".")[0] not in ALLOWED_IMPORTS:
                errs.append(f"line {node.lineno}: import {mod!r} not allowed "
                            f"(allowed: {sorted(ALLOWED_IMPORTS)})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errs.append(f"line {node.lineno}: access to {node.attr!r} "
                        "(underscore attributes) not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"line {node.lineno}: {node.id!r} not allowed")
    return errs


def load_agent(code: str, name: str, game: str):
    """Exec screened code in a restricted namespace; return the class."""
    gd = GAMES[game]
    safe_builtins = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES
                     if hasattr(builtins, n)}
    safe_builtins["__import__"] = _guarded_import
    safe_builtins["__build_class__"] = builtins.__build_class__
    ns = {"__builtins__": safe_builtins, "__name__": "submission",
          gd.base_class_name: gd.base_class}
    exec(compile(code, "<submission>", "exec"), ns)  # noqa: S102 — sandboxed
    classes = [v for v in ns.values()
               if isinstance(v, type) and issubclass(v, gd.base_class)
               and v is not gd.base_class]
    if len(classes) != 1:
        raise ValueError(
            f"submission must define exactly one {gd.base_class_name} "
            f"subclass for game {game!r}, found {len(classes)}")
    cls = classes[0]
    cls.name = name
    return cls


def evaluate_code(code: str, name: str, n_matches: int, seed: int,
                  game: str = "saboteur") -> dict:
    """Run the submitted agent against the game's baseline pool.

    Deterministic in (code, name, n_matches, seed, game). Designed to
    run in a worker process (signal-based sandboxing needs a main
    thread; the parent enforces a wall-clock kill on top).
    """
    gd = GAMES[game]
    cls = load_agent(code, name, game)
    rng = random.Random(seed)
    wins = 0
    faults = 0
    moves = 0
    time_spent = 0.0
    hashes = []
    acc = gd.tally_init()
    sample = None

    for i in range(n_matches):
        lineup = gd.make_lineup(rng, gd.ranked)
        seat = rng.randrange(gd.n_seats)
        agents = [gd.registry[n]() for n in lineup]
        agents[seat] = cls()
        r = play_with_agents(game, agents, rng.getrandbits(48),
                             keep_transcript=(i == 0))
        wins += gd.seat_won(r, seat)
        gd.tally_update(acc, r, seat)
        faults += r["faults"].get(seat, 0)
        time_spent += r["move_time_s"].get(seat, 0.0)
        moves += r["n_moves"] // gd.n_seats
        hashes.append(r["hash"])
        if i == 0:
            sample = {"lineup": r["lineup"], "roles": r["roles"],
                      "winner": r["winner"], "seat": seat,
                      "events": r["events"]}

    return {
        "name": name,
        "game": game,
        "matches": n_matches,
        "seed": seed,
        "win_pct": round(100 * wins / n_matches, 1),
        **gd.tally_final(acc),
        "faults": faults,
        "avg_move_ms": round(1000 * time_spent / max(1, moves), 3),
        "eval_hash": hashlib.sha256("".join(hashes).encode()).hexdigest(),
        "sample_match": sample,
    }


def clean_name(raw: str | None) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "", raw or "")[:24]
    return name or "submitted"
