"""SQLite result store: matches, per-game ratings, benchmark stats."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(__file__).resolve().parent.parent / "results" / "arena.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    game TEXT NOT NULL,
    seed INTEGER NOT NULL,
    lineup TEXT NOT NULL,
    roles TEXT NOT NULL,
    winner TEXT NOT NULL,
    round_wins TEXT NOT NULL,
    hash TEXT NOT NULL,
    faults TEXT NOT NULL,
    n_moves INTEGER NOT NULL,
    transcript TEXT
);
CREATE INDEX IF NOT EXISTS idx_matches_game ON matches(game);
CREATE TABLE IF NOT EXISTS ratings (
    game TEXT NOT NULL,
    name TEXT NOT NULL,
    mu REAL NOT NULL,
    sigma REAL NOT NULL,
    matches INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    extra TEXT NOT NULL,
    PRIMARY KEY (game, name)
);
CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def insert_matches(con: sqlite3.Connection, results: list[dict]) -> None:
    con.executemany(
        "INSERT INTO matches (game, seed, lineup, roles, winner, round_wins,"
        " hash, faults, n_moves, transcript) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r["game"],
                r["seed"],
                json.dumps(r["lineup"]),
                json.dumps(r["roles"]),
                r["winner"],
                json.dumps(r["round_wins"]),
                r["hash"],
                json.dumps(r["faults"]),
                r["n_moves"],
                json.dumps(r["events"]) if r.get("events") else None,
            )
            for r in results
        ],
    )
    con.commit()


def save_ratings(con: sqlite3.Connection, game: str,
                 rows: list[dict]) -> None:
    con.execute("DELETE FROM ratings WHERE game = ?", (game,))
    con.executemany(
        "INSERT INTO ratings VALUES (?,?,?,?,?,?,?)",
        [
            (game, r["name"], r["mu"], r["sigma"], r["matches"], r["wins"],
             json.dumps(r["extra"]))
            for r in rows
        ],
    )
    con.commit()


def set_stat(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute(
        "INSERT INTO stats (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value)),
    )
    con.commit()


def get_stats(con: sqlite3.Connection) -> dict[str, Any]:
    return {k: json.loads(v) for k, v in con.execute("SELECT * FROM stats")}
