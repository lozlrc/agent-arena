"""Skill ratings over match results — game-agnostic.

Each finished match is mapped to openskill teams/ranks by its GameDef
(Saboteur: loyalists-vs-saboteurs 3v2; Dilemma: 1v1 with draws) and
rated under the Plackett-Luce model. Because seeded schedules balance
seats and roles over many matches, ratings converge to role-balanced
skill estimates. Displayed rating is the conservative mu - 3*sigma.

Leaderboard extras (per-role win rates, points per round, cooperation
rate, ...) come from the GameDef tally hooks.
"""

from __future__ import annotations

from openskill.models import PlackettLuce

from arena.games import GAMES


def compute_ratings(results: list[dict], game: str) -> list[dict]:
    gd = GAMES[game]
    model = PlackettLuce()
    ratings: dict[str, object] = {}
    tallies: dict[str, dict] = {}
    accs: dict[str, dict] = {}

    for r in results:
        lineup = r["lineup"]
        # openskill can't take the same rating object twice in one
        # match; seeded default schedules always produce distinct
        # lineups, but custom pools may not — skip those updates.
        if len(set(lineup)) == len(lineup):
            teams_idx, ranks = gd.rating_teams(r)
            for name in lineup:
                ratings.setdefault(name, model.rating(name=name))
            teams = [[ratings[lineup[i]] for i in team] for team in teams_idx]
            new = model.rate(teams, ranks=ranks)
            for team_ratings, team in zip(new, teams_idx):
                for rating, i in zip(team_ratings, team):
                    ratings[lineup[i]] = rating

        for seat, name in enumerate(lineup):
            t = tallies.setdefault(name, {"matches": 0, "wins": 0})
            t["matches"] += 1
            t["wins"] += gd.seat_won(r, seat)
            gd.tally_update(accs.setdefault(name, gd.tally_init()), r, seat)

    rows = []
    for name, t in tallies.items():
        rt = ratings.get(name)
        rows.append(
            {"name": name,
             "mu": rt.mu if rt else 25.0,
             "sigma": rt.sigma if rt else 25.0 / 3,
             "matches": t["matches"],
             "wins": t["wins"],
             "extra": gd.tally_final(accs[name])}
        )
    rows.sort(key=lambda x: x["mu"] - 3 * x["sigma"], reverse=True)
    return rows
