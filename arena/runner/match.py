"""Run one complete, deterministic Saboteur match."""

from __future__ import annotations

import random

from arena.game.saboteur import (
    LOYALIST,
    MAX_PROPOSALS,
    N_PLAYERS,
    ROUNDS_TO_WIN,
    SABOTEUR,
    TEAM_SIZES,
    GameState,
    MatchResult,
)
from arena.runner.sandbox import GuardedAgent


def run_match(agents: list, seed: int) -> MatchResult:
    """agents: 5 agent instances (seat order). Deterministic in (agents, seed)."""
    assert len(agents) == N_PLAYERS
    state = GameState(seed)
    fallback_rng = random.Random(seed ^ 0xFA11BACC)
    guarded = [GuardedAgent(a, i) for i, a in enumerate(agents)]
    events = state.public_log  # engine appends; we alias for the transcript

    # Deterministic per-seat private RNG streams.
    for i, a in enumerate(agents):
        a.reset(
            player_id=i,
            role=state.roles[i],
            saboteurs=list(state.saboteurs) if state.roles[i] == SABOTEUR else None,
            rng=random.Random((seed << 3) | i),
        )

    def record_fault(seat: int, phase: str, reason: str) -> None:
        events.append({"t": "fault", "seat": seat, "phase": phase, "reason": reason})

    while max(state.round_wins.values()) < ROUNDS_TO_WIN:
        size = TEAM_SIZES[state.round_num]
        approved_team: list[int] | None = None
        proposer = state.leader

        for attempt in range(MAX_PROPOSALS):
            leader = state.leader
            # 1. Proposal
            team, fault = guarded[leader].call(
                "propose",
                state.observation(leader, "propose", attempt=attempt),
                validate=lambda t, size=size: (
                    isinstance(t, list)
                    and len(t) == size
                    and len(set(t)) == size
                    and all(isinstance(p, int) and 0 <= p < N_PLAYERS for p in t)
                ),
                fallback=lambda size=size: sorted(
                    fallback_rng.sample(range(N_PLAYERS), size)
                ),
            )
            if fault:
                record_fault(leader, "propose", fault)
            team = sorted(team)
            events.append(
                {"t": "propose", "round": state.round_num, "attempt": attempt,
                 "leader": leader, "team": team}
            )

            # 2. Discussion (seat order starting after leader)
            for offset in range(1, N_PLAYERS + 1):
                p = (leader + offset) % N_PLAYERS
                msg, fault = guarded[p].call(
                    "discuss",
                    state.observation(p, "discuss", team=team, attempt=attempt),
                    validate=lambda m: (
                        isinstance(m, dict)
                        and set(m) <= {"accuse", "vouch"}
                        and all(
                            v is None or (isinstance(v, int) and 0 <= v < N_PLAYERS)
                            for v in m.values()
                        )
                    ),
                    fallback=lambda: {"accuse": None, "vouch": None},
                )
                if fault:
                    record_fault(p, "discuss", fault)
                events.append(
                    {"t": "discuss", "player": p,
                     "accuse": msg.get("accuse"), "vouch": msg.get("vouch")}
                )

            # 3. Vote (simultaneous — all observe the same pre-vote log)
            if attempt == MAX_PROPOSALS - 1:
                approved_team = team
                events.append({"t": "auto_approve", "team": team})
            else:
                votes = []
                pre_vote_obs = [
                    state.observation(p, "vote", team=team, attempt=attempt)
                    for p in range(N_PLAYERS)
                ]
                for p in range(N_PLAYERS):
                    v, fault = guarded[p].call(
                        "vote",
                        pre_vote_obs[p],
                        validate=lambda v: isinstance(v, bool),
                        fallback=lambda: bool(fallback_rng.getrandbits(1)),
                    )
                    if fault:
                        record_fault(p, "vote", fault)
                    votes.append(v)
                events.append({"t": "vote", "team": team, "votes": votes})
                if sum(votes) * 2 > N_PLAYERS:
                    approved_team = team

            state.leader = (state.leader + 1) % N_PLAYERS
            if approved_team is not None:
                proposer = leader  # obs "leader" during the mission
                break

        # 4. Mission — team members play secretly; loyalists can't sabotage.
        n_sab = 0
        for p in approved_team:
            card, fault = guarded[p].call(
                "mission",
                state.observation(p, "mission", team=approved_team,
                                  leader=proposer),
                validate=lambda c: isinstance(c, bool),
                fallback=lambda: True,
            )
            if fault:
                record_fault(p, "mission", fault)
            if state.roles[p] == LOYALIST:
                card = True
            if not card:
                n_sab += 1
        round_winner = SABOTEUR if n_sab > 0 else LOYALIST
        state.round_wins[round_winner] += 1
        events.append(
            {"t": "mission", "round": state.round_num, "team": approved_team,
             "sabotages": n_sab, "round_winner": round_winner}
        )
        state.round_num += 1

    winner = max(state.round_wins, key=state.round_wins.get)
    return MatchResult(
        seed=seed,
        agent_names=[g.agent.name for g in guarded],
        roles=list(state.roles),
        winner=winner,
        round_wins=dict(state.round_wins),
        events=events,
        faults={g.seat: g.faults for g in guarded if g.faults},
        move_time_s={g.seat: round(g.time_spent, 6) for g in guarded},
        n_moves=sum(g.n_moves for g in guarded),
    )
