"""Baseline agents.

Every agent must play both roles: the same policy gets assigned
loyalist or saboteur per match. The interface (see BaseAgent) is what
student submissions implement — four methods, dict observations in,
plain Python values out. Agents must only use the RNG handed to
`reset` so matches stay deterministic.
"""

from __future__ import annotations

import random

from arena.game.saboteur import LOYALIST, N_PLAYERS, SABOTEUR


class BaseAgent:
    name = "base"

    def reset(self, player_id: int, role: str, saboteurs: list[int] | None,
              rng: random.Random) -> None:
        self.me = player_id
        self.role = role
        self.saboteurs = saboteurs  # None unless we are a saboteur
        self.rng = rng

    def propose(self, obs: dict) -> list[int]:
        others = [p for p in range(N_PLAYERS) if p != self.me]
        return sorted([self.me] + self.rng.sample(others, obs["team_size"] - 1))

    def discuss(self, obs: dict) -> dict:
        return {"accuse": None, "vouch": None}

    def vote(self, obs: dict) -> bool:
        return True

    def mission(self, obs: dict) -> bool:
        return True


class RandomAgent(BaseAgent):
    """Uniformly random valid actions. The floor of the leaderboard."""
    name = "random"

    def propose(self, obs):
        return sorted(self.rng.sample(range(N_PLAYERS), obs["team_size"]))

    def discuss(self, obs):
        return {"accuse": self.rng.randrange(N_PLAYERS), "vouch": None}

    def vote(self, obs):
        return bool(self.rng.getrandbits(1))

    def mission(self, obs):
        return self.role == LOYALIST or bool(self.rng.getrandbits(1))


class NaiveTruster(BaseAgent):
    """Approves everything, never suspects anyone, always sabotages if evil."""
    name = "naive_truster"

    def mission(self, obs):
        return self.role == LOYALIST


class SuspicionAgent(BaseAgent):
    """Evidence-based loyalist play + deceptive saboteur play.

    Tracks per-player suspicion from failed missions (everyone on a
    sabotaged mission gains suspicion, split by sabotage count) and from
    voting/proposal association. As a saboteur, it mirrors the honest
    policy outwardly but deflects: accuses trusted loyalists, avoids
    accusing its partner, and sabotages only when the mission would
    otherwise succeed quietly (not on the first round, where exposure
    is costliest relative to information gained).
    """
    name = "suspicion"

    def _scores(self, obs) -> list[float]:
        s = [0.0] * N_PLAYERS
        team_of_attempt: list[int] = []
        proposer = None
        for ev in obs["public_log"]:
            if ev["t"] == "propose":
                team_of_attempt = ev["team"]
                proposer = ev["leader"]
            elif ev["t"] == "mission" and ev["sabotages"] > 0:
                for p in ev["team"]:
                    s[p] += ev["sabotages"] / len(ev["team"])
                if proposer is not None and proposer not in ev["team"]:
                    s[proposer] += 0.3  # proposed a team that failed
            elif ev["t"] == "vote":
                # Approving a team that later fails is weak evidence,
                # handled implicitly; rejecting obviously-clean teams is noise.
                pass
        s[self.me] = -1.0  # we know ourselves
        if self.saboteurs:
            for p in self.saboteurs:
                s[p] = -1.0  # (saboteur) don't target partners via this score
        return s

    def _least_suspect(self, obs, k: int, include_me: bool = True) -> list[int]:
        s = self._scores(obs)
        pool = sorted(range(N_PLAYERS), key=lambda p: (s[p], p))
        team = [self.me] if include_me else []
        for p in pool:
            if len(team) >= k:
                break
            if p not in team:
                team.append(p)
        return sorted(team[:k])

    def propose(self, obs):
        if self.role == SABOTEUR:
            # Look reasonable but ensure exactly one saboteur (me) on team.
            s = self._scores(obs)
            clean = sorted(
                (p for p in range(N_PLAYERS)
                 if p != self.me and p not in self.saboteurs),
                key=lambda p: (s[p], p),
            )
            return sorted([self.me] + clean[: obs["team_size"] - 1])
        return self._least_suspect(obs, obs["team_size"])

    def discuss(self, obs):
        s = self._scores(obs)
        if self.role == SABOTEUR:
            # Deflect onto the most-trusted loyalist; vouch for partner
            # only while partner looks clean (vouching for a suspect burns us).
            loyal = [p for p in range(N_PLAYERS)
                     if p != self.me and p not in self.saboteurs]
            target = min(loyal, key=lambda p: (s[p], p))
            partner = next(p for p in self.saboteurs if p != self.me)
            partner_clean = self._scores(obs)  # partner score from public view
            vouch = partner if partner_clean[partner] <= 0.5 else None
            return {"accuse": target, "vouch": vouch}
        top = max(range(N_PLAYERS), key=lambda p: s[p])
        return {"accuse": top if s[top] > 0.4 else None, "vouch": None}

    def vote(self, obs):
        team = obs["team"]
        if self.role == SABOTEUR:
            n_evil = sum(1 for p in team if p in self.saboteurs)
            # Approve teams containing a saboteur; reject all-clean teams
            # unless rejecting looks too suspicious (late attempts).
            return n_evil > 0 or obs["attempt"] >= 3
        s = self._scores(obs)
        threshold = 0.4 + 0.2 * obs["attempt"]  # get less picky as clock runs
        return all(s[p] < threshold for p in team if p != self.me)

    def mission(self, obs):
        if self.role == LOYALIST:
            return True
        n_evil = sum(1 for p in obs["team"] if p in self.saboteurs)
        if obs["round_wins"]["saboteur"] == 2:
            return False  # win now
        if obs["round"] == 0 and n_evil == 1 and len(obs["team"]) == 2:
            return True   # lying low on a 2-man opener is too revealing
        # If both saboteurs are on the team, only the lower id sabotages
        # (double sabotage screams "both of these two are evil").
        if n_evil == 2 and self.me != min(p for p in obs["team"]
                                          if p in self.saboteurs):
            return True
        return False


class AggressiveSaboteur(SuspicionAgent):
    """Same loyalist play, but sabotages every mission it is on."""
    name = "aggressive_sab"

    def mission(self, obs):
        return self.role == LOYALIST


class CautiousVoter(BaseAgent):
    """Rejects any team it isn't on (until forced); always sabotages."""
    name = "cautious_voter"

    def vote(self, obs):
        return self.me in obs["team"] or obs["attempt"] >= 3

    def mission(self, obs):
        return self.role == LOYALIST


class CrashAgent(BaseAgent):
    """Fault-injection: raises on every proposal. For isolation tests."""
    name = "crash_test"

    def propose(self, obs):
        raise RuntimeError("intentional crash")


class SlowAgent(BaseAgent):
    """Fault-injection: infinite-loops on vote. For timeout tests."""
    name = "slow_test"

    def vote(self, obs):
        while True:
            pass


REGISTRY: dict[str, type[BaseAgent]] = {
    a.name: a
    for a in [RandomAgent, NaiveTruster, SuspicionAgent, AggressiveSaboteur,
              CautiousVoter, CrashAgent, SlowAgent]
}

# Agents eligible for ranked play (fault-injection bots excluded).
RANKED = ["random", "naive_truster", "suspicion", "aggressive_sab",
          "cautious_voter"]
