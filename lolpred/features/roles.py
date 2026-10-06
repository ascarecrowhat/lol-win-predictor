"""Infer lane assignment from champions and player history alone.

The prediction contract is: ten summoners and ten champions, nothing else. Riot's
`teamPosition` is assigned after the game, and Spectator-V5 exposes champion and
team but not lane, so roles must be reconstructed from what is actually knowable
before the game.

Two priors are combined and the best one-to-one assignment of five champions to
five roles is solved exactly (it is a 5x5 linear assignment problem):

* P(role | champion)          - champions are played in characteristic lanes.
* P(role | player)            - a given summoner's own role history.
* P(role | player, champion)  - this summoner on this champion specifically, which
  resolves flex picks: someone who plays Gragas jungle rather than top.

The rest of the team composition is used implicitly but powerfully: because the
assignment is one-to-one, an unambiguous jungler displaces the other four from
jungle, so each champion's role depends on all of its teammates.

Both priors are built from matches strictly before the one being predicted, so
this stays usable inside the causal single pass.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ROLE_INDEX = {r: i for i, r in enumerate(ROLES)}
NEG_INF = -1e6


class RolePrior:
    """Rolling counts of champion-role and player-role frequency."""

    def __init__(
        self,
        champion_weight: float = 1.0,
        player_weight: float = 0.6,
        pair_weight: float = 1.6,
    ):
        self.champ: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(5))
        self.player: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(5))
        self.pair: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros(5))
        self.champion_weight = champion_weight
        self.player_weight = player_weight
        self.pair_weight = pair_weight

    def update(self, records: list[dict]) -> None:
        for r in records:
            role = r.get("position")
            if role not in ROLE_INDEX:
                continue
            idx = ROLE_INDEX[role]
            self.champ[int(r["champion_id"])][idx] += 1
            self.player[r["puuid"]][idx] += 1
            self.pair[(r["puuid"], int(r["champion_id"]))][idx] += 1

    def _log_prob(self, counts: np.ndarray, alpha: float) -> np.ndarray:
        total = counts.sum()
        if total == 0:
            return np.full(5, np.log(0.2))
        probs = (counts + alpha) / (total + 5 * alpha)
        return np.log(probs)

    def score_matrix(self, team: list[dict]) -> np.ndarray:
        """Rows are players, columns are roles; higher is a better fit."""
        scores = np.zeros((len(team), 5))
        for i, r in enumerate(team):
            champ_id = int(r["champion_id"])
            champ_lp = self._log_prob(self.champ.get(champ_id, np.zeros(5)), 0.5)
            player_lp = self._log_prob(self.player.get(r["puuid"], np.zeros(5)), 1.0)
            pair_counts = self.pair.get((r["puuid"], champ_id), np.zeros(5))
            # Heavier shrinkage: this cell is the sparsest of the three.
            pair_lp = self._log_prob(pair_counts, 2.0)
            scores[i] = (
                self.champion_weight * champ_lp
                + self.player_weight * player_lp
                + self.pair_weight * pair_lp
            )
        return scores

    def assign(self, team: list[dict]) -> list[str]:
        """Best one-to-one assignment of this team's five players to five roles."""
        if len(team) != 5:
            return [ROLES[i % 5] for i in range(len(team))]
        scores = self.score_matrix(team)
        rows, cols = linear_sum_assignment(-scores)  # maximise total score
        out = [""] * len(team)
        for r, c in zip(rows, cols):
            out[r] = ROLES[c]
        return out

    def assign_match(self, records: list[dict]) -> dict[str, str]:
        """Infer roles for both teams. Returns puuid -> role."""
        inferred: dict[str, str] = {}
        for team_id in (100, 200):
            team = [r for r in records if int(r["team_id"]) == team_id]
            for r, role in zip(team, self.assign(team)):
                inferred[r["puuid"]] = role
        return inferred
