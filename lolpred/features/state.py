"""Rolling, strictly-causal player and champion state.

The leakage rule is enforced by construction rather than by discipline: the whole
feature set is produced in a single pass over matches ordered by start time, and
for each match we *read* the state first and *write* it afterwards. A match can
therefore never contribute to its own features, which is the bug that produces
fake 80-90% accuracy in this problem.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

# Smoothing: a rate seen over few games is pulled toward a prior, so a 3-0 record
# reads as ~0.6 rather than 1.0.
PRIOR_WEIGHT = 10.0
CHAMP_PRIOR_WEIGHT = 6.0
ROLE_PRIOR_WEIGHT = 8.0
RECENT_WINDOW = 10
DAY_SECONDS = 86400.0


def smoothed(wins: float, games: float, prior: float, weight: float) -> float:
    return (wins + prior * weight) / (games + weight)


@dataclass
class PlayerState:
    games: int = 0
    wins: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    champ: dict[int, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    role: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    sum_cspm: float = 0.0
    sum_gpm: float = 0.0
    sum_kda: float = 0.0
    sum_dmg_share: float = 0.0
    sum_vpm: float = 0.0
    last_ts: float = 0.0
    recent_ts: deque = field(default_factory=lambda: deque(maxlen=40))

    def main_role(self) -> str | None:
        if not self.role:
            return None
        return max(self.role.items(), key=lambda kv: kv[1][0])[0]

    def games_since(self, ts: float, window: float = DAY_SECONDS) -> int:
        return sum(1 for t in self.recent_ts if ts - t <= window)


class FeatureState:
    """Holds every rolling aggregate needed to describe a player before a match."""

    def __init__(self) -> None:
        self.players: dict[str, PlayerState] = defaultdict(PlayerState)
        # (patch, champion, role) -> [games, wins]
        self.champ_role: dict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])
        self.global_games = 0
        self.global_wins = 0

    # -- read ---------------------------------------------------------------
    def player_features(
        self, puuid: str, champion_id: int, role: str, ts: float, patch: str
    ) -> dict[str, float]:
        p = self.players.get(puuid)
        if p is None or p.games == 0:
            # No history at all: everything falls back to the prior, and
            # `games` lets the model learn how much to distrust the row.
            return {
                "games": 0.0, "winrate": 0.5, "recent_winrate": 0.5,
                "champ_games": 0.0, "champ_winrate": 0.5, "first_time_champ": 1.0,
                "role_games": 0.0, "role_winrate": 0.5, "off_role": 0.0,
                "cs_per_min": float("nan"), "gold_per_min": float("nan"),
                "kda": float("nan"), "dmg_share": float("nan"),
                "vision_per_min": float("nan"),
                "days_since_last": float("nan"), "games_today": 0.0,
                "champ_role_winrate": self._champ_role_wr(patch, champion_id, role),
            }

        base_wr = smoothed(p.wins, p.games, 0.5, PRIOR_WEIGHT)
        cg, cw = p.champ.get(champion_id, [0, 0])
        rg, rw = p.role.get(role, [0, 0])
        main = p.main_role()
        n = float(p.games)

        return {
            "games": n,
            "winrate": base_wr,
            "recent_winrate": (
                smoothed(sum(p.recent), len(p.recent), base_wr, 5.0) if p.recent else base_wr
            ),
            "champ_games": float(cg),
            # champion win rate is shrunk toward the player's own overall rate,
            # not toward 0.5 - a strong player on a new champion is still strong
            "champ_winrate": smoothed(cw, cg, base_wr, CHAMP_PRIOR_WEIGHT),
            "first_time_champ": 1.0 if cg == 0 else 0.0,
            "role_games": float(rg),
            "role_winrate": smoothed(rw, rg, base_wr, ROLE_PRIOR_WEIGHT),
            "off_role": 1.0 if (main is not None and p.games >= 5 and role != main) else 0.0,
            "cs_per_min": p.sum_cspm / n,
            "gold_per_min": p.sum_gpm / n,
            "kda": p.sum_kda / n,
            "dmg_share": p.sum_dmg_share / n,
            "vision_per_min": p.sum_vpm / n,
            "days_since_last": (ts - p.last_ts) / DAY_SECONDS if p.last_ts else float("nan"),
            "games_today": float(p.games_since(ts)),
            "champ_role_winrate": self._champ_role_wr(patch, champion_id, role),
        }

    def _champ_role_wr(self, patch: str, champion_id: int, role: str) -> float:
        games, wins = self.champ_role.get((patch, champion_id, role), [0, 0])
        return smoothed(wins, games, 0.5, 50.0)

    def history_depth(self, puuid: str) -> int:
        p = self.players.get(puuid)
        return p.games if p else 0

    # -- write --------------------------------------------------------------
    def update(self, rows: list[dict], ts: float, patch: str, duration_s: int) -> None:
        """Fold one finished match into the state. Call *after* reading features."""
        minutes = max(duration_s, 1) / 60.0
        team_damage = {100: 0.0, 200: 0.0}
        for r in rows:
            team_damage[r["team_id"]] = team_damage.get(r["team_id"], 0.0) + r["dmg_champs"]

        for r in rows:
            p = self.players[r["puuid"]]
            win = 1 if r["win"] else 0
            p.games += 1
            p.wins += win
            p.recent.append(win)
            champ = p.champ[r["champion_id"]]
            champ[0] += 1
            champ[1] += win
            role = p.role[r["position"]]
            role[0] += 1
            role[1] += win
            p.sum_cspm += r["cs"] / minutes
            p.sum_gpm += r["gold_earned"] / minutes
            p.sum_kda += (r["kills"] + r["assists"]) / max(r["deaths"], 1)
            denom = team_damage.get(r["team_id"], 0.0)
            p.sum_dmg_share += (r["dmg_champs"] / denom) if denom > 0 else 0.2
            p.sum_vpm += r["vision_score"] / minutes
            p.last_ts = ts
            p.recent_ts.append(ts)

            cr = self.champ_role[(patch, r["champion_id"], r["position"])]
            cr[0] += 1
            cr[1] += win

        self.global_games += 1
        self.global_wins += 1 if rows[0]["win"] and rows[0]["team_id"] == 100 else 0
