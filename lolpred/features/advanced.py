"""Playstyle traits, champion scaling, premades and matchup interactions.

Everything here is maintained in the same strictly-causal single pass as the base
features: read before write, so a match never informs its own row.

Design notes earned from measurement rather than assumed:

* Playstyle metrics are stored as deviations from the *current role average*. Raw
  values largely encode which role someone plays (supports ward, carries deal
  damage); residualising keeps the part that is about the player. Measured
  split-half reliability after residualising: vision 0.87, damage taken 0.80,
  team damage share 0.71, damage/min 0.66.
* `summonerLevel` is deliberately not treated as a smurf signal. Fresh accounts
  measurably outperform (797 vs 756 damage/min) yet win at 50.1% - matchmaking
  equalises anything it can observe. The exploitable signals are the ones it
  cannot see: autofill, champion unfamiliarity, fatigue and premades.
* Laning-advantage metrics are excluded as traits (reliability 0.29-0.36): they
  describe the matchup, not the player.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# Metrics kept as player traits, with measured post-residualisation reliability.
STYLE_METRICS = [
    "damage_per_minute",      # 0.66
    "vision_per_min",         # 0.87
    "solo_kills",             # 0.65
    "team_damage_pct",        # 0.71
    "damage_taken_pct",       # 0.80
    "kill_participation",     # 0.44
    "turret_plates",          # 0.41
    "max_cs_adv_lane",        # 0.64
    "time_spent_dead",        # 0.60
]

# Traits compared lane-against-lane rather than only as team aggregates, because
# the direct opponent is where a playstyle clash actually resolves.
LANE_METRICS = ["damage_per_minute", "max_cs_adv_lane", "solo_kills", "damage_taken_pct"]

LONG_GAME_S = 2100   # > 35 min
SHORT_GAME_S = 1500  # < 25 min
SCALING_PRIOR = 40.0


@dataclass
class StyleState:
    """Running mean of each role-residualised trait, plus duration preference."""

    n: int = 0
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    short_games: int = 0
    short_wins: int = 0
    long_games: int = 0
    long_wins: int = 0

    def mean(self, metric: str) -> float:
        c = self.counts.get(metric, 0)
        return self.sums[metric] / c if c else float("nan")

    def duration_preference(self) -> float:
        """Win rate in long games minus win rate in short games, smoothed.

        Positive means this player tends to win games that go late - a scaling
        preference, independent of the champions they pick.
        """
        if self.long_games + self.short_games == 0:
            return 0.0
        long_wr = (self.long_wins + 0.5 * 8) / (self.long_games + 8)
        short_wr = (self.short_wins + 0.5 * 8) / (self.short_games + 8)
        return long_wr - short_wr


class AdvancedState:
    def __init__(self) -> None:
        self.style: dict[str, StyleState] = defaultdict(StyleState)
        # running per-role averages, used to residualise
        self.role_sums: dict[tuple[str, str], float] = defaultdict(float)
        self.role_counts: dict[tuple[str, str], int] = defaultdict(int)
        # (champion, role) -> [short_games, short_wins, long_games, long_wins]
        self.champ_duration: dict[tuple[int, str], list[int]] = defaultdict(
            lambda: [0, 0, 0, 0]
        )
        # how often a pair of players has appeared on the same team before
        self.pairs: dict[tuple[str, str], int] = defaultdict(int)

    # -- read ---------------------------------------------------------------
    def player_style(self, puuid: str) -> dict[str, float]:
        s = self.style.get(puuid)
        if s is None or s.n == 0:
            out = {m: float("nan") for m in STYLE_METRICS}
            out["duration_preference"] = 0.0
            out["style_games"] = 0.0
            return out
        out = {m: s.mean(m) for m in STYLE_METRICS}
        out["duration_preference"] = s.duration_preference()
        out["style_games"] = float(s.n)
        return out

    def champion_scaling(self, champion_id: int, role: str) -> float:
        """Long-game win rate minus short-game win rate for this champion+role.

        Derived from our own matches, so it is population-matched, patch-local and
        carries its own sample size - unlike an external aggregate scraped from a
        stats site, which cannot be made time-safe.
        """
        sg, sw, lg, lw = self.champ_duration.get((champion_id, role), [0, 0, 0, 0])
        if sg + lg == 0:
            return 0.0
        long_wr = (lw + 0.5 * SCALING_PRIOR) / (lg + SCALING_PRIOR)
        short_wr = (sw + 0.5 * SCALING_PRIOR) / (sg + SCALING_PRIOR)
        return long_wr - short_wr

    def premade_pairs(self, puuids: list[str], threshold: int = 2) -> int:
        """How many pairs on this team have played together before.

        Premade status is invisible to the matchmaker, which is what makes it a
        candidate signal rather than something already balanced away.
        """
        n = 0
        for i in range(len(puuids)):
            for j in range(i + 1, len(puuids)):
                key = (puuids[i], puuids[j]) if puuids[i] < puuids[j] else (puuids[j], puuids[i])
                if self.pairs.get(key, 0) >= threshold:
                    n += 1
        return n

    # -- write --------------------------------------------------------------
    def update(self, rows: list[dict], duration_s: int) -> None:
        is_long = duration_s > LONG_GAME_S
        is_short = duration_s < SHORT_GAME_S

        for r in rows:
            puuid, role = r["puuid"], r["position"]
            s = self.style[puuid]
            s.n += 1
            for m in STYLE_METRICS:
                v = r.get(m)
                if v is None or v != v:  # None or NaN
                    continue
                key = (role, m)
                count = self.role_counts[key]
                role_mean = self.role_sums[key] / count if count else v
                s.sums[m] += v - role_mean
                s.counts[m] += 1
                self.role_sums[key] += v
                self.role_counts[key] += 1

            win = 1 if r["win"] else 0
            if is_long:
                s.long_games += 1
                s.long_wins += win
            elif is_short:
                s.short_games += 1
                s.short_wins += win

            cd = self.champ_duration[(r["champion_id"], role)]
            if is_long:
                cd[2] += 1
                cd[3] += win
            elif is_short:
                cd[0] += 1
                cd[1] += win

        for team in (100, 200):
            mates = sorted(r["puuid"] for r in rows if r["team_id"] == team)
            for i in range(len(mates)):
                for j in range(i + 1, len(mates)):
                    self.pairs[(mates[i], mates[j])] += 1


def interaction_terms(
    blue: dict[str, float], red: dict[str, float]
) -> dict[str, float]:
    """Antisymmetric clash terms: A's trait set against B's, minus the mirror.

    A plain product like blue_aggression * red_scaling is not antisymmetric, so it
    would break side-swap augmentation. Taking the difference of the term and its
    mirror keeps the feature valid under a side swap while still expressing "my
    aggression against your scaling".
    """

    def safe(d: dict[str, float], k: str) -> float:
        v = d.get(k, 0.0)
        return 0.0 if v != v else v

    out: dict[str, float] = {}
    b_aggr, r_aggr = safe(blue, "aggression"), safe(red, "aggression")
    b_scale, r_scale = safe(blue, "scaling"), safe(red, "scaling")
    out["x_aggression_vs_scaling"] = b_aggr * r_scale - r_aggr * b_scale
    b_poke, r_poke = safe(blue, "damage_per_minute"), safe(red, "damage_taken_pct")
    out["x_damage_vs_durability"] = (
        b_poke * r_poke - safe(red, "damage_per_minute") * safe(blue, "damage_taken_pct")
    )
    out["x_vision_vs_aggression"] = (
        safe(blue, "vision_per_min") * r_aggr - safe(red, "vision_per_min") * b_aggr
    )
    return out
