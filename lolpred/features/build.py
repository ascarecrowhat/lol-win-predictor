"""Single-pass, time-ordered feature construction.

Produces one row per match: blue-minus-red differences of team aggregates, plus
coverage counts so the model can learn how far to trust each row. The same code
serves training and inference - at serve time the state is primed from history
fetched on demand for the ten players in question.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..storage.db import Store
from .advanced import (
    LANE_METRICS,
    STYLE_METRICS,
    AdvancedState,
    interaction_terms,
)
from .roles import RolePrior
from .state import FeatureState

log = logging.getLogger(__name__)

# Per-player features aggregated into team numbers.
PLAYER_FEATURES = [
    "games", "winrate", "recent_winrate", "champ_games", "champ_winrate",
    "first_time_champ", "role_games", "role_winrate", "off_role",
    "cs_per_min", "gold_per_min", "kda", "dmg_share", "vision_per_min",
    "days_since_last", "games_today", "champ_role_winrate",
]
# min/max add little for flags and counters; the weakest link matters for skill.
AGGS = ("mean", "min", "max")
NO_MINMAX = {"first_time_champ", "off_role", "games_today"}


def feature_columns() -> list[str]:
    cols: list[str] = []
    for f in PLAYER_FEATURES:
        for a in AGGS:
            if a != "mean" and f in NO_MINMAX:
                continue
            cols.append(f"d_{f}_{a}")
    cols += ["d_coverage", "blue_coverage", "red_coverage", "min_coverage"]
    return cols


def _aggregate(values: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            out[f"{f}_mean"] = np.nan
            out[f"{f}_min"] = np.nan
            out[f"{f}_max"] = np.nan
            continue
        out[f"{f}_mean"] = float(valid.mean())
        out[f"{f}_min"] = float(valid.min())
        out[f"{f}_max"] = float(valid.max())
    return out


ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _aggression(style: dict[str, float]) -> float:
    """Composite early-aggression axis from the traits that proved stable."""
    parts = [style.get(k) for k in ("solo_kills", "turret_plates", "max_cs_adv_lane")]
    vals = [v for v in parts if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def build(
    store: Store,
    min_history: int = 10,
    require_coverage: int = 0,
    limit: int | None = None,
    advanced: bool = True,
    infer_roles: bool = True,
    save_state: str | None = None,
) -> pd.DataFrame:
    """Walk every stored match in time order and emit a feature row for each.

    `min_history` is how many prior games a player needs to count as "covered";
    `require_coverage` drops rows where fewer than that many of the ten players
    are covered. Rows are still *built* from every match, so a dropped match
    still contributes to the state of later ones.
    """
    has_extras = bool(
        store.con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'participant_extras'"
        ).fetchone()[0]
    )
    use_advanced = advanced and has_extras
    if advanced and not has_extras:
        log.warning("participant_extras missing - run `extras` first; advanced features off")

    extra_cols = ", ".join(f"e.{m}" for m in STYLE_METRICS)
    sql = f"""
        SELECT m.match_id, m.game_start, m.patch, m.duration_s, m.winner,
               p.puuid, p.team_id, p.position, p.champion_id, p.win,
               p.kills, p.deaths, p.assists, p.gold_earned, p.cs,
               p.dmg_champs, p.vision_score
               {", " + extra_cols if use_advanced else ""}
        FROM matches m JOIN participants p ON p.match_id = m.match_id
        {"LEFT JOIN participant_extras e ON e.match_id = p.match_id AND e.puuid = p.puuid"
         if use_advanced else ""}
        ORDER BY m.game_start, m.match_id
    """
    frame = store.con.execute(sql).df()
    if frame.empty:
        return pd.DataFrame(columns=["match_id", "blue_win", *feature_columns()])

    frame["ts"] = frame.game_start.astype("int64") / 1e9
    log.info("Building features from %d matches", frame.match_id.nunique())

    state = FeatureState()
    adv = AdvancedState()
    # The prediction contract is ten summoners and ten champions, so lane must be
    # inferred rather than read from the post-game `teamPosition`. The prior is
    # updated with the true roles of *finished* matches, which is legitimate: only
    # the match being predicted needs inference.
    role_prior = RolePrior() if infer_roles else None
    rows: list[dict] = []
    skipped = 0

    for match_id, grp in frame.groupby("match_id", sort=False):
        if len(grp) != 10:
            skipped += 1
            continue
        ts = float(grp.ts.iloc[0])
        patch = grp.patch.iloc[0]
        duration = int(grp.duration_s.iloc[0])
        records = grp.to_dict("records")
        true_positions = [r["position"] for r in records]
        if role_prior is not None:
            inferred = role_prior.assign_match(records)
            for r in records:
                r["position"] = inferred.get(r["puuid"], r["position"])

        # ---- READ state (strictly prior matches only) ----
        per_team: dict[int, dict[str, list[float]]] = {
            100: {f: [] for f in PLAYER_FEATURES},
            200: {f: [] for f in PLAYER_FEATURES},
        }
        coverage = {100: 0, 200: 0}
        for r in records:
            feats = state.player_features(
                r["puuid"], int(r["champion_id"]), r["position"], ts, patch
            )
            team = int(r["team_id"])
            for f in PLAYER_FEATURES:
                per_team[team][f].append(feats[f])
            if state.history_depth(r["puuid"]) >= min_history:
                coverage[team] += 1

        blue = _aggregate(per_team[100])
        red = _aggregate(per_team[200])

        advanced_row: dict[str, float] = {}
        if use_advanced:
            advanced_row = _advanced_features(adv, records)

        row: dict[str, float | str] = {
            "match_id": match_id,
            "game_start": grp.game_start.iloc[0],
            "patch": patch,
            "blue_win": 1 if int(grp.winner.iloc[0]) == 100 else 0,
        }
        for f in PLAYER_FEATURES:
            for a in AGGS:
                if a != "mean" and f in NO_MINMAX:
                    continue
                row[f"d_{f}_{a}"] = blue[f"{f}_{a}"] - red[f"{f}_{a}"]
        row["blue_coverage"] = coverage[100]
        row["red_coverage"] = coverage[200]
        row["d_coverage"] = coverage[100] - coverage[200]
        row["min_coverage"] = min(coverage[100], coverage[200])

        row.update(advanced_row)

        if coverage[100] + coverage[200] >= require_coverage:
            rows.append(row)

        # ---- WRITE state (after reading: no leakage) ----
        # State is folded in using the real post-game roles, which are known once
        # a match is over; only the prediction itself relies on inference.
        for r, truth in zip(records, true_positions):
            r["position"] = truth
        if role_prior is not None:
            role_prior.update(records)
        state.update(records, ts, patch, duration)
        if use_advanced:
            adv.update(records, duration)

        if limit and len(rows) >= limit:
            break

    if save_state:
        from .snapshot import export_globals, save
        latest_patch = (
            frame.patch.dropna().iloc[-1] if frame.patch.notna().any() else None
        )
        path = save(
            save_state,
            export_globals(
                state, adv, role_prior or RolePrior(),
                {
                    "matches": int(frame.match_id.nunique()),
                    "latest_patch": latest_patch,
                    "built_from": str(store.path),
                },
            ),
        )
        log.info("Saved serving state snapshot -> %s", path)

    if skipped:
        log.warning("Skipped %d matches without exactly 10 participants", skipped)
    out = pd.DataFrame(rows)
    log.info("Built %d feature rows, %d columns", len(out), len(out.columns))
    return out


def _advanced_features(adv: AdvancedState, records: list[dict]) -> dict[str, float]:
    """Playstyle aggregates, lane-versus-lane pairings, premades and clash terms."""
    out: dict[str, float] = {}
    styles = {int(r["team_id"]): [] for r in records}
    by_role: dict[tuple[int, str], dict] = {}
    champ_scaling: dict[int, list[float]] = {100: [], 200: []}

    for r in records:
        team, role = int(r["team_id"]), r["position"]
        style = adv.player_style(r["puuid"])
        styles[team].append(style)
        by_role[(team, role)] = style
        champ_scaling[team].append(adv.champion_scaling(int(r["champion_id"]), role))

    # team aggregates of each residualised trait, as blue-minus-red
    for metric in STYLE_METRICS + ["duration_preference", "style_games"]:
        agg = {}
        for team in (100, 200):
            vals = [s.get(metric) for s in styles[team]]
            vals = [v for v in vals if v is not None and v == v]
            agg[team] = float(np.mean(vals)) if vals else np.nan
        out[f"d_style_{metric}_mean"] = agg[100] - agg[200]

    # champion scaling: how far each draft leans late-game
    out["d_champ_scaling_mean"] = float(np.mean(champ_scaling[100])) - float(
        np.mean(champ_scaling[200])
    )

    # lane versus lane: the direct opponent, not a team average
    for role in ROLES:
        b, r_ = by_role.get((100, role)), by_role.get((200, role))
        for metric in LANE_METRICS:
            key = f"d_lane_{role.lower()}_{metric}"
            if not b or not r_:
                out[key] = np.nan
                continue
            bv, rv = b.get(metric), r_.get(metric)
            out[key] = (
                bv - rv if bv is not None and rv is not None and bv == bv and rv == rv
                else np.nan
            )

    # premades, which the matchmaker cannot see
    teams = {t: [r["puuid"] for r in records if int(r["team_id"]) == t] for t in (100, 200)}
    out["d_premade_pairs"] = adv.premade_pairs(teams[100]) - adv.premade_pairs(teams[200])

    # explicit clash terms, kept antisymmetric so side-swapping stays valid
    summary = {}
    for team, label in ((100, "blue"), (200, "red")):
        vals = styles[team]
        summary[label] = {
            "aggression": float(np.nanmean([_aggression(s) for s in vals])),
            "scaling": float(np.nanmean([s.get("duration_preference", 0.0) for s in vals])),
            "damage_per_minute": float(
                np.nanmean([s.get("damage_per_minute", np.nan) for s in vals])
            ),
            "damage_taken_pct": float(
                np.nanmean([s.get("damage_taken_pct", np.nan) for s in vals])
            ),
            "vision_per_min": float(
                np.nanmean([s.get("vision_per_min", np.nan) for s in vals])
            ),
        }
    out.update(interaction_terms(summary["blue"], summary["red"]))
    return out


def augment_side_swap(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror every row by swapping blue and red.

    Every feature is a blue-minus-red difference, so swapping sides is exactly a
    sign flip plus a label flip. This doubles the training set and stops the model
    from learning side advantage as a shortcut.
    """
    diff_cols = [c for c in frame.columns if c.startswith("d_")]
    mirrored = frame.copy()
    mirrored[diff_cols] = -mirrored[diff_cols]
    mirrored["blue_win"] = 1 - mirrored["blue_win"]
    mirrored[["blue_coverage", "red_coverage"]] = mirrored[
        ["red_coverage", "blue_coverage"]
    ].to_numpy()
    mirrored["match_id"] = mirrored["match_id"].astype(str) + "_swap"
    return pd.concat([frame, mirrored], ignore_index=True)
