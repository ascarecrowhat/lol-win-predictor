"""Extract the advanced per-participant metrics out of stored raw payloads.

Match-V5 carries 156 participant fields and a 126-metric `challenges` block that
the flattened `participants` table deliberately ignores. Because the raw gzipped
JSON was kept, these can be derived at any time without re-crawling - which is the
whole reason for keeping it.

Each metric here maps to a specific hypothesis about why a team wins, grouped
below so the feature layer can build interpretable playstyle axes rather than a
pile of anonymous numbers.
"""
from __future__ import annotations

import gzip
import json
from typing import Any, Iterator

# (column, source, key, default) - source is "p" for the participant object or
# "c" for its challenges block.
EXTRA_SPEC: list[tuple[str, str, str, Any]] = [
    # identity / account age: the smurf signal
    ("summoner_level", "p", "summonerLevel", None),
    ("champ_level", "p", "champLevel", None),
    # laning and early game
    ("laning_gold_exp_adv", "c", "laningPhaseGoldExpAdvantage", None),
    ("early_laning_gold_exp_adv", "c", "earlyLaningPhaseGoldExpAdvantage", None),
    ("max_cs_adv_lane", "c", "maxCsAdvantageOnLaneOpponent", None),
    ("max_level_lead_lane", "c", "maxLevelLeadLaneOpponent", None),
    ("turret_plates", "c", "turretPlatesTaken", None),
    ("lane_minions_10", "c", "laneMinionsFirst10Minutes", None),
    # aggression and risk appetite
    ("solo_kills", "c", "soloKills", None),
    ("kill_participation", "c", "killParticipation", None),
    ("damage_per_minute", "c", "damagePerMinute", None),
    ("team_damage_pct", "c", "teamDamagePercentage", None),
    ("kills_near_enemy_turret", "c", "killsNearEnemyTurret", None),
    ("outnumbered_kills", "c", "outnumberedKills", None),
    ("multikills", "c", "multikills", None),
    # objectives and map play
    ("dragon_takedowns", "c", "dragonTakedowns", None),
    ("herald_takedowns", "c", "riftHeraldTakedowns", None),
    ("turret_takedowns", "c", "turretTakedowns", None),
    ("damage_to_turrets", "p", "damageDealtToTurrets", None),
    # vision
    ("vision_per_min", "c", "visionScorePerMinute", None),
    ("vision_adv_lane", "c", "visionScoreAdvantageLaneOpponent", None),
    ("control_wards", "c", "controlWardsPlaced", None),
    # survival and mechanics
    ("damage_taken_pct", "c", "damageTakenOnTeamPercentage", None),
    ("skillshots_dodged", "c", "skillshotsDodged", None),
    ("survived_low_hp", "c", "survivedSingleDigitHpCount", None),
    ("time_spent_dead", "p", "totalTimeSpentDead", None),
    ("time_ccing", "p", "timeCCingOthers", None),
    ("first_blood", "p", "firstBloodKill", None),
]

EXTRA_COLS = ["match_id", "puuid"] + [c for c, _, _, _ in EXTRA_SPEC]

DDL = """
CREATE TABLE IF NOT EXISTS participant_extras (
    match_id VARCHAR,
    puuid    VARCHAR,
""" + ",\n".join(
    f"    {col} {'BOOLEAN' if col == 'first_blood' else 'DOUBLE'}"
    for col, _, _, _ in EXTRA_SPEC
) + """,
    PRIMARY KEY (match_id, puuid)
);
"""


def extract(match_id: str, payload: dict) -> list[tuple]:
    """Flatten one match payload into per-participant extra rows."""
    rows: list[tuple] = []
    for p in (payload.get("info") or {}).get("participants", []):
        if not p.get("puuid"):
            continue
        ch = p.get("challenges") or {}
        values: list[Any] = [match_id, p["puuid"]]
        for col, source, key, default in EXTRA_SPEC:
            src = p if source == "p" else ch
            v = src.get(key, default)
            if col == "first_blood":
                values.append(bool(v) if v is not None else None)
            else:
                values.append(float(v) if isinstance(v, (int, float, bool)) else None)
        rows.append(tuple(values))
    return rows


def iter_raw(store, batch: int = 500, only_missing: bool = True) -> Iterator[list[tuple]]:
    """Yield batches of extra rows, decompressing raw payloads as we go."""
    store.con.execute(DDL)
    where = (
        "WHERE r.match_id NOT IN (SELECT DISTINCT match_id FROM participant_extras)"
        if only_missing
        else ""
    )
    total = store.con.execute(
        f"SELECT count(*) FROM raw_matches r {where}"
    ).fetchone()[0]
    if not total:
        return
    offset = 0
    while True:
        chunk = store.con.execute(
            f"""
            SELECT r.match_id, r.payload FROM raw_matches r {where}
            LIMIT {batch} OFFSET {offset}
            """
        ).fetchall()
        if not chunk:
            return
        rows: list[tuple] = []
        for match_id, blob in chunk:
            try:
                payload = json.loads(gzip.decompress(blob).decode("utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt blob must not stop the backfill
                continue
            rows.extend(extract(match_id, payload))
        yield rows
        # only_missing makes the result set shrink as we insert, so paging from 0
        # each time is correct; without it we must advance the offset.
        if not only_missing:
            offset += batch
