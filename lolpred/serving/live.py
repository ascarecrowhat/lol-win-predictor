"""Predict a live game from ten summoners and ten champions.

The input contract is deliberately narrow: PUUIDs and champion IDs, nothing else
about the game in progress. Everything predictive is reconstructed from each
player's *earlier* matches, using the same feature code that produced the training
set, so train and serve cannot drift apart.

History comes from the local corpus first and Riot only for the gaps, because
rate limit is the binding cost: a player already crawled is free, an unknown one
costs roughly a dozen requests.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.advanced import STYLE_METRICS
from ..features.build import PLAYER_FEATURES, _advanced_features, _aggregate, AGGS, NO_MINMAX
from ..features.snapshot import load as load_snapshot
from ..ingestion.riot_client import RiotClient
from ..storage.db import Store
from ..storage.extras import EXTRA_COLS
from ..storage.extras import extract as extract_extras

log = logging.getLogger(__name__)

SOLO_QUEUE_ID = 420
# u.gg profile URLs look like /lol/profile/euw1/name-tag/overview (and /live-game)
UGG_RE = re.compile(
    r"u\.gg/lol/profile/(?P<platform>[a-z0-9]+)/(?P<name>[^/]+?)-(?P<tag>[^/-]+)(?:/|$)",
    re.IGNORECASE,
)
RIOT_ID_RE = re.compile(r"^\s*(?P<name>[^#]{1,32})#(?P<tag>[A-Za-z0-9]{2,8})\s*$")


class NotEligible(Exception):
    """The game exists but falls outside what the model was trained on."""


@dataclass
class LiveGame:
    game_id: int
    platform: str
    queue_id: int
    participants: list[dict] = field(default_factory=list)


def parse_target(text: str) -> tuple[str, str, str | None]:
    """Accept a u.gg profile link or a plain Riot ID. Returns (name, tag, platform).

    Only the identifier is taken from the URL - no page is fetched. Every fact
    used downstream comes from Riot's own API.
    """
    m = UGG_RE.search(text or "")
    if m:
        name = m.group("name").replace("+", " ").replace("%20", " ")
        return name, m.group("tag"), m.group("platform").lower()
    m = RIOT_ID_RE.match(text or "")
    if m:
        return m.group("name").strip(), m.group("tag"), None
    raise ValueError(
        "Give a Riot ID like Faker#KR1 or a u.gg profile link "
        "(https://u.gg/lol/profile/euw1/name-tag/overview)"
    )


class LivePredictor:
    def __init__(
        self,
        client: RiotClient,
        store: Store,
        snapshot_path: str | Path,
        model_path: str | Path,
        history_per_player: int = 12,
        allowed_platforms: tuple[str, ...] = ("EUW1",),
    ):
        import joblib

        self.client = client
        self.store = store
        # A fresh serving cache has no extras table yet; creating it here keeps
        # the first request from failing on a brand-new deployment.
        from ..storage.extras import DDL as EXTRAS_DDL

        self.store.con.execute(EXTRAS_DDL)
        self.history_per_player = history_per_player
        self.allowed_platforms = tuple(p.upper() for p in allowed_platforms)
        self.base_state, self.base_adv, self.base_roles, self.snapshot_meta = load_snapshot(
            snapshot_path
        )
        artefact = joblib.load(model_path)
        self.model = artefact["model"]
        self.calibrator = artefact["calibrator"]
        self.features: list[str] = artefact["features"]
        self.model_name = artefact.get("model_name", "model")

    # -- lookup ------------------------------------------------------------
    def resolve(self, text: str) -> dict:
        name, tag, platform = parse_target(text)
        account = self.client.account_by_riot_id(name, tag)
        if not account or not account.get("puuid"):
            raise NotEligible(f"No Riot account found for {name}#{tag}")
        return {"puuid": account["puuid"], "name": name, "tag": tag, "platform": platform}

    def active_game(self, puuid: str) -> LiveGame:
        game = self.client.active_game(puuid)
        if not game:
            raise NotEligible("That summoner is not in a game right now.")
        platform = str(game.get("platformId", "")).upper()
        queue = int(game.get("gameQueueConfigId") or 0)
        parts = [
            {"puuid": p.get("puuid"), "champion_id": int(p.get("championId") or 0),
             "team_id": int(p.get("teamId") or 0), "name": p.get("riotId") or p.get("summonerName")}
            for p in game.get("participants", [])
        ]
        live = LiveGame(int(game.get("gameId") or 0), platform, queue, parts)

        if platform not in self.allowed_platforms:
            raise NotEligible(
                f"Game is on {platform}; this model was trained on "
                f"{', '.join(self.allowed_platforms)} only."
            )
        if queue != SOLO_QUEUE_ID:
            raise NotEligible(
                f"Queue {queue} is not ranked solo/duo (420), which is all the model has seen."
            )
        if len(parts) != 10 or any(not p["puuid"] for p in parts):
            raise NotEligible(f"Expected 10 identified participants, got {len(parts)}.")
        return live

    # -- history -----------------------------------------------------------
    def _local_history(self, puuids: list[str], exclude: str | None = None) -> pd.DataFrame:
        """Prior matches for these players that the corpus already holds."""
        extra_cols = ", ".join(f"e.{m}" for m in STYLE_METRICS)
        return self.store.con.execute(
            f"""
            SELECT m.match_id, m.game_start, m.patch, m.duration_s, m.winner,
                   p.puuid, p.team_id, p.position, p.champion_id, p.win,
                   p.kills, p.deaths, p.assists, p.gold_earned, p.cs,
                   p.dmg_champs, p.vision_score, {extra_cols}
            FROM matches m
            JOIN participants p ON p.match_id = m.match_id
            LEFT JOIN participant_extras e
                   ON e.match_id = p.match_id AND e.puuid = p.puuid
            WHERE m.match_id IN (
                SELECT DISTINCT match_id FROM participants WHERE puuid IN (SELECT unnest(?))
            )
              AND (? IS NULL OR m.match_id <> ?)
            ORDER BY m.game_start, m.match_id
            """,
            [puuids, exclude, exclude],
        ).df()

    def _fetch_history(self, puuid: str, have: int) -> list[dict]:
        """Top a player up from Riot when the corpus does not know them well."""
        need = max(0, self.history_per_player - have)
        if need == 0:
            return []
        ids = self.client.match_ids(puuid, queue=SOLO_QUEUE_ID, count=need)
        payloads = []
        for match_id in ids:
            if self.store.con.execute(
                "SELECT 1 FROM matches WHERE match_id = ?", [match_id]
            ).fetchone():
                continue
            payload = self.client.match(match_id)
            if payload:
                payloads.append((match_id, payload))
        return payloads

    def gather_history(
        self, puuids: list[str], top_up: bool = True, exclude: str | None = None
    ) -> tuple[pd.DataFrame, dict]:
        local = self._local_history(puuids, exclude=exclude)
        depth = (
            local[local.puuid.isin(puuids)].groupby("puuid").size().to_dict()
            if not local.empty else {}
        )
        fetched = 0
        if top_up:
            from ..storage.transform import flatten, validate

            new_rows = []
            for puuid in puuids:
                for match_id, payload in self._fetch_history(puuid, depth.get(puuid, 0)):
                    try:
                        patch = validate(payload, SOLO_QUEUE_ID, (0, 0))
                        match_row, part_rows = flatten(match_id, payload, patch)
                    except Exception:  # noqa: BLE001
                        continue
                    self.store.store_matches(
                        [match_row], part_rows, [(match_id, patch, self.store.compress(payload))]
                    )
                    self.store._insert_new(
                        "participant_extras", EXTRA_COLS,
                        extract_extras(match_id, payload), ["match_id", "puuid"],
                    )
                    new_rows.append(match_id)
                    fetched += 1
            if new_rows:
                local = self._local_history(puuids, exclude=exclude)
                depth = local[local.puuid.isin(puuids)].groupby("puuid").size().to_dict()
        return local, {"fetched_from_riot": fetched, "history_depth": depth}

    # -- predict -----------------------------------------------------------
    def predict(self, text: str, top_up: bool = True) -> dict[str, Any]:
        who = self.resolve(text)
        live = self.active_game(who["puuid"])
        puuids = [p["puuid"] for p in live.participants]
        this_match = f"{live.platform}_{live.game_id}" if live.game_id else None
        history, info = self.gather_history(puuids, top_up=top_up, exclude=this_match)

        state, adv, roles = self.base_state, self.base_adv, self.base_roles
        # Replay the players' prior matches in time order so their rolling state
        # matches what the training pass would have produced at this moment.
        if not history.empty:
            history = history.sort_values(["game_start", "match_id"])
            history["ts"] = history.game_start.astype("int64") / 1e9
            for _, grp in history.groupby("match_id", sort=False):
                if len(grp) != 10:
                    continue
                recs = grp.to_dict("records")
                ts = float(grp.ts.iloc[0])
                state.update(recs, ts, grp.patch.iloc[0], int(grp.duration_s.iloc[0]))
                adv.update(recs, int(grp.duration_s.iloc[0]))
                roles.update(recs)

        now_ts = pd.Timestamp.utcnow().timestamp()
        records = [
            {"puuid": p["puuid"], "champion_id": p["champion_id"], "team_id": p["team_id"]}
            for p in live.participants
        ]
        inferred = roles.assign_match(records)
        for r in records:
            r["position"] = inferred.get(r["puuid"], "MIDDLE")

        per_team = {100: {f: [] for f in PLAYER_FEATURES}, 200: {f: [] for f in PLAYER_FEATURES}}
        coverage = {100: 0, 200: 0}
        patch = str(self.snapshot_meta.get("latest_patch") or "")
        for r in records:
            feats = state.player_features(
                r["puuid"], int(r["champion_id"]), r["position"], now_ts, patch
            )
            for f in PLAYER_FEATURES:
                per_team[r["team_id"]][f].append(feats[f])
            if state.history_depth(r["puuid"]) >= 10:
                coverage[r["team_id"]] += 1

        blue, red = _aggregate(per_team[100]), _aggregate(per_team[200])
        row: dict[str, float] = {}
        for f in PLAYER_FEATURES:
            for a in AGGS:
                if a != "mean" and f in NO_MINMAX:
                    continue
                row[f"d_{f}_{a}"] = blue[f"{f}_{a}"] - red[f"{f}_{a}"]
        row["blue_coverage"] = coverage[100]
        row["red_coverage"] = coverage[200]
        row["d_coverage"] = coverage[100] - coverage[200]
        row["min_coverage"] = min(coverage.values())
        row.update(_advanced_features(adv, records))

        frame = pd.DataFrame([row]).reindex(columns=self.features)
        raw = float(self.model.predict_proba(frame)[:, 1][0])
        prob = float(np.clip(self.calibrator.predict([raw])[0], 0.01, 0.99))

        return {
            "game_id": live.game_id,
            "platform": live.platform,
            "queue_id": live.queue_id,
            "blue_win_probability": round(prob, 4),
            "red_win_probability": round(1 - prob, 4),
            "coverage": {"blue": coverage[100], "red": coverage[200]},
            "history": info,
            "model": self.model_name,
            "teams": [
                {
                    "team": "blue" if r["team_id"] == 100 else "red",
                    "name": next(
                        (p.get("name") for p in live.participants if p["puuid"] == r["puuid"]), None
                    ),
                    "champion_id": r["champion_id"],
                    "inferred_role": r["position"],
                    "prior_games": state.history_depth(r["puuid"]),
                }
                for r in records
            ],
            "explanation": self._explain(frame),
        }

    def _explain(self, frame: pd.DataFrame, top: int = 6) -> list[dict]:
        """Which inputs pushed the number, for a linear model: coefficient x value."""
        clf = getattr(self.model, "named_steps", {}).get("clf")
        if clf is None or not hasattr(clf, "coef_"):
            return []
        try:
            x = self.model.named_steps["scale"].transform(
                self.model.named_steps["impute"].transform(frame)
            )[0]
        except Exception:  # noqa: BLE001
            return []
        contrib = clf.coef_[0] * x
        order = np.argsort(-np.abs(contrib))[:top]
        return [
            {
                "feature": self.features[i],
                "effect": round(float(contrib[i]), 4),
                "favours": "blue" if contrib[i] > 0 else "red",
            }
            for i in order
        ]
