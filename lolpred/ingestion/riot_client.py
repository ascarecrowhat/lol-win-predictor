"""Thin, rate-limited, retrying Riot API client.

Endpoint versions are correct as of writing but Riot deprecates routes; if a call
starts 404-ing for every input, check the developer portal for a bumped version.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Iterable

import requests

from .rate_limit import LimiterRegistry

log = logging.getLogger(__name__)

PLATFORM_TO_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "ph2": "sea", "sg2": "sea", "th2": "sea", "tw2": "sea", "vn2": "sea",
}

TIERS_WITH_DIVISIONS = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND",
]
DIVISIONS = ["I", "II", "III", "IV"]
APEX_TIERS = ["challenger", "grandmaster", "master"]


class RiotError(RuntimeError):
    pass


class RiotAuthError(RiotError):
    """403/401 - key missing, expired, or not allowed on this endpoint."""


class RiotClient:
    def __init__(
        self,
        api_key: str,
        platform: str,
        region: str | None = None,
        app_limits: list[tuple[int, int]] | None = None,
        max_retries: int = 5,
        timeout: float = 15.0,
    ):
        self.api_key = api_key
        self.platform = platform
        self.region = region or PLATFORM_TO_REGION.get(platform, "europe")
        self.limiters = LimiterRegistry(app_limits or [(20, 1), (100, 120)])
        self.max_retries = max_retries
        self.timeout = timeout
        self._local = threading.local()
        self.stats = {"requests": 0, "retries": 0, "rate_limited": 0, "errors": 0}
        self._stats_lock = threading.Lock()

    # -- plumbing ---------------------------------------------------------
    @property
    def _session(self) -> requests.Session:
        """One session per thread; requests.Session is not thread-safe."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"X-Riot-Token": self.api_key})
            self._local.session = session
        return session

    def _bump(self, key: str, n: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] += n

    def _get(
        self,
        host: str,
        path: str,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """GET with rate limiting and retries. Returns None on 404."""
        url = f"https://{host}.api.riotgames.com{path}"
        for attempt in range(self.max_retries + 1):
            self.limiters.acquire(method)
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                self._bump("errors")
                if attempt == self.max_retries:
                    raise RiotError(f"{method}: network failure: {exc}") from exc
                time.sleep(self._backoff(attempt))
                continue

            self._bump("requests")
            self.limiters.learn_method_limits(method, resp.headers.get("X-Method-Rate-Limit"))

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code in (401, 403):
                raise RiotAuthError(
                    f"{method}: HTTP {resp.status_code}. Development keys expire every 24 "
                    "hours; regenerate it at developer.riotgames.com."
                )
            if resp.status_code == 429:
                self._bump("rate_limited")
                scope = resp.headers.get("X-Rate-Limit-Type", "application")
                retry_after = float(resp.headers.get("Retry-After", "1"))
                log.warning("429 (%s), sleeping %.1fs [%s]", scope, retry_after, method)
                self.limiters.penalise(method, retry_after, scope)
                self._bump("retries")
                continue
            if resp.status_code >= 500 or resp.status_code == 408:
                self._bump("retries")
                if attempt == self.max_retries:
                    raise RiotError(f"{method}: HTTP {resp.status_code} after retries")
                time.sleep(self._backoff(attempt))
                continue
            raise RiotError(f"{method}: HTTP {resp.status_code}: {resp.text[:300]}")
        raise RiotError(f"{method}: exhausted retries")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(2.0 ** attempt, 30.0) * (0.5 + random.random())

    # -- account-v1 (regional routing) ------------------------------------
    def account_by_riot_id(self, game_name: str, tag_line: str) -> dict | None:
        return self._get(
            self.region,
            f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}",
            "account.by-riot-id",
        )

    def account_by_puuid(self, puuid: str) -> dict | None:
        return self._get(
            self.region, f"/riot/account/v1/accounts/by-puuid/{puuid}", "account.by-puuid"
        )

    # -- match-v5 (regional routing) --------------------------------------
    def match_ids(
        self,
        puuid: str,
        queue: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        start: int = 0,
        count: int = 100,
    ) -> list[str]:
        params: dict[str, Any] = {"start": start, "count": min(count, 100)}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = start_time  # epoch SECONDS, not ms
        if end_time is not None:
            params["endTime"] = end_time
        return self._get(
            self.region, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", "match.ids", params
        ) or []

    def match(self, match_id: str) -> dict | None:
        return self._get(self.region, f"/lol/match/v5/matches/{match_id}", "match.by-id")

    def match_timeline(self, match_id: str) -> dict | None:
        return self._get(
            self.region, f"/lol/match/v5/matches/{match_id}/timeline", "match.timeline"
        )

    # -- league-v4 (platform routing) -------------------------------------
    def league_entries(
        self, tier: str, division: str, queue: str = "RANKED_SOLO_5x5", page: int = 1
    ) -> list[dict]:
        return self._get(
            self.platform,
            f"/lol/league/v4/entries/{queue}/{tier}/{division}",
            "league.entries",
            {"page": page},
        ) or []

    def apex_league(self, tier: str, queue: str = "RANKED_SOLO_5x5") -> dict | None:
        """tier is one of challenger, grandmaster, master."""
        return self._get(
            self.platform,
            f"/lol/league/v4/{tier}leagues/by-queue/{queue}",
            f"league.{tier}",
        )

    def league_entries_by_puuid(self, puuid: str) -> list[dict]:
        return self._get(
            self.platform, f"/lol/league/v4/entries/by-puuid/{puuid}", "league.by-puuid"
        ) or []

    # -- summoner-v4 (platform routing) -----------------------------------
    def summoner_by_id(self, summoner_id: str) -> dict | None:
        return self._get(
            self.platform, f"/lol/summoner/v4/summoners/{summoner_id}", "summoner.by-id"
        )

    # -- champion-mastery-v4 / spectator-v5 (platform routing) ------------
    def masteries(self, puuid: str) -> list[dict]:
        return self._get(
            self.platform,
            f"/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}",
            "mastery.by-puuid",
        ) or []

    def active_game(self, puuid: str) -> dict | None:
        return self._get(
            self.platform,
            f"/lol/spectator/v5/active-games/by-summoner/{puuid}",
            "spectator.active-game",
        )

    # -- helpers -----------------------------------------------------------
    def iter_ranked_ladder(
        self, tiers: Iterable[str] | None = None, pages_per_division: int = 1
    ) -> Iterable[dict]:
        """Yield league entries across the ladder, apex tiers first."""
        tier_list = list(tiers) if tiers is not None else APEX_TIERS + TIERS_WITH_DIVISIONS
        for tier in tier_list:
            if tier.lower() in APEX_TIERS:
                league = self.apex_league(tier.lower())
                for entry in (league or {}).get("entries", []):
                    yield {"tier": tier.upper(), "rank": "I", **entry}
                continue
            for division in DIVISIONS:
                for page in range(1, pages_per_division + 1):
                    entries = self.league_entries(tier.upper(), division, page=page)
                    if not entries:
                        break
                    yield from entries
