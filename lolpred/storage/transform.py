"""Validate a Match-V5 payload and flatten it into clean rows."""
from __future__ import annotations

from datetime import datetime, timezone

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
POSITION_IDX = {p: i for i, p in enumerate(POSITIONS)}

MIN_DURATION_S = 900  # 15 min: below this a game is usually a remake or early FF


class RejectedMatch(Exception):
    """Payload parsed fine but is not a match we want in the training set."""


def parse_patch(game_version: str | None) -> str | None:
    """'15.19.704.1234' -> '15.19'."""
    if not game_version:
        return None
    bits = game_version.split(".")
    if len(bits) < 2:
        return None
    return f"{bits[0]}.{bits[1]}"


def patch_tuple(patch: str | None) -> tuple[int, int]:
    if not patch:
        return (0, 0)
    major, _, minor = patch.partition(".")
    try:
        return (int(major), int(minor or 0))
    except ValueError:
        return (0, 0)


def _duration_seconds(info: dict) -> int:
    """gameDuration was milliseconds before patch 11.20, seconds after."""
    duration = int(info.get("gameDuration") or 0)
    if duration > 100_000:  # no real game runs 27+ hours
        duration //= 1000
    return duration


def validate(payload: dict, queue_id: int, min_patch: tuple[int, int]) -> str:
    """Return the patch string, or raise RejectedMatch with a reason."""
    info = payload.get("info") or {}
    participants = info.get("participants") or []

    if queue_id is not None and int(info.get("queueId") or -1) != int(queue_id):
        raise RejectedMatch(f"queue={info.get('queueId')}")
    if len(participants) != 10:
        raise RejectedMatch(f"participants={len(participants)}")
    if _duration_seconds(info) < MIN_DURATION_S:
        raise RejectedMatch("too_short")
    if info.get("endOfGameResult") not in (None, "GameComplete"):
        raise RejectedMatch(str(info.get("endOfGameResult")))

    patch = parse_patch(info.get("gameVersion"))
    if patch_tuple(patch) < min_patch:
        raise RejectedMatch(f"patch={patch}")

    by_team: dict[int, set[str]] = {100: set(), 200: set()}
    for p in participants:
        pos = p.get("teamPosition") or ""
        if pos not in POSITION_IDX:
            raise RejectedMatch("missing_position")
        by_team.setdefault(int(p.get("teamId") or 0), set()).add(pos)
    if len(by_team.get(100, set())) != 5 or len(by_team.get(200, set())) != 5:
        raise RejectedMatch("duplicate_position")

    return patch or ""


def flatten(match_id: str, payload: dict, patch: str) -> tuple[tuple, list[tuple]]:
    info = payload["info"]
    participants = info["participants"]

    start_ms = info.get("gameStartTimestamp") or info.get("gameCreation") or 0
    game_start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).replace(tzinfo=None)

    winner = next(
        (int(t["teamId"]) for t in info.get("teams", []) if t.get("win")),
        100 if participants[0].get("win") and int(participants[0]["teamId"]) == 100 else 200,
    )

    match_row = (
        match_id,
        info.get("platformId"),
        int(info.get("queueId") or 0),
        patch,
        info.get("gameVersion"),
        game_start,
        _duration_seconds(info),
        winner,
    )

    rows: list[tuple] = []
    for p in participants:
        pos = p.get("teamPosition")
        rows.append(
            (
                match_id,
                p.get("puuid"),
                int(p.get("teamId") or 0),
                pos,
                int(p.get("championId") or 0),
                bool(p.get("win")),
                int(p.get("kills") or 0),
                int(p.get("deaths") or 0),
                int(p.get("assists") or 0),
                int(p.get("goldEarned") or 0),
                int(p.get("totalMinionsKilled") or 0) + int(p.get("neutralMinionsKilled") or 0),
                int(p.get("totalDamageDealtToChampions") or 0),
                int(p.get("totalDamageTaken") or 0),
                int(p.get("visionScore") or 0),
                int(p.get("wardsPlaced") or 0),
                int(p.get("turretKills") or 0),
                POSITION_IDX.get(pos or "", -1),
            )
        )
    return match_row, rows


def participant_puuids(payload: dict) -> list[str]:
    return [
        p["puuid"]
        for p in (payload.get("info") or {}).get("participants", [])
        if p.get("puuid") and not str(p.get("puuid", "")).startswith("BOT")
    ]
