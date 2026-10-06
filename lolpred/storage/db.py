"""DuckDB-backed store for raw matches, clean tables and crawl state.

Single-writer by design: the crawler does HTTP on worker threads but funnels every
write through the main thread, because a DuckDB connection is not thread-safe.

Everything is written in bulk. DuckDB is columnar, so row-at-a-time inserts are
slow and `INSERT ... ON CONFLICT DO NOTHING` is catastrophically slow (measured at
78 rows/s here, versus 329k rows/s for a staged bulk insert). Deduplication is
therefore done with an explicit anti-join against a registered staging frame.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import pandas as pd

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

PARTICIPANT_COLS = [
    "match_id", "puuid", "team_id", "position", "champion_id", "win", "kills",
    "deaths", "assists", "gold_earned", "cs", "dmg_champs", "dmg_taken",
    "vision_score", "wards_placed", "turret_kills", "team_pos_idx",
]
MATCH_COLS = [
    "match_id", "platform", "queue_id", "patch", "game_version", "game_start",
    "duration_s", "winner",
]
RAW_COLS = ["match_id", "fetched_at", "patch", "payload"]
PLAYER_COLS = [
    "puuid", "tier", "division", "lp", "source", "discovered_at", "processed_at",
    "n_match_ids",
]
RANK_COLS = ["puuid", "queue", "tier", "division", "lp", "wins", "losses", "observed_at"]
CRAWL_MATCH_COLS = ["match_id", "state", "reason", "discovered_at", "resolved_at"]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Store:
    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self.con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            self._migrate()

    def _migrate(self) -> None:
        """Bring an older database up to the current schema in place."""
        existing = {
            row[1] for row in self.con.execute("PRAGMA table_info('crawl_matches')").fetchall()
        }
        if "attempts" not in existing:
            self.con.execute("ALTER TABLE crawl_matches ADD COLUMN attempts INTEGER DEFAULT 0")
            self.con.execute("UPDATE crawl_matches SET attempts = 0 WHERE attempts IS NULL")
        if "discovery_count" not in existing:
            self.con.execute(
                "ALTER TABLE crawl_matches ADD COLUMN discovery_count INTEGER DEFAULT 0"
            )
            self.con.execute(
                "UPDATE crawl_matches SET discovery_count = 0 WHERE discovery_count IS NULL"
            )

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- bulk primitives ---------------------------------------------------
    def _insert_new(
        self,
        table: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        key_cols: Sequence[str],
    ) -> int:
        """Insert only rows whose key is not already present. Returns rows added."""
        if not rows:
            return 0
        # dtype=object keeps None as NULL instead of letting pandas coerce an
        # integer column to float and turn missing values into NaN.
        frame = pd.DataFrame(list(rows), columns=list(columns), dtype=object)
        frame = frame.drop_duplicates(subset=list(key_cols))
        self.con.register("staging", frame)
        try:
            predicate = " AND ".join(f"t.{c} = s.{c}" for c in key_cols)
            result = self.con.execute(
                f"""
                INSERT INTO {table} ({", ".join(columns)})
                SELECT {", ".join("s." + c for c in columns)} FROM staging s
                WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {predicate})
                """
            ).fetchone()
        finally:
            self.con.unregister("staging")
        return int(result[0]) if result else 0

    # -- crawl frontier: players ------------------------------------------
    def add_players(self, rows: Iterable[dict], source: str) -> int:
        now = _now()
        payload = [
            (r["puuid"], r.get("tier"), r.get("rank"), r.get("leaguePoints"), source, now, None, None)
            for r in rows
            if r.get("puuid")
        ]
        return self._insert_new("crawl_players", PLAYER_COLS, payload, ["puuid"])

    def add_rank_snapshots(self, rows: Iterable[dict], queue: str = "RANKED_SOLO_5x5") -> int:
        observed = _now()
        payload = [
            (
                r["puuid"], queue, r.get("tier"), r.get("rank"), r.get("leaguePoints"),
                r.get("wins"), r.get("losses"), observed,
            )
            for r in rows
            if r.get("puuid") and r.get("tier")
        ]
        return self._insert_new(
            "player_ranks", RANK_COLS, payload, ["puuid", "queue", "observed_at"]
        )

    def claim_players(self, limit: int, order: str = "random") -> list[str]:
        """Take the next batch of unprocessed players from the frontier."""
        ordering = "random()" if order == "random" else "discovered_at"
        rows = self.con.execute(
            f"""
            SELECT puuid FROM crawl_players
            WHERE processed_at IS NULL
            ORDER BY {ordering}
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [r[0] for r in rows]

    def mark_players_processed(self, puuids: Sequence[str], counts: Sequence[int]) -> None:
        if not puuids:
            return
        frame = pd.DataFrame(
            {"puuid": list(puuids), "n": [int(c) for c in counts]}, dtype=object
        )
        self.con.register("staging", frame)
        try:
            self.con.execute(
                """
                UPDATE crawl_players
                SET processed_at = ?, n_match_ids = CAST(s.n AS INTEGER)
                FROM staging s
                WHERE crawl_players.puuid = s.puuid
                """,
                [_now()],
            )
        finally:
            self.con.unregister("staging")

    # -- crawl frontier: matches ------------------------------------------
    def add_match_ids(self, match_ids: Iterable[str]) -> int:
        now = _now()
        rows = [(mid, "pending", None, now, None) for mid in match_ids if mid]
        return self._insert_new("crawl_matches", CRAWL_MATCH_COLS, rows, ["match_id"])

    def pending_match_ids(
        self, limit: int, min_coverage: int = 0, use_plan: bool = False
    ) -> list[str]:
        """Pending matches, best-covered first when a coverage floor is given."""
        if use_plan:
            rows = self.con.execute(
                """
                SELECT m.match_id FROM crawl_matches m
                JOIN crawl_plan p ON p.match_id = m.match_id
                WHERE m.state = 'pending'
                ORDER BY coalesce(p.priority, 0) DESC, m.discovery_count DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            return [r[0] for r in rows]
        if min_coverage <= 0:
            rows = self.con.execute(
                "SELECT match_id FROM crawl_matches WHERE state = 'pending' LIMIT ?", [limit]
            ).fetchall()
        else:
            rows = self.con.execute(
                """
                SELECT match_id FROM crawl_matches
                WHERE state = 'pending' AND discovery_count >= ?
                ORDER BY discovery_count DESC
                LIMIT ?
                """,
                [min_coverage, limit],
            ).fetchall()
        return [r[0] for r in rows]

    # -- discovery map -----------------------------------------------------
    def add_discoveries(self, pairs: Sequence[tuple[str, str]]) -> int:
        """Record (match_id, puuid) edges and refresh the affected coverage counts.

        Returns the number of match IDs newly added to the pending queue.
        """
        if not pairs:
            return 0
        added_edges = self._insert_new(
            "match_discoveries", ["match_id", "puuid"], pairs, ["match_id", "puuid"]
        )
        new_matches = self.add_match_ids({mid for mid, _ in pairs})
        if added_edges:
            touched = pd.DataFrame(
                sorted({mid for mid, _ in pairs}), columns=["match_id"], dtype=object
            )
            self.con.register("staging", touched)
            try:
                self.con.execute(
                    """
                    UPDATE crawl_matches
                    SET discovery_count = c.n
                    FROM (
                        SELECT d.match_id, count(*) AS n
                        FROM match_discoveries d
                        WHERE d.match_id IN (SELECT match_id FROM staging)
                        GROUP BY d.match_id
                    ) c
                    WHERE crawl_matches.match_id = c.match_id
                    """
                )
            finally:
                self.con.unregister("staging")
        return new_matches

    def set_plan(self, match_ids: Sequence[str]) -> int:
        """Replace the crawl plan with an explicit list of matches to fetch."""
        self.con.execute("CREATE TABLE IF NOT EXISTS crawl_plan (match_id VARCHAR PRIMARY KEY)")
        self.con.execute("DELETE FROM crawl_plan")
        if not match_ids:
            return 0
        frame = pd.DataFrame(sorted(set(match_ids)), columns=["match_id"], dtype=object)
        self.con.register("staging", frame)
        try:
            self.con.execute("INSERT INTO crawl_plan SELECT match_id FROM staging")
        finally:
            self.con.unregister("staging")
        return self.count("crawl_plan")

    def prioritise_plan(self, target_players: Sequence[str]) -> dict:
        """Score planned matches by how many target-set players they carry.

        The plan is fetched priority-first, so matches that deepen the history of
        players in the designated target matches come first. That makes fully
        covered rows appear early in the crawl instead of only at the very end;
        the total work is unchanged.
        """
        self.con.execute("CREATE TABLE IF NOT EXISTS crawl_plan (match_id VARCHAR PRIMARY KEY)")
        cols = {r[1] for r in self.con.execute("PRAGMA table_info('crawl_plan')").fetchall()}
        if "priority" not in cols:
            self.con.execute("ALTER TABLE crawl_plan ADD COLUMN priority INTEGER DEFAULT 0")
        frame = pd.DataFrame(sorted(set(target_players)), columns=["puuid"], dtype=object)
        self.con.register("staging", frame)
        try:
            self.con.execute("UPDATE crawl_plan SET priority = 0")
            self.con.execute(
                """
                UPDATE crawl_plan
                SET priority = c.n
                FROM (
                    SELECT d.match_id, count(*) AS n
                    FROM match_discoveries d
                    JOIN staging s ON s.puuid = d.puuid
                    GROUP BY d.match_id
                ) c
                WHERE crawl_plan.match_id = c.match_id
                """
            )
        finally:
            self.con.unregister("staging")
        rows = self.con.execute(
            """
            SELECT p.priority, count(*)
            FROM crawl_plan p JOIN crawl_matches m ON m.match_id = p.match_id
            WHERE m.state = 'pending'
            GROUP BY 1 ORDER BY 1 DESC
            """
        ).fetchall()
        return {"target_players": len(frame), "pending_by_priority": rows[:8]}

    def planned_match_ids(self) -> list[str]:
        try:
            return [r[0] for r in self.con.execute("SELECT match_id FROM crawl_plan").fetchall()]
        except duckdb.CatalogException:
            return []

    def resolved_match_ids(self) -> list[str]:
        return [
            r[0]
            for r in self.con.execute(
                "SELECT match_id FROM crawl_matches WHERE state <> 'pending'"
            ).fetchall()
        ]

    def fully_processed_participants(self) -> pd.DataFrame:
        """(match_id, puuid) for stored matches whose all ten players are processed."""
        return self.con.execute(
            """
            WITH per_match AS (
                SELECT p.match_id,
                       count(*) FILTER (WHERE c.processed_at IS NOT NULL) AS cov
                FROM participants p
                LEFT JOIN crawl_players c ON c.puuid = p.puuid
                GROUP BY p.match_id
            )
            SELECT p.match_id, p.puuid
            FROM participants p
            JOIN per_match pm ON pm.match_id = p.match_id
            WHERE pm.cov = 10
            """
        ).df()

    def extend_plan(self, match_ids: Sequence[str]) -> int:
        """Add matches to the plan without dropping what is already there."""
        self.con.execute("CREATE TABLE IF NOT EXISTS crawl_plan (match_id VARCHAR PRIMARY KEY)")
        if not match_ids:
            return 0
        return self._insert_new(
            "crawl_plan", ["match_id"], [(mid,) for mid in set(match_ids)], ["match_id"]
        )

    def plan_remaining(self) -> int:
        try:
            return int(
                self.con.execute(
                    """
                    SELECT count(*) FROM crawl_plan p
                    JOIN crawl_matches m ON m.match_id = p.match_id
                    WHERE m.state = 'pending'
                    """
                ).fetchone()[0]
            )
        except duckdb.CatalogException:
            return 0

    def discovery_edges(self) -> pd.DataFrame:
        return self.con.execute("SELECT match_id, puuid FROM match_discoveries").df()

    def coverage_histogram(self) -> list[tuple[int, int]]:
        """How many pending matches sit at each discovery count."""
        return self.con.execute(
            """
            SELECT discovery_count, count(*)
            FROM crawl_matches WHERE state = 'pending'
            GROUP BY 1 ORDER BY 1 DESC
            """
        ).fetchall()

    def stored_coverage_histogram(self) -> list[tuple[int, int]]:
        """For already-stored matches, how many participants have been processed."""
        return self.con.execute(
            """
            WITH per_match AS (
                SELECT p.match_id,
                       count(*) FILTER (WHERE c.processed_at IS NOT NULL) AS covered
                FROM participants p
                LEFT JOIN crawl_players c ON c.puuid = p.puuid
                GROUP BY p.match_id
            )
            SELECT covered, count(*) FROM per_match GROUP BY 1 ORDER BY 1 DESC
            """
        ).fetchall()

    def set_match_states(self, items: Sequence[tuple[str, str, str | None]]) -> None:
        """items: (match_id, state, reason)."""
        if not items:
            return
        frame = pd.DataFrame(
            list(items), columns=["match_id", "state", "reason"], dtype=object
        ).drop_duplicates(subset=["match_id"])
        self.con.register("staging", frame)
        try:
            self.con.execute(
                """
                UPDATE crawl_matches
                SET state = s.state, reason = s.reason, resolved_at = ?
                FROM staging s
                WHERE crawl_matches.match_id = s.match_id
                """,
                [_now()],
            )
        finally:
            self.con.unregister("staging")

    def register_failures(
        self, items: Sequence[tuple[str, str]], max_attempts: int = 4
    ) -> tuple[int, int]:
        """Record transient fetch failures: retry until max_attempts, then give up.

        A network outage or a 5xx should never permanently remove a match from the
        queue - only a 404 or a validation reject is final. Returns (requeued, dead).
        """
        if not items:
            return 0, 0
        frame = pd.DataFrame(
            list(items), columns=["match_id", "reason"], dtype=object
        ).drop_duplicates(subset=["match_id"])
        self.con.register("staging", frame)
        try:
            self.con.execute(
                """
                UPDATE crawl_matches
                SET attempts = coalesce(crawl_matches.attempts, 0) + 1,
                    reason = s.reason,
                    state = CASE WHEN coalesce(crawl_matches.attempts, 0) + 1 >= ?
                                 THEN 'error' ELSE 'pending' END,
                    resolved_at = CASE WHEN coalesce(crawl_matches.attempts, 0) + 1 >= ?
                                       THEN ? ELSE NULL END
                FROM staging s
                WHERE crawl_matches.match_id = s.match_id
                """,
                [max_attempts, max_attempts, _now()],
            )
            dead = int(
                self.con.execute(
                    """
                    SELECT count(*) FROM crawl_matches m JOIN staging s
                    ON s.match_id = m.match_id WHERE m.state = 'error'
                    """
                ).fetchone()[0]
            )
        finally:
            self.con.unregister("staging")
        return len(frame) - dead, dead

    def requeue_errors(self, reset_attempts: bool = True) -> int:
        """Put matches that failed transiently back in the queue."""
        n = int(
            self.con.execute(
                "SELECT count(*) FROM crawl_matches WHERE state = 'error'"
            ).fetchone()[0]
        )
        if n:
            self.con.execute(
                f"""
                UPDATE crawl_matches
                SET state = 'pending', resolved_at = NULL
                    {", attempts = 0" if reset_attempts else ""}
                WHERE state = 'error'
                """
            )
        return n

    # -- match payloads ----------------------------------------------------
    @staticmethod
    def compress(payload: dict) -> bytes:
        return gzip.compress(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"), compresslevel=6
        )

    def store_matches(
        self,
        match_rows: Sequence[Sequence[Any]],
        participant_rows: Sequence[Sequence[Any]],
        raw_rows: Sequence[tuple[str, str | None, bytes]],
    ) -> int:
        """Write a whole batch of matches: raw payloads, headers and participants."""
        now = _now()
        added = self._insert_new("matches", MATCH_COLS, match_rows, ["match_id"])
        self._insert_new(
            "participants", PARTICIPANT_COLS, participant_rows, ["match_id", "puuid"]
        )
        self._insert_new(
            "raw_matches",
            RAW_COLS,
            [(mid, now, patch, blob) for mid, patch, blob in raw_rows],
            ["match_id"],
        )
        return added

    def load_raw(self, match_id: str) -> dict | None:
        row = self.con.execute(
            "SELECT payload FROM raw_matches WHERE match_id = ?", [match_id]
        ).fetchone()
        if not row:
            return None
        return json.loads(gzip.decompress(row[0]).decode("utf-8"))

    # -- introspection -----------------------------------------------------
    def count(self, table: str) -> int:
        return int(self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    def progress(self) -> dict[str, Any]:
        states = dict(
            self.con.execute(
                "SELECT state, count(*) FROM crawl_matches GROUP BY state"
            ).fetchall()
        )
        frontier = self.con.execute(
            """
            SELECT count(*) FILTER (WHERE processed_at IS NULL),
                   count(*) FILTER (WHERE processed_at IS NOT NULL)
            FROM crawl_players
            """
        ).fetchone()
        return {
            "matches_stored": int(states.get("stored", 0)),
            "matches_pending": int(states.get("pending", 0)),
            "matches_rejected": int(states.get("rejected", 0)),
            "matches_error": int(states.get("error", 0)),
            "players_queued": int(frontier[0]),
            "players_done": int(frontier[1]),
        }

    def patch_breakdown(self) -> list[tuple[str, int]]:
        return self.con.execute(
            "SELECT patch, count(*) FROM matches GROUP BY patch ORDER BY patch"
        ).fetchall()

    def tier_breakdown(self) -> list[tuple[str, int]]:
        return self.con.execute(
            """
            SELECT coalesce(tier, '(unknown)'), count(*)
            FROM crawl_players GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
