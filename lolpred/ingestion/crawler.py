"""Snowball crawler for ranked matches.

Seed from the ranked ladder, pull each player's recent match IDs, fetch each match,
then add that match's ten participants back into the frontier. Every piece of state
lives in DuckDB, so Ctrl-C and restart simply continues.

HTTP runs on a small thread pool (the rate limiter is shared and thread-safe);
all database writes happen on the calling thread.
"""
from __future__ import annotations

import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence, TypeVar

from ..storage.db import Store
from ..storage.transform import RejectedMatch, flatten, participant_puuids, validate
from .riot_client import RiotAuthError, RiotClient, RiotError

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class Crawler:
    def __init__(
        self,
        client: RiotClient,
        store: Store,
        queue_id: int = 420,
        min_patch: tuple[int, int] = (0, 0),
        lookback_days: int = 21,
        workers: int = 8,
        ids_per_player: int = 30,
        match_batch: int = 200,
        player_batch: int = 40,
        pending_low_water: int = 400,
    ):
        self.client = client
        self.store = store
        self.queue_id = queue_id
        self.min_patch = min_patch
        self.lookback_days = lookback_days
        self.workers = workers
        self.ids_per_player = ids_per_player
        self.match_batch = match_batch
        self.player_batch = player_batch
        self.pending_low_water = pending_low_water
        self.stop_requested = False

    # -- lifecycle ---------------------------------------------------------
    def install_signal_handler(self) -> None:
        def handler(signum, frame):  # noqa: ANN001
            if self.stop_requested:
                log.warning("Second interrupt - exiting now.")
                raise KeyboardInterrupt
            self.stop_requested = True
            log.warning("Interrupt received: finishing the current batch, then stopping.")

        signal.signal(signal.SIGINT, handler)

    def _map(self, fn: Callable[[T], R], items: Sequence[T]) -> list[tuple[T, R | None, str | None]]:
        """Run fn over items on the thread pool. Returns (item, result, error)."""
        out: list[tuple[T, R | None, str | None]] = []
        if not items:
            return out
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(fn, item): item for item in items}
            for future, item in futures.items():
                try:
                    out.append((item, future.result(), None))
                except RiotAuthError:
                    raise
                except RiotError as exc:
                    out.append((item, None, str(exc)))
                except Exception as exc:  # noqa: BLE001
                    out.append((item, None, f"{type(exc).__name__}: {exc}"))
        return out

    # -- seeding -----------------------------------------------------------
    def seed(self, tiers: Iterable[str] | None = None, pages_per_division: int = 1) -> int:
        """Fill the frontier from the ranked ladder. Safe to re-run."""
        batch: list[dict] = []
        added = 0
        unresolved = 0
        for entry in self.client.iter_ranked_ladder(tiers, pages_per_division):
            if not entry.get("puuid") and entry.get("summonerId"):
                summoner = self.client.summoner_by_id(entry["summonerId"])
                if summoner and summoner.get("puuid"):
                    entry["puuid"] = summoner["puuid"]
                else:
                    unresolved += 1
            if not entry.get("puuid"):
                continue
            batch.append(entry)
            if len(batch) >= 2000:
                added += self._flush_seed(batch)
                batch = []
            if self.stop_requested:
                break
        added += self._flush_seed(batch)
        if unresolved:
            log.warning("%d ladder entries had no puuid and could not be resolved", unresolved)
        log.info("Seeded %d new players (frontier now %d queued)", added,
                 self.store.progress()["players_queued"])
        return added

    def _flush_seed(self, batch: list[dict]) -> int:
        if not batch:
            return 0
        added = self.store.add_players(batch, source="seed")
        self.store.add_rank_snapshots(batch)
        return added

    # -- main loop ---------------------------------------------------------
    def run(self, target_matches: int, max_seconds: float | None = None) -> dict:
        start = time.monotonic()
        stored_at_start = self.store.progress()["matches_stored"]
        stored = stored_at_start
        last_report = start

        log.info(
            "Starting crawl: target %d stored matches (currently %d), %d workers",
            target_matches, stored, self.workers,
        )

        while not self.stop_requested and stored - stored_at_start < target_matches:
            if max_seconds and time.monotonic() - start > max_seconds:
                log.info("Time budget reached.")
                break

            progress = self.store.progress()
            if progress["matches_pending"] < self.pending_low_water:
                if not self._expand_frontier():
                    if progress["matches_pending"] == 0:
                        log.warning("Frontier exhausted - run `seed` again or widen the window.")
                        break

            _, fetched = self._drain_matches(target_matches - (stored - stored_at_start))
            stored += fetched

            if fetched == 0 and self.store.progress()["matches_pending"] == 0:
                continue

            now = time.monotonic()
            if now - last_report > 20:
                self._report(stored - stored_at_start, now - start)
                last_report = now

        elapsed = time.monotonic() - start
        self._report(stored - stored_at_start, elapsed, final=True)
        return {
            "stored_this_run": stored - stored_at_start,
            "elapsed_s": elapsed,
            **self.store.progress(),
            **{f"http_{k}": v for k, v in self.client.stats.items()},
        }

    # -- discover / densify ------------------------------------------------
    def discover(self, max_players: int) -> dict:
        """Phase 1: spend requests on match-ID lookups only.

        One request per player returns up to 100 match IDs *and* tells us that
        this player was in each of them. Recording those edges builds a coverage
        map, so phase 2 can fetch only matches where several participants already
        have deep history - instead of fetching matches blind, which is what made
        the breadth crawl produce nothing usable per player.
        """
        start = time.monotonic()
        processed = 0
        new_matches = 0
        last_report = start
        log.info("Discover: processing up to %d players (match-ids only)", max_players)

        while processed < max_players and not self.stop_requested:
            puuids = self.store.claim_players(min(self.player_batch, max_players - processed))
            if not puuids:
                log.warning("No unprocessed players left - seed more ladder pages.")
                break

            results = self._map(self._fetch_match_ids, puuids)
            pairs: list[tuple[str, str]] = []
            done: list[str] = []
            counts: list[int] = []
            for puuid, ids, error in results:
                if error:
                    log.debug("match_ids failed for %s: %s", puuid[:8], error)
                    continue
                ids = ids or []
                pairs.extend((mid, puuid) for mid in ids)
                done.append(puuid)
                counts.append(len(ids))

            new_matches += self.store.add_discoveries(pairs)
            self.store.mark_players_processed(done, counts)
            processed += len(done)

            now = time.monotonic()
            if now - last_report > 20:
                rate = processed / (now - start) * 60
                log.info(
                    "... %d players processed (%.0f/min) | %d matches mapped | http %d req",
                    processed, rate, new_matches, self.client.stats["requests"],
                )
                last_report = now

        elapsed = time.monotonic() - start
        log.info("Discover done: %d players, %d matches mapped, %.0fs",
                 processed, new_matches, elapsed)
        return {"players_processed": processed, "matches_mapped": new_matches,
                "elapsed_s": elapsed}

    def densify(
        self,
        target_matches: int,
        min_coverage: int = 5,
        enqueue_players: bool = True,
        use_plan: bool = False,
    ) -> dict:
        """Phase 2: fetch only matches with at least `min_coverage` known players."""
        start = time.monotonic()
        stored_at_start = self.store.progress()["matches_stored"]
        stored = stored_at_start
        attempted = 0
        last_report = start
        log.info(
            "Densify: target %d matches with coverage >= %d", target_matches, min_coverage
        )

        while not self.stop_requested and stored - stored_at_start < target_matches:
            candidates, fetched = self._drain_matches(
                target_matches - (stored - stored_at_start),
                min_coverage=min_coverage,
                enqueue_players=enqueue_players,
                use_plan=use_plan,
            )
            if candidates == 0:
                log.warning(
                    "Nothing left to fetch (%s). Loosen the thresholds, or run "
                    "`discover` on more players.",
                    "plan exhausted" if use_plan else f"coverage >= {min_coverage}",
                )
                break
            attempted += candidates
            stored += fetched

            now = time.monotonic()
            if now - last_report > 20:
                self._report(stored - stored_at_start, now - start)
                last_report = now

        elapsed = time.monotonic() - start
        self._report(stored - stored_at_start, elapsed, final=True)
        return {"stored_this_run": stored - stored_at_start, "attempted": attempted,
                "elapsed_s": elapsed, **self.store.progress()}

    def plan_core(self, min_players_per_match: int, min_matches_per_player: int) -> dict:
        """Pick the densest cohort in the discovery graph, offline and free.

        Peels the player/match bipartite graph to a k-core: every match keeps at
        least `min_players_per_match` known participants and every player keeps at
        least `min_matches_per_player` matches. Fetching that core buys far more
        per-player history per request than fetching by raw coverage, because it
        optimises player depth and match coverage jointly.
        """
        edges = self.store.discovery_edges()
        if edges.empty:
            log.warning("No discovery edges - run `discover` first.")
            return {"matches": 0, "players": 0}

        before = (edges.puuid.nunique(), edges.match_id.nunique())
        while True:
            counts = edges.groupby("match_id").size()
            edges2 = edges[edges.match_id.isin(counts[counts >= min_players_per_match].index)]
            counts = edges2.groupby("puuid").size()
            edges3 = edges2[edges2.puuid.isin(counts[counts >= min_matches_per_player].index)]
            if len(edges3) == len(edges) or edges3.empty:
                edges = edges3
                break
            edges = edges3

        if edges.empty:
            log.warning(
                "Core is empty at (match>=%d, player>=%d) - loosen the thresholds.",
                min_players_per_match, min_matches_per_player,
            )
            return {"matches": 0, "players": 0}

        players = int(edges.puuid.nunique())
        matches = int(edges.match_id.nunique())
        planned = self.store.set_plan(edges.match_id.unique().tolist())
        pending = self.store.plan_remaining()
        log.info(
            "Core (match>=%d, player>=%d): %d players, %d matches (from %d/%d), "
            "%.1f games per player | %d still to fetch",
            min_players_per_match, min_matches_per_player, players, matches,
            before[0], before[1], len(edges) / players, pending,
        )
        return {
            "players": players, "matches": matches, "planned": planned,
            "pending": pending, "games_per_player": round(len(edges) / players, 1),
        }

    def plan_topup(self, min_coverage: int = 0, games: int = 10) -> dict:
        """Extend the plan so target matches reach full 10/10 player coverage.

        A target match is one whose ten participants have all been processed. For
        each participant still short of `games` prior matches once the current plan
        is fetched, greedily add the history matches that serve the most short
        players at once. Players with fewer than `games` ranked games in the whole
        lookback window cannot be topped up at all and are reported, not chased.
        """
        edges = self.store.discovery_edges()
        if edges.empty:
            return {"added": 0}
        plan = set(self.store.planned_match_ids())
        resolved = set(self.store.resolved_match_ids())
        targets = self.store.fully_processed_participants()
        if targets.empty:
            log.warning("No fully-processed matches yet - nothing to top up.")
            return {"added": 0}

        edges["in_plan"] = edges.match_id.isin(plan)
        post = edges[edges.in_plan].groupby("puuid").size()
        targets["post"] = targets.puuid.map(post).fillna(0).astype(int)

        covered = targets.assign(ok=targets.post >= games).groupby("match_id").ok.sum()
        keep = set(covered[covered >= min_coverage].index) if min_coverage > 0 else set(covered.index)
        players = set(targets[targets.match_id.isin(keep)].puuid)

        need = {p: games - int(post.get(p, 0)) for p in players if int(post.get(p, 0)) < games}
        short_at_start = len(need)
        candidates = edges[(~edges.in_plan) & (~edges.match_id.isin(resolved))]
        candidates = candidates[candidates.puuid.isin(need)]

        chosen: list[str] = []
        if not candidates.empty:
            by_match = candidates.groupby("match_id").puuid.apply(list)
            order = candidates.groupby("match_id").size().sort_values(ascending=False).index
            for mid in order:
                if not need:
                    break
                hits = [p for p in by_match[mid] if p in need]
                if not hits:
                    continue
                chosen.append(mid)
                for p in hits:
                    need[p] -= 1
                    if need[p] <= 0:
                        del need[p]

        added = self.store.extend_plan(chosen)
        log.info(
            "Top-up: %d target matches, %d players short of %d games -> %d matches added "
            "to the plan (%d players cannot reach it: too few ranked games in the window)",
            len(keep), short_at_start, games, added, len(need),
        )
        return {
            "target_matches": len(keep), "players_short": short_at_start,
            "added": added, "unreachable_players": len(need),
            "plan_pending": self.store.plan_remaining(),
        }

    def _fetch_match_ids(self, puuid: str) -> list[str]:
        start_time = int(
            datetime.now(tz=timezone.utc).timestamp() - self.lookback_days * 86400
        )
        return self.client.match_ids(
            puuid, queue=self.queue_id, start_time=start_time, count=self.ids_per_player
        )

    def _expand_frontier(self) -> bool:
        """Pull match IDs for the next batch of players. False if nobody is left."""
        puuids = self.store.claim_players(self.player_batch)
        if not puuids:
            return False

        results = self._map(self._fetch_match_ids, puuids)
        pairs: list[tuple[str, str]] = []
        done_puuids: list[str] = []
        done_counts: list[int] = []
        for puuid, ids, error in results:
            if error:
                log.debug("match_ids failed for %s: %s", puuid[:8], error)
                continue
            ids = ids or []
            pairs.extend((mid, puuid) for mid in ids)
            done_puuids.append(puuid)
            done_counts.append(len(ids))

        # Record the edges here too, so even the breadth crawl accumulates coverage.
        new_ids = self.store.add_discoveries(pairs)
        self.store.mark_players_processed(done_puuids, done_counts)
        log.debug("Frontier: %d players -> %d ids (%d new)", len(done_puuids), len(pairs), new_ids)
        return True

    def _drain_matches(
        self,
        remaining: int,
        min_coverage: int = 0,
        enqueue_players: bool = True,
        use_plan: bool = False,
    ) -> tuple[int, int]:
        """Fetch and store a batch of pending matches.

        Returns (candidates_attempted, matches_stored) so callers can tell an empty
        queue apart from a batch that was fetched but entirely rejected.
        """
        limit = max(1, min(self.match_batch, remaining))
        match_ids = self.store.pending_match_ids(
            limit, min_coverage=min_coverage, use_plan=use_plan
        )
        if not match_ids:
            return 0, 0

        results = self._map(self.client.match, match_ids)

        state_updates: list[tuple[str, str, str | None]] = []
        new_players: list[dict] = []
        match_rows: list[tuple] = []
        all_participants: list[tuple] = []
        raw_rows: list[tuple[str, str | None, bytes]] = []

        for match_id, payload, error in results:
            if error:
                state_updates.append((match_id, "error", error[:200]))
                continue
            if payload is None:
                state_updates.append((match_id, "rejected", "404"))
                continue
            try:
                patch = validate(payload, self.queue_id, self.min_patch)
            except RejectedMatch as exc:
                state_updates.append((match_id, "rejected", str(exc)[:200]))
                continue
            except Exception as exc:  # noqa: BLE001
                state_updates.append((match_id, "error", f"parse: {exc}"[:200]))
                continue

            try:
                match_row, participant_rows = flatten(match_id, payload, patch)
            except Exception as exc:  # noqa: BLE001
                state_updates.append((match_id, "error", f"parse: {exc}"[:200]))
                continue

            match_rows.append(match_row)
            all_participants.extend(participant_rows)
            raw_rows.append((match_id, patch, self.store.compress(payload)))
            state_updates.append((match_id, "stored", None))
            new_players.extend({"puuid": p} for p in participant_puuids(payload))

        # One bulk write per batch rather than ~22 statements per match.
        stored = self.store.store_matches(match_rows, all_participants, raw_rows)
        # Transient failures go back in the queue; only 404s and validation
        # rejects are final, so an outage costs time rather than coverage.
        failures = [(mid, reason or "") for mid, st, reason in state_updates if st == "error"]
        final = [(mid, st, reason) for mid, st, reason in state_updates if st != "error"]
        self.store.set_match_states(final)
        if failures:
            requeued, dead = self.store.register_failures(failures)
            if requeued:
                log.warning("%d fetches failed transiently - requeued for retry", requeued)
            if dead:
                log.error("%d matches gave up after repeated failures", dead)
        if new_players and enqueue_players:
            self.store.add_players(new_players, source="snowball")
        return len(match_ids), stored

    def _report(self, stored_this_run: int, elapsed: float, final: bool = False) -> None:
        progress = self.store.progress()
        rate = stored_this_run / elapsed * 60 if elapsed > 0 else 0.0
        log.info(
            "%s %d matches this run (%.0f/min) | total stored %d | pending %d | "
            "rejected %d | errors %d | frontier %d | http %d req, %d 429s",
            "DONE:" if final else "...",
            stored_this_run,
            rate,
            progress["matches_stored"],
            progress["matches_pending"],
            progress["matches_rejected"],
            progress["matches_error"],
            progress["players_queued"],
            self.client.stats["requests"],
            self.client.stats["rate_limited"],
        )
