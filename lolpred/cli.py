"""Command line entry point:  python -m lolpred.cli <command>"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from .config import settings
from .features.build import build as build_features
from .ingestion.crawler import Crawler
from .ingestion.riot_client import APEX_TIERS, TIERS_WITH_DIVISIONS, RiotAuthError, RiotClient
from .storage.db import Store


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def make_client() -> RiotClient:
    return RiotClient(
        api_key=settings.require_key(),
        platform=settings.platform,
        region=settings.region,
        app_limits=settings.app_limits,
    )


def make_crawler(store: Store, args: argparse.Namespace) -> Crawler:
    return Crawler(
        client=make_client(),
        store=store,
        queue_id=settings.queue_id,
        min_patch=settings.min_patch,
        lookback_days=settings.lookback_days,
        workers=getattr(args, "workers", 8),
        ids_per_player=getattr(args, "ids_per_player", 30),
    )


def cmd_check(args: argparse.Namespace) -> int:
    """Verify the key, routing and rate limits with a couple of cheap calls."""
    client = make_client()
    print(f"platform={settings.platform} region={settings.region} queue={settings.queue_id}")
    try:
        league = client.apex_league("challenger")
    except RiotAuthError as exc:
        print(f"AUTH FAILED: {exc}")
        return 1
    entries = (league or {}).get("entries", [])
    print(f"challenger ladder: {len(entries)} entries")
    if not entries:
        print("No entries returned - check the platform routing value.")
        return 1

    sample = entries[0]
    puuid = sample.get("puuid")
    if not puuid and sample.get("summonerId"):
        puuid = (client.summoner_by_id(sample["summonerId"]) or {}).get("puuid")
    if not puuid:
        print("Ladder entries carry no puuid; seeding will need summoner-v4 lookups.")
        return 1

    ids = client.match_ids(puuid, queue=settings.queue_id, count=5)
    print(f"sample player has {len(ids)} recent ranked match ids")
    if ids:
        match = client.match(ids[0])
        info = (match or {}).get("info", {})
        print(f"sample match {ids[0]}: patch {info.get('gameVersion')}, "
              f"queue {info.get('queueId')}, {info.get('gameDuration')}s")
    print(f"http stats: {client.stats}")
    print("OK")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    settings.ensure_dirs()
    tiers = args.tiers.split(",") if args.tiers else None
    with Store(settings.db_path) as store:
        crawler = make_crawler(store, args)
        crawler.install_signal_handler()
        crawler.seed(tiers=tiers, pages_per_division=args.pages)
        print(store.progress())
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    settings.ensure_dirs()
    with Store(settings.db_path) as store:
        crawler = make_crawler(store, args)
        crawler.install_signal_handler()

        if store.progress()["players_queued"] == 0 and store.progress()["matches_pending"] == 0:
            tiers = args.tiers.split(",") if args.tiers else None
            print(f"Frontier is empty - seeding from the ladder first (tiers: {tiers or 'all'}).")
            crawler.seed(tiers=tiers, pages_per_division=args.pages)

        summary = crawler.run(target_matches=args.target, max_seconds=args.max_seconds)
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    """Phase 1: build the coverage map with match-ids calls only."""
    settings.ensure_dirs()
    with Store(settings.db_path) as store:
        crawler = make_crawler(store, args)
        crawler.install_signal_handler()
        summary = crawler.discover(max_players=args.players)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("\npending matches by coverage (processed players known to be in them):")
        for cov, count in store.coverage_histogram()[:12]:
            print(f"  >= {cov:2} players: {count}")
    return 0


def cmd_densify(args: argparse.Namespace) -> int:
    """Phase 2: fetch only well-covered matches."""
    settings.ensure_dirs()
    with Store(settings.db_path) as store:
        crawler = make_crawler(store, args)
        crawler.install_signal_handler()
        summary = crawler.densify(
            target_matches=args.target,
            min_coverage=args.min_coverage,
            enqueue_players=not args.no_enqueue,
            use_plan=args.use_plan,
        )
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Choose the densest cohort to fetch, using the discovery graph only."""
    settings.ensure_dirs()
    with Store(settings.db_path) as store:
        crawler = make_crawler(store, args)
        summary = crawler.plan_core(args.core_match, args.core_player)
        for key, value in summary.items():
            print(f"{key}: {value}")
        if summary.get("pending"):
            hours = summary["pending"] / (60 * 47.6)
            print(f"estimated fetch time at current limits: {hours:.1f}h")
    return 0


def cmd_topup(args: argparse.Namespace) -> int:
    """Extend the plan so target matches reach full 10/10 coverage."""
    settings.ensure_dirs()
    with Store(settings.db_path) as store:
        crawler = make_crawler(store, args)
        summary = crawler.plan_topup(min_coverage=args.min_coverage, games=args.games)
        for key, value in summary.items():
            print(f"{key}: {value}")
        if summary.get("plan_pending"):
            print(f"total plan still to fetch: {summary['plan_pending']:,} "
                  f"({summary['plan_pending']/(60*47.6):.1f}h at current limits)")
    return 0


def cmd_extras(args: argparse.Namespace) -> int:
    """Derive the advanced per-participant metrics from stored raw payloads."""
    from .storage.extras import EXTRA_COLS, iter_raw

    db = Path(args.db) if args.db else settings.db_path
    with Store(db) as store:
        total = 0
        for batch in iter_raw(store, batch=args.batch, only_missing=not args.rebuild):
            if not batch:
                continue
            total += store._insert_new(
                "participant_extras", EXTRA_COLS, batch, ["match_id", "puuid"]
            )
            if total % 20000 < 10:
                print(f"  ... {total:,} rows")
        print(f"participant_extras: {store.count('participant_extras'):,} rows "
              f"({total:,} added this run)")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Build the feature table from stored matches."""
    db = Path(args.db) if args.db else settings.db_path
    with Store(db, read_only=True) as store:
        frame = build_features(
            store,
            min_history=args.min_history,
            infer_roles=not args.post_game_roles,
            save_state=args.save_state,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out)
    print(f"wrote {len(frame):,} rows x {len(frame.columns)} cols -> {out}")
    print(f"blue win rate {frame.blue_win.mean():.4f}")
    print("coverage per side (players with history):")
    print(frame.groupby(["blue_coverage"]).size().head(12).to_string())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the baseline ladder and the gradient boosted model."""
    from .training.dataset import make_split
    from .training.train import train as run_training

    frame = pd.read_parquet(args.features)
    print(f"loaded {len(frame):,} feature rows")
    split = make_split(
        frame,
        min_covered=args.min_covered,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        augment=not args.no_augment,
    )
    print(split.describe())
    print()
    run_training(split, settings.model_dir, label=args.label)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the prediction web service."""
    import uvicorn

    uvicorn.run("lolpred.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_prioritise(args: argparse.Namespace) -> int:
    """Reorder the plan so target-match coverage lands early."""
    settings.ensure_dirs()
    with Store(settings.db_path) as store:
        targets = store.fully_processed_participants()
        if targets.empty:
            print("No fully-processed matches yet - nothing to prioritise.")
            return 1
        summary = store.prioritise_plan(targets.puuid.unique().tolist())
        print(f"target players: {summary['target_players']:,}")
        print("pending plan matches by priority (target players carried):")
        for prio, count in summary["pending_by_priority"]:
            print(f"  priority {prio:2}: {count:,}")
    return 0


def cmd_requeue(args: argparse.Namespace) -> int:
    """Return matches that failed transiently to the pending queue."""
    with Store(settings.db_path) as store:
        n = store.requeue_errors()
        print(f"requeued {n} matches that had been marked as errors")
        print(store.progress())
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """Show how much per-player history we actually have."""
    if not settings.db_path.exists():
        print(f"No database at {settings.db_path} yet.")
        return 1
    try:
        store = Store(settings.db_path, read_only=True)
    except duckdb.IOException:
        print("Database is locked by a running crawler. Stop it first.")
        return 1
    with store:
        print("pending matches by coverage:")
        rows = store.coverage_histogram()
        if not rows:
            print("  (none - run `discover` first)")
        total = sum(c for _, c in rows)
        cumulative = 0
        for cov, count in rows:
            cumulative += count
            print(f"  coverage >= {cov:2}: {cumulative:8} matches  (exactly {cov}: {count})")
        print(f"  total pending: {total}")

        print("\nstored matches by number of participants we have processed:")
        for cov, count in store.stored_coverage_histogram():
            print(f"  {cov:2} of 10 covered: {count}")

        players = store.con.execute(
            """
            SELECT count(*) AS known,
                   count(*) FILTER (WHERE processed_at IS NOT NULL) AS processed
            FROM crawl_players
            """
        ).fetchone()
        print(f"\nplayers known {players[0]}, processed {players[1]}")
        edges = store.count("match_discoveries")
        print(f"discovery edges: {edges}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not settings.db_path.exists():
        print(f"No database at {settings.db_path} yet - run `seed` or `crawl` first.")
        return 1
    try:
        store = Store(settings.db_path, read_only=True)
    except duckdb.IOException:
        print("Database is locked by the running crawler (DuckDB allows one writer).")
        print("Watch data/crawl.log instead, or stop the crawl first.")
        return 1
    with store:
        progress = store.progress()
        for key, value in progress.items():
            print(f"{key:18} {value}")
        rows = store.patch_breakdown()
        if rows:
            print("\nmatches by patch")
            for patch, count in rows:
                print(f"  {patch:8} {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lolpred")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify API key and routing").set_defaults(func=cmd_check)

    p_seed = sub.add_parser("seed", help="fill the frontier from the ranked ladder")
    p_seed.add_argument(
        "--tiers",
        default=None,
        help=f"comma separated, default all: {','.join(APEX_TIERS + TIERS_WITH_DIVISIONS)}",
    )
    p_seed.add_argument("--pages", type=int, default=1, help="league pages per division")
    p_seed.add_argument("--workers", type=int, default=8)
    p_seed.set_defaults(func=cmd_seed)

    p_crawl = sub.add_parser("crawl", help="crawl matches until the target is reached")
    p_crawl.add_argument("--target", type=int, default=50_000, help="matches to store this run")
    p_crawl.add_argument("--workers", type=int, default=8)
    p_crawl.add_argument("--ids-per-player", type=int, default=30, dest="ids_per_player")
    p_crawl.add_argument("--max-seconds", type=float, default=None)
    p_crawl.add_argument("--pages", type=int, default=1)
    p_crawl.add_argument("--tiers", default=None, help="tiers to seed with, if seeding is needed")
    p_crawl.set_defaults(func=cmd_crawl)

    p_disc = sub.add_parser(
        "discover", help="phase 1: map matches to players with match-ids calls only"
    )
    p_disc.add_argument("--players", type=int, default=100_000, help="players to process")
    p_disc.add_argument("--workers", type=int, default=8)
    p_disc.add_argument("--ids-per-player", type=int, default=100, dest="ids_per_player")
    p_disc.set_defaults(func=cmd_discover)

    p_dens = sub.add_parser(
        "densify", help="phase 2: fetch only matches with enough known participants"
    )
    p_dens.add_argument("--target", type=int, default=50_000)
    p_dens.add_argument(
        "--min-coverage", type=int, default=5, dest="min_coverage",
        help="minimum processed participants a match must have (of 10)",
    )
    p_dens.add_argument("--workers", type=int, default=8)
    p_dens.add_argument(
        "--use-plan", action="store_true", dest="use_plan",
        help="fetch only the cohort chosen by `plan`",
    )
    p_dens.add_argument(
        "--no-enqueue", action="store_true", dest="no_enqueue",
        help="do not add newly seen participants to the player pool",
    )
    p_dens.set_defaults(func=cmd_densify)

    p_plan = sub.add_parser(
        "plan", help="pick the densest cohort to fetch (k-core on the discovery graph)"
    )
    p_plan.add_argument("--core-match", type=int, default=3, dest="core_match",
                        help="min known participants a match must keep")
    p_plan.add_argument("--core-player", type=int, default=5, dest="core_player",
                        help="min matches a player must keep")
    p_plan.set_defaults(func=cmd_plan)

    p_top = sub.add_parser(
        "topup", help="extend the plan so target matches reach full 10/10 coverage"
    )
    p_top.add_argument("--min-coverage", type=int, default=0, dest="min_coverage",
                       help="only top up targets already at this coverage (0 = all)")
    p_top.add_argument("--games", type=int, default=10,
                       help="prior games required per player")
    p_top.set_defaults(func=cmd_topup)

    p_ex = sub.add_parser("extras", help="derive advanced metrics from raw payloads")
    p_ex.add_argument("--db", default=None)
    p_ex.add_argument("--batch", type=int, default=500)
    p_ex.add_argument("--rebuild", action="store_true", help="redo every match")
    p_ex.set_defaults(func=cmd_extras)

    p_feat = sub.add_parser("features", help="build the feature table")
    p_feat.add_argument("--out", default="data/features.parquet")
    p_feat.add_argument("--db", default=None, help="database to read (default: configured)")
    p_feat.add_argument("--min-history", type=int, default=10, dest="min_history",
                        help="prior games before a player counts as covered")
    p_feat.add_argument("--save-state", default=None, dest="save_state",
                        help="also write the serving state snapshot here")
    p_feat.add_argument("--post-game-roles", action="store_true", dest="post_game_roles",
                        help="use Riot's teamPosition instead of inferring (not serve-legal)")
    p_feat.set_defaults(func=cmd_features)

    p_srv = sub.add_parser("serve", help="run the prediction web service")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--reload", action="store_true")
    p_srv.set_defaults(func=cmd_serve)

    p_train = sub.add_parser("train", help="train and calibrate the model")
    p_train.add_argument("--features", default="data/features.parquet")
    p_train.add_argument("--min-covered", type=int, default=0, dest="min_covered",
                         help="require this many covered players per side")
    p_train.add_argument("--test-frac", type=float, default=0.2, dest="test_frac")
    p_train.add_argument("--val-frac", type=float, default=0.1, dest="val_frac")
    p_train.add_argument("--no-augment", action="store_true", dest="no_augment")
    p_train.add_argument("--label", default="lolpred")
    p_train.set_defaults(func=cmd_train)

    sub.add_parser(
        "prioritise", help="reorder the plan so target-match coverage lands early"
    ).set_defaults(func=cmd_prioritise)

    sub.add_parser(
        "requeue", help="retry matches previously marked as errors"
    ).set_defaults(func=cmd_requeue)

    sub.add_parser("coverage", help="per-player history coverage report").set_defaults(
        func=cmd_coverage
    )
    sub.add_parser("status", help="show crawl progress").set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; rerun the same command to continue.")
        return 130
    except RiotAuthError as exc:
        print(f"\n{exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
