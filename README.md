# LoL Win Predictor — data collection

Stage one only: a resumable, rate-limited crawler that builds a corpus of ranked
matches. Feature engineering, training and serving come later and read from the
same DuckDB file.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # POSIX
```

Put your key in `config/.env` (gitignored):

```
RIOT_API_KEY=RGAPI-...
RIOT_PLATFORM=euw1      # platform routing: league / summoner / spectator
RIOT_REGION=europe      # regional routing: account / match
```

Development keys expire every 24 hours. If the crawler suddenly reports 403, that
is almost always why — regenerate the key and restart; progress is not lost.

## Use

```bash
.venv/Scripts/python -m lolpred.cli check                  # key + routing sanity check
.venv/Scripts/python -m lolpred.cli seed --pages 2         # fill the frontier from the ladder
.venv/Scripts/python -m lolpred.cli crawl --target 50000   # crawl until 50k matches are stored
.venv/Scripts/python -m lolpred.cli status                 # progress + patch breakdown
```

`crawl` seeds automatically the first time. Ctrl-C finishes the current batch and
exits cleanly; rerunning the same command picks up exactly where it stopped, so
long crawls can be split across days (and across key regenerations).

Useful flags: `--workers` (parallel HTTP, the limiter is still the ceiling),
`--ids-per-player` (lower = more players, broader rank spread; higher = fewer
API calls per match), `--max-seconds` (time budget), `-v` (debug logging).

## How it works

```
ladder seed ──► crawl_players ──► match-v5 ids ──► crawl_matches ──► match-v5 detail
                     ▲                                                     │
                     └──────────── 10 participants per match ◄─────────────┘
```

Snowballing means the frontier grows about ten players per match stored, so it
never runs dry. Seeding across every tier is what keeps the sample from
collapsing into a single rank band.

### Rate limiting

`lolpred/ingestion/rate_limit.py` enforces the app limits from `APP_RATE_LIMITS`
as sliding windows, and *learns* each endpoint's method limit from the
`X-Method-Rate-Limit` response header, so moving to a production key needs no
retuning. A 429 backdates the offending window by `Retry-After`, which parks
every worker rather than letting them hammer the endpoint.

### Storage

One DuckDB file (`data/lol.duckdb`), four layers:

| table | holds |
|---|---|
| `raw_matches` | gzipped Match-V5 JSON, so features can be re-derived without re-crawling |
| `matches`, `participants` | flattened rows for training |
| `player_ranks` | time-stamped rank snapshots seen while seeding |
| `crawl_players`, `crawl_matches` | the frontier and per-match outcome — this is what makes restarts free |

Gzip keeps raw JSON to roughly 20–30 KB per match, so 50k matches lands near
1–1.5 GB.

### What gets rejected

`lolpred/storage/transform.py` drops anything that would poison the training set:
wrong queue, under 15 minutes (remakes and early surrenders), not ten
participants, a missing or duplicated `teamPosition`, or below `MIN_PATCH`.
Rejections are recorded in `crawl_matches` with a reason rather than silently
dropped, so `status` shows whether a filter is too aggressive.

## Throughput

Every stored match costs about one request, plus one `match/ids` call per player.
A development key (20 req/s, 100 req/2min) is capped by the 2-minute window at
50 requests/minute. After `match/ids` calls and rejected matches, that works out
to roughly **2k stored matches/hour**, so 50k matches is about a day of wall
clock — split it across runs, it resumes for free. A production key lifts that by orders of magnitude. Raising `--workers`
helps only until the limiter is saturated; 8 is plenty for a dev key.

## Running the crawl detached (multi-day)

A 200k-match crawl at development-key limits takes days, so run it as a detached
process that survives closing the terminal:

```powershell
$p = Start-Process -FilePath "$PWD\.venv\Scripts\python.exe" `
  -ArgumentList '-u','-m','lolpred.cli','crawl','--target','200000','--ids-per-player','100','--workers','8' `
  -WorkingDirectory "$PWD" `
  -RedirectStandardOutput "$PWD\data\crawl.detached.log" `
  -RedirectStandardError  "$PWD\data\crawl.detached.err" `
  -WindowStyle Hidden -PassThru
$p.Id | Out-File -Encoding utf8 "$PWD\data\crawl.pid"
```

Progress logging goes to **stderr**, so the interesting file is
`data/crawl.detached.err`:

```powershell
Get-Content data\crawl.detached.err -Tail 5 -Wait      # follow live
Get-Process -Id (Get-Content data\crawl.pid)           # still alive?
Stop-Process -Id (Get-Content data\crawl.pid)          # stop it
```

Stopping mid-batch loses at most the batch in flight; everything already written
is committed, and relaunching the same command resumes from there.

Only one crawler at a time: DuckDB takes an exclusive write lock, so a second
process exits immediately with an IO error. `lolpred.cli status` detects the lock
and tells you to read the log instead of failing with a traceback.

### Swapping in a higher-limit key

Edit `config/.env` with the new key and its app limits, then stop and relaunch:

```
RIOT_API_KEY=RGAPI-...
APP_RATE_LIMITS=500:10,30000:600
```

Nothing else needs changing. The limiter derives its pacing interval from these
values (`max(span/count) * 1.05`), and method limits are still learned from
response headers.
