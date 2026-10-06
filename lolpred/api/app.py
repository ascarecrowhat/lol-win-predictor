"""FastAPI service: paste a Riot ID or u.gg profile link, get a win probability.

Beta scope, deliberately: EUW ranked solo/duo only, which is what the model was
trained on. Anything outside that is refused with a reason rather than answered
with a number the model has no basis for.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import settings
from ..ingestion.riot_client import RiotAuthError, RiotClient, RiotError
from ..serving.live import LivePredictor, NotEligible
from ..storage.db import Store

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="LoL Win Predictor (beta)", version="0.1.0")
_predictor: LivePredictor | None = None

# One cold prediction costs ~110 Riot requests, and a development key allows 100
# per two minutes. A public demo therefore has to serialise work and turn people
# away politely rather than exhaust the key and fail for everyone.
PREDICT_LOCK = threading.Semaphore(1)
GLOBAL_WINDOW_S = 600.0
# Set DEMO_THROTTLE=off (or either limit to 0) while testing. Leave the defaults
# on for anything public: one cold prediction is ~110 Riot requests against a key
# that allows 100 every two minutes, so two concurrent visitors break it for both.
THROTTLE_ON = os.getenv("DEMO_THROTTLE", "on").strip().lower() not in {"off", "0", "false", "no"}
GLOBAL_MAX = int(os.getenv("MAX_PREDICTIONS_PER_10MIN", "6"))
PER_IP_COOLDOWN_S = float(os.getenv("PER_IP_COOLDOWN_S", "300"))
LOCK_WAIT_S = float(os.getenv("PREDICT_LOCK_WAIT_S", "5"))
_recent: deque[float] = deque()
_by_ip: dict[str, float] = {}
_guard = threading.Lock()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_quota(request: Request) -> None:
    if not THROTTLE_ON:
        return
    ip = _client_ip(request)
    now = time.monotonic()
    with _guard:
        while _recent and now - _recent[0] > GLOBAL_WINDOW_S:
            _recent.popleft()
        if GLOBAL_MAX > 0 and len(_recent) >= GLOBAL_MAX:
            wait = int(GLOBAL_WINDOW_S - (now - _recent[0])) + 1
            raise HTTPException(
                429,
                f"Demo quota reached ({GLOBAL_MAX} predictions per 10 minutes). "
                f"Try again in about {wait}s - the Riot development key only allows "
                "100 requests per 2 minutes and one prediction uses roughly 110.",
                headers={"Retry-After": str(wait)},
            )
        last = _by_ip.get(ip)
        if PER_IP_COOLDOWN_S > 0 and last is not None and now - last < PER_IP_COOLDOWN_S:
            wait = int(PER_IP_COOLDOWN_S - (now - last)) + 1
            raise HTTPException(
                429,
                f"One prediction per {int(PER_IP_COOLDOWN_S)}s per visitor. "
                f"Try again in about {wait}s.",
                headers={"Retry-After": str(wait)},
            )
        _recent.append(now)
        _by_ip[ip] = now


class PredictRequest(BaseModel):
    target: str
    top_up: bool = True


def get_predictor() -> LivePredictor:
    global _predictor
    if _predictor is None:
        model_path = settings.model_dir / "lolpred.joblib"
        if not model_path.exists():
            candidates = sorted(settings.model_dir.glob("lolpred*.joblib"))
            if not candidates:
                raise HTTPException(503, "No trained model found - run `lolpred train` first.")
            model_path = candidates[0]
        snapshot = settings.model_dir / "state.joblib"
        if not snapshot.exists():
            raise HTTPException(
                503, "No serving state snapshot - run `lolpred features --save-state`."
            )
        if not settings.api_key:
            raise HTTPException(
                503,
                "RIOT_API_KEY is not set on this deployment. Add it in the service "
                "environment settings and redeploy.",
            )
        _predictor = LivePredictor(
            client=RiotClient(
                api_key=settings.require_key(),
                platform=settings.platform,
                region=settings.region,
                app_limits=settings.app_limits,
            ),
            # DuckDB allows a single writer, so while a crawl is running the
            # service needs its own copy. SERVE_DB_PATH points at one.
            store=Store(os.getenv("SERVE_DB_PATH") or settings.db_path),
            snapshot_path=snapshot,
            model_path=model_path,
            history_per_player=int(os.getenv("HISTORY_PER_PLAYER", "10")),
        )
        log.info("Loaded %s with %d features", model_path.name, len(_predictor.features))
    return _predictor


@app.get("/health")
def health() -> dict:
    model_dir = settings.model_dir
    return {
        "status": "ok",
        "platform": settings.platform,
        "queue": settings.queue_id,
        "model_present": any(model_dir.glob("lolpred*.joblib")),
        "state_present": (model_dir / "state.joblib").exists(),
        "key_present": bool(settings.api_key),
        "throttle": "on" if THROTTLE_ON else "off",
    }


@app.post("/predict")
def predict(req: PredictRequest, request: Request) -> dict:
    check_quota(request)
    predictor = get_predictor()
    if not PREDICT_LOCK.acquire(timeout=LOCK_WAIT_S):
        raise HTTPException(
            429, "Another prediction is in progress; the rate limit allows only one at a time.",
            headers={"Retry-After": "60"},
        )
    try:
        return predictor.predict(req.target, top_up=req.top_up)
    except NotEligible as exc:
        # Expected outcome, not a server fault: no live game, wrong region, wrong queue.
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RiotAuthError as exc:
        raise HTTPException(502, f"Riot API key rejected: {exc}") from exc
    except RiotError as exc:
        raise HTTPException(502, f"Riot API error: {exc}") from exc
    finally:
        PREDICT_LOCK.release()


@app.get("/")
def index() -> FileResponse:
    page = ROOT / "web" / "index.html"
    if not page.exists():
        raise HTTPException(404, "web/index.html is missing")
    return FileResponse(page)
