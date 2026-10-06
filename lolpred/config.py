"""Runtime configuration, loaded from environment / config/.env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# config/.env wins over a plain .env at the repo root; neither overrides a real env var.
load_dotenv(ROOT / "config" / ".env")
load_dotenv(ROOT / ".env")


def _parse_limits(raw: str) -> list[tuple[int, int]]:
    """'20:1,100:120' -> [(20, 1), (100, 120)]  (count per seconds)."""
    out: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        count, _, span = chunk.partition(":")
        out.append((int(count), int(span)))
    return out


def _parse_patch(raw: str) -> tuple[int, int]:
    """Empty or unparseable means no patch floor; LOOKBACK_DAYS does the filtering."""
    raw = raw.strip()
    if not raw:
        return (0, 0)
    major, _, minor = raw.partition(".")
    try:
        return int(major), int(minor or 0)
    except ValueError:
        return (0, 0)


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("RIOT_API_KEY", ""))
    platform: str = field(default_factory=lambda: os.getenv("RIOT_PLATFORM", "euw1"))
    region: str = field(default_factory=lambda: os.getenv("RIOT_REGION", "europe"))
    queue_id: int = field(default_factory=lambda: int(os.getenv("QUEUE_ID", "420")))
    min_patch: tuple[int, int] = field(
        default_factory=lambda: _parse_patch(os.getenv("MIN_PATCH", ""))
    )
    lookback_days: int = field(default_factory=lambda: int(os.getenv("LOOKBACK_DAYS", "21")))
    app_limits: list[tuple[int, int]] = field(
        default_factory=lambda: _parse_limits(os.getenv("APP_RATE_LIMITS", "20:1,100:120"))
    )
    data_dir: Path = field(default_factory=lambda: ROOT / os.getenv("DATA_DIR", "data"))
    db_path: Path = field(default_factory=lambda: ROOT / os.getenv("DB_PATH", "data/lol.duckdb"))
    model_dir: Path = field(default_factory=lambda: ROOT / os.getenv("MODEL_DIR", "models"))

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "RIOT_API_KEY is not set. Copy config/.env.example to config/.env and fill it in."
            )
        return self.api_key

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
