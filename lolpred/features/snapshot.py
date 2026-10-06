"""Persist the corpus-wide parts of the feature state for use at serve time.

Per-player state is *not* saved: there are hundreds of thousands of players and
their history is rebuilt on demand from the ten players in the game being
predicted. What must be carried over is everything estimated from the corpus as a
whole and impossible to reconstruct from ten players:

* champion/role win rates per patch
* the per-role trait averages used to residualise playstyle
* champion duration curves (the scaling score)
* champion role frequencies, for role inference

Saving these keeps training and serving on identical definitions, which is the
only way the numbers measured offline mean anything in production.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .advanced import AdvancedState
from .roles import RolePrior
from .state import FeatureState

SNAPSHOT_VERSION = 1


def export_globals(
    state: FeatureState, adv: AdvancedState, roles: RolePrior, meta: dict | None = None
) -> dict[str, Any]:
    return {
        "version": SNAPSHOT_VERSION,
        "champ_role": dict(state.champ_role),
        "role_sums": dict(adv.role_sums),
        "role_counts": dict(adv.role_counts),
        "champ_duration": {k: list(v) for k, v in adv.champ_duration.items()},
        "role_champ": {k: v.tolist() for k, v in roles.champ.items()},
        "meta": meta or {},
    }


def save(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path, compress=3)
    return path


def load(path: str | Path) -> tuple[FeatureState, AdvancedState, RolePrior, dict]:
    """Rebuild empty states pre-loaded with the corpus-wide aggregates."""
    import numpy as np

    payload = joblib.load(path)
    if payload.get("version") != SNAPSHOT_VERSION:
        raise ValueError(
            f"state snapshot version {payload.get('version')} != {SNAPSHOT_VERSION}; rebuild it"
        )

    state = FeatureState()
    state.champ_role.update(payload["champ_role"])

    adv = AdvancedState()
    adv.role_sums.update(payload["role_sums"])
    adv.role_counts.update(payload["role_counts"])
    for key, value in payload["champ_duration"].items():
        adv.champ_duration[key] = list(value)

    roles = RolePrior()
    for key, value in payload["role_champ"].items():
        roles.champ[key] = np.asarray(value, dtype=float)

    return state, adv, roles, payload.get("meta", {})
