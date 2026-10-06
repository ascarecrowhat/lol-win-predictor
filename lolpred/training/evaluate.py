"""Metrics, calibration and model comparison.

Accuracy is the wrong headline for this problem: matchmaking balances teams, so a
good model is barely above a coin flip on accuracy while still being useful if its
probabilities are honest. Log loss and the calibration curve are what matter.

Model differences here are small (0.003-0.01 log loss), so comparisons use a
*paired* bootstrap on the same test rows. Comparing two unpaired means at this
effect size would need an impractically large test set.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-15


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, EPS, 1 - EPS)
    out = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
    }
    # AUC is undefined if the test slice happens to be single-class
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def constant_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Predict the training base rate for every match. Everything must beat this."""
    rate = float(np.mean(y_train))
    return metrics(y_test, np.full(len(y_test), rate))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Does a '70%' prediction win about 70% of the time?"""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
                "n": int(mask.sum()),
                "predicted": float(p[mask].mean()),
                "actual": float(y[mask].mean()),
                "gap": float(y[mask].mean() - p[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    rows = calibration_table(y, p, bins)
    n = len(y)
    return float(sum(r["n"] / n * abs(r["gap"]) for r in rows))


def paired_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Log-loss advantage of A over B, resampling the same rows for both models.

    Positive `mean_gain` means A has the lower (better) log loss. If the interval
    spans zero the two models are indistinguishable on this much test data.
    """
    rng = np.random.default_rng(seed)
    a = np.clip(p_a, EPS, 1 - EPS)
    b = np.clip(p_b, EPS, 1 - EPS)
    loss_a = -(y * np.log(a) + (1 - y) * np.log(1 - a))
    loss_b = -(y * np.log(b) + (1 - y) * np.log(1 - b))
    per_row = loss_b - loss_a  # positive where A is better
    n = len(y)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = per_row[rng.integers(0, n, n)].mean()
    return {
        "mean_gain": float(per_row.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_a_better": float((draws > 0).mean()),
    }


def format_report(name: str, m: dict[str, float], baseline: dict[str, float] | None = None) -> str:
    line = (
        f"{name:<22} log_loss {m['log_loss']:.5f}  brier {m['brier']:.5f}  "
        f"acc {m['accuracy']:.4f}  auc {m['auc']:.4f}"
    )
    if baseline:
        line += f"   (vs baseline {baseline['log_loss'] - m['log_loss']:+.5f})"
    return line
