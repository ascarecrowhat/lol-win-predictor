"""Train the baseline ladder and the gradient-boosted model, then calibrate.

Order matters: the constant baseline sets the bar, logistic regression says whether
the features carry signal at all, and LightGBM only earns its place by beating
logistic on a paired comparison. A fancier model that cannot beat logistic is
telling you the features are the problem.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import Split
from .evaluate import (
    constant_baseline,
    expected_calibration_error,
    calibration_table,
    format_report,
    metrics,
    paired_bootstrap,
)

log = logging.getLogger(__name__)


def build_logistic() -> Pipeline:
    """Impute then scale: linear models need both, LightGBM needs neither."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def build_lgbm(n_rows: int):
    import lightgbm as lgb

    # Small, heavily regularised: the signal is weak, so an unconstrained model
    # memorises noise and loses to logistic regression.
    return lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=max(40, n_rows // 200),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5.0,
        verbose=-1,
    )


class Identity:
    """Pass-through calibrator, used when calibration does not help."""

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)


def fit_calibrator(p_val: np.ndarray, y_val: np.ndarray):
    """Fit isotonic regression, but keep it only if it beats doing nothing.

    A model whose probabilities are already honest gains nothing from isotonic
    regression, and on a small validation slice isotonic will happily overfit and
    make calibration worse. Checked by cross-validation on the validation set so
    the decision is not made on the data the calibrator was fitted to.
    """
    from sklearn.model_selection import KFold

    if len(p_val) < 200:
        log.warning("Validation set too small to calibrate safely - skipping")
        return Identity(), {"applied": False, "reason": "val_too_small"}

    raw_loss, cal_loss = [], []
    for train_idx, test_idx in KFold(n_splits=4, shuffle=True, random_state=0).split(p_val):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(p_val[train_idx], y_val[train_idx])
        held_p, held_y = p_val[test_idx], y_val[test_idx]
        raw_loss.append(log_loss(held_y, np.clip(held_p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        cal_loss.append(
            log_loss(held_y, np.clip(iso.predict(held_p), 1e-6, 1 - 1e-6), labels=[0, 1])
        )
    raw_mean, cal_mean = float(np.mean(raw_loss)), float(np.mean(cal_loss))
    detail = {"val_log_loss_raw": raw_mean, "val_log_loss_calibrated": cal_mean}
    if cal_mean < raw_mean:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(p_val, y_val)
        log.info("Isotonic calibration helps on validation (%.5f -> %.5f) - applying",
                 raw_mean, cal_mean)
        return iso, {"applied": True, **detail}
    log.info("Isotonic calibration does not help on validation (%.5f -> %.5f) - skipping",
             raw_mean, cal_mean)
    return Identity(), {"applied": False, **detail}


def train(split: Split, model_dir: Path, label: str = "lolpred") -> dict[str, Any]:
    results: dict[str, Any] = {"split": split.info}
    base = constant_baseline(split.y_train, split.y_test)
    print(format_report("constant baseline", base))
    results["baseline"] = base

    predictions: dict[str, np.ndarray] = {}

    # -- logistic ---------------------------------------------------------
    logistic = build_logistic()
    logistic.fit(split.X_train, split.y_train)
    p_log = logistic.predict_proba(split.X_test)[:, 1]
    predictions["logistic"] = p_log
    results["logistic"] = metrics(split.y_test, p_log)
    print(format_report("logistic", results["logistic"], base))

    # -- gradient boosting -------------------------------------------------
    best_model, best_name = logistic, "logistic"
    try:
        gbm = build_lgbm(len(split.X_train))
        gbm.fit(
            split.X_train, split.y_train,
            eval_set=[(split.X_val, split.y_val)],
            eval_metric="binary_logloss",
        )
        p_gbm = gbm.predict_proba(split.X_test)[:, 1]
        predictions["lightgbm"] = p_gbm
        results["lightgbm"] = metrics(split.y_test, p_gbm)
        print(format_report("lightgbm", results["lightgbm"], base))

        vs = paired_bootstrap(split.y_test, p_gbm, p_log)
        results["lightgbm_vs_logistic"] = vs
        verdict = (
            "lightgbm wins" if vs["ci_low"] > 0
            else "logistic wins" if vs["ci_high"] < 0
            else "indistinguishable on this much test data"
        )
        print(
            f"  paired: lightgbm - logistic = {vs['mean_gain']:+.5f} log loss "
            f"[{vs['ci_low']:+.5f}, {vs['ci_high']:+.5f}] -> {verdict}"
        )
        if vs["ci_low"] > 0:
            best_model, best_name = gbm, "lightgbm"

        importance = (
            pd.Series(gbm.feature_importances_, index=split.features)
            .sort_values(ascending=False)
        )
        results["top_features"] = importance.head(15).round(1).to_dict()
        print("\n  top features by gain:")
        for name, gain in importance.head(10).items():
            print(f"    {name:<28} {gain:.0f}")
    except ImportError:
        log.warning("lightgbm not installed - skipping the gradient boosted model")

    # -- calibration -------------------------------------------------------
    p_val = best_model.predict_proba(split.X_val)[:, 1]
    iso, cal_info = fit_calibrator(p_val, split.y_val)
    results["calibration"] = cal_info
    p_best = predictions[best_name]
    p_cal = iso.predict(p_best)
    results["calibrated"] = metrics(split.y_test, p_cal)
    suffix = "+ isotonic" if cal_info["applied"] else "(uncalibrated)"
    print(format_report(f"{best_name} {suffix}", results["calibrated"], base))

    results["ece_before"] = expected_calibration_error(split.y_test, p_best)
    results["ece_after"] = expected_calibration_error(split.y_test, p_cal)
    print(f"  calibration error: {results['ece_before']:.4f} -> {results['ece_after']:.4f}")
    results["calibration_table"] = calibration_table(split.y_test, p_cal)
    print("\n  calibration of the final model:")
    print(f"    {'bin':<12}{'n':>7}{'predicted':>11}{'actual':>9}{'gap':>8}")
    for row in results["calibration_table"]:
        print(f"    {row['bin']:<12}{row['n']:>7}{row['predicted']:>11.3f}"
              f"{row['actual']:>9.3f}{row['gap']:>+8.3f}")

    # -- persist -----------------------------------------------------------
    model_dir.mkdir(parents=True, exist_ok=True)
    artefact = {
        "model": best_model,
        "calibrator": iso,
        "features": split.features,
        "model_name": best_name,
    }
    path = model_dir / f"{label}.joblib"
    joblib.dump(artefact, path)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": best_name,
        "results": {k: v for k, v in results.items() if k != "calibration_table"},
        "calibration_table": results["calibration_table"],
    }
    (model_dir / f"{label}.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nsaved {path.name} and {label}.json ({best_name}, calibrated)")
    results["artefact"] = str(path)
    return results


def predict(artefact_path: Path, frame: pd.DataFrame) -> np.ndarray:
    """Load a saved model and return calibrated blue-win probabilities."""
    art = joblib.load(artefact_path)
    raw = art["model"].predict_proba(frame[art["features"]])[:, 1]
    return art["calibrator"].predict(raw)
