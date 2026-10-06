"""Dataset assembly: time-based splits, coverage filtering, side-swap augmentation.

Two rules are enforced here rather than left to the caller, because getting either
wrong silently inflates the scores:

1. Splits are by time, never random. A random split lets the model see a player's
   later games while predicting their earlier ones.
2. Augmentation happens *after* splitting and only on the training part. A match
   and its mirror image must never straddle the split, or the test set contains
   the training set with the sign flipped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features.build import augment_side_swap

log = logging.getLogger(__name__)

META_COLS = ["match_id", "game_start", "patch", "blue_win"]


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    features: list[str]
    info: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"train {len(self.X_train):,} | val {len(self.X_val):,} | "
            f"test {len(self.X_test):,} | {len(self.features)} features"
        )


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in META_COLS]


def filter_coverage(frame: pd.DataFrame, min_covered: int) -> pd.DataFrame:
    """Keep rows where both teams have at least `min_covered` players with history."""
    if min_covered <= 0:
        return frame
    keep = (frame.blue_coverage >= min_covered) & (frame.red_coverage >= min_covered)
    dropped = len(frame) - int(keep.sum())
    log.info(
        "Coverage filter >=%d per side: kept %d rows, dropped %d",
        min_covered, int(keep.sum()), dropped,
    )
    return frame.loc[keep].copy()


def make_split(
    frame: pd.DataFrame,
    min_covered: int = 0,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    augment: bool = True,
) -> Split:
    frame = frame.sort_values("game_start").reset_index(drop=True)
    frame = filter_coverage(frame, min_covered)
    if len(frame) < 50:
        raise ValueError(f"only {len(frame)} rows after filtering - not enough to train")

    n = len(frame)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_test - n_val
    if n_train < 20:
        raise ValueError(f"only {n_train} training rows after splitting")

    train = frame.iloc[:n_train]
    val = frame.iloc[n_train : n_train + n_val]
    test = frame.iloc[n_train + n_val :]

    boundaries = {
        "train_end": str(train.game_start.max()),
        "val_end": str(val.game_start.max()),
        "test_start": str(test.game_start.min()),
        "test_end": str(test.game_start.max()),
    }
    log.info(
        "Time split: train <= %s, val <= %s, test %s .. %s",
        boundaries["train_end"], boundaries["val_end"],
        boundaries["test_start"], boundaries["test_end"],
    )

    # Mirror only the training rows; val and test stay as they really happened so
    # the reported numbers describe real matches, not synthetic ones.
    train_aug = augment_side_swap(train) if augment else train
    features = feature_names(frame)

    return Split(
        X_train=train_aug[features],
        y_train=train_aug.blue_win.to_numpy(),
        X_val=val[features],
        y_val=val.blue_win.to_numpy(),
        X_test=test[features],
        y_test=test.blue_win.to_numpy(),
        features=features,
        info={
            "rows_total": n,
            "rows_train_raw": int(n_train),
            "rows_train_augmented": len(train_aug),
            "rows_val": len(val),
            "rows_test": len(test),
            "min_covered": min_covered,
            "augmented": augment,
            "blue_win_rate_train": float(train.blue_win.mean()),
            "blue_win_rate_test": float(test.blue_win.mean()),
            "patches": sorted(frame.patch.dropna().unique().tolist()),
            **boundaries,
        },
    )
