"""Borehole labels and depth-window feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def enlarged_labels(
    df: pd.DataFrame,
    threshold: float,
    expand_half: int = cfg.EXPAND_HALF,
    max_gap_m: float = cfg.MAX_GAP_M,
) -> np.ndarray:
    """Return a boolean mask of enlarged samples.

    A sample is labelled enlarged when ``CBD > threshold``. The label is then
    expanded by ``expand_half`` samples on each side, but expansion stops when
    the depth gap between consecutive samples exceeds ``max_gap_m``. This
    mirrors the window expansion used in the paper.
    """
    cbd = df["CBD"].to_numpy()
    depth = df["DEPTH"].to_numpy()
    n = len(df)
    labels = cbd > threshold

    for i in np.flatnonzero(labels):
        k = i - 1
        steps = 0
        while k >= 0 and steps < expand_half and abs(depth[k + 1] - depth[k]) <= max_gap_m:
            labels[k] = True
            k -= 1
            steps += 1
        k = i + 1
        steps = 0
        while k < n and steps < expand_half and abs(depth[k] - depth[k - 1]) <= max_gap_m:
            labels[k] = True
            k += 1
            steps += 1
    return labels


def label_well(df: pd.DataFrame, well: str) -> pd.DataFrame:
    """Add ``LABEL`` (normal/enlarged) and ``GATE`` (CBD above threshold)."""
    df = df.copy()
    threshold = cfg.CBD_THRESHOLDS[well]
    df["GATE"] = (df["CBD"] > threshold).astype(int)
    df["LABEL"] = np.where(
        enlarged_labels(df, threshold), "enlarged", "normal"
    )
    return df


def build_window_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build 5-point depth-window features for the five base inputs.

    For each base feature at depth z, values from z-2 to z+2 are included.
    A component is set to NaN when its source sample lies across a depth gap
    (spacing larger than 1.5 times the median step per sample offset), so a
    window never mixes values from opposite sides of a gap. Rows with
    incomplete windows are dropped.
    """
    df = df.sort_values("DEPTH").reset_index(drop=True).copy()
    depth = df["DEPTH"].to_numpy(dtype=float)
    step = float(np.median(np.diff(depth))) if len(depth) > 1 else 0.0
    window_cols: list[str] = []
    for feat in cfg.BASE_FEATURES:
        if feat not in df.columns:
            raise ValueError(f"missing base feature: {feat}")
        for offset in range(-cfg.WINDOW_HALF, cfg.WINDOW_HALF + 1):
            suffix = f"_{offset:+d}" if offset != 0 else ""
            col = f"{feat}{suffix}"
            if offset == 0:
                values = df[feat]
            else:
                values = df[feat].shift(-offset)
                span = (df["DEPTH"] - df["DEPTH"].shift(-offset)).abs()
                values = values.where(span <= 1.5 * abs(offset) * step)
            df[col] = values
            window_cols.append(col)
    df = df.dropna(subset=window_cols).reset_index(drop=True)
    return df, window_cols


def risky_indices(feature_columns: list[str]) -> np.ndarray:
    """Indices of window columns that are contaminated by enlargement."""
    return np.asarray(
        [i for i, c in enumerate(feature_columns)
         if c.startswith(cfg.RISKY_PREFIXES)],
        dtype=int,
    )


def prepare_well(df: pd.DataFrame, well: str) -> tuple[pd.DataFrame, list[str]]:
    """Label a well, build its 25-dimensional window features, and return both."""
    labelled = label_well(df, well)
    with_windows, cols = build_window_features(labelled)
    return with_windows, cols
