"""Metrics and grouped summaries for non-enlarged and enlarged intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "r2": np.nan, "mae": np.nan,
                "rmse": np.nan, "mape": np.nan, "bias": np.nan}
    mape = np.mean(np.abs((y_pred - y_true) / np.where(y_true == 0, np.nan, y_true))) * 100
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape),
        "bias": float(np.mean(y_pred - y_true)),
    }


def direction_rates(raw: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Fraction of samples moved upward / downward by the correction."""
    raw = np.asarray(raw, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(raw) & np.isfinite(predicted)
    raw, predicted = raw[mask], predicted[mask]
    if len(raw) == 0:
        return {"n": 0, "up_pct": np.nan, "down_pct": np.nan}
    return {
        "n": int(len(raw)),
        "up_pct": float(np.mean(predicted > raw) * 100),
        "down_pct": float(np.mean(predicted < raw) * 100),
    }


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a small DataFrame as a GitHub-flavoured Markdown table."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    # itertuples preserves the column dtypes; iterrows casts all-numeric rows
    # to float and would render integer counts as "8900.0000".
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if not np.isfinite(value) else floatfmt.format(value))
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
