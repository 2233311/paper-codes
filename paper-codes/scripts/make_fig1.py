#!/usr/bin/env python
"""Reproduce the CBD response analysis behind Fig. 1.

The three training wells are pooled, points with CBD > 5 in are removed, and
the five base inputs are binned by CBD. Each panel shows the pooled bin mean
and standard deviation, the per-well bin means, and the mean +/- 1 sigma of
the points at or below the per-well threshold (the non-enlarged reference).
Bins with fewer than 15 pooled points, or fewer than 10 points in one well,
are omitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbdxgb import config as cfg  # noqa: E402
from cbdxgb.data import load_well  # noqa: E402

FEATURES = ["GR", "CNCF", "LG_RD", "LG_RS", "R_ratio"]
LABELS = {
    "GR": "GR (API)",
    "CNCF": "CNCF (%)",
    "LG_RD": "log10 RD (ohm.m)",
    "LG_RS": "log10 RS (ohm.m)",
    "R_ratio": "RD / RS",
}


def bin_edges(max_cbd: float = 5.0) -> np.ndarray:
    edges = [0.0]
    while edges[-1] < 1.5:
        edges.append(round(edges[-1] + 0.25, 2))
    while edges[-1] < max_cbd:
        edges.append(round(edges[-1] + 0.5, 2))
    return np.asarray(edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--out-dir", type=Path,
                        default=cfg.RESULTS_DIR / "figures")
    parser.add_argument("--min-count", type=int, default=15)
    parser.add_argument("--min-count-per-well", type=int, default=10)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wells = [load_well(well, args.data_dir) for well in cfg.TRAIN_WELLS]
    pooled = pd.concat(wells, ignore_index=True)
    pooled = pooled[pooled["CBD"] <= 5.0]
    edges = bin_edges(5.0)

    normal_mean, normal_std = {}, {}
    for feature in FEATURES:
        normal = np.concatenate([
            well.loc[well["CBD"] <= cfg.CBD_THRESHOLDS[well["WELL"].iloc[0]],
                     feature].to_numpy()
            for well in wells
        ])
        normal_mean[feature] = float(np.mean(normal))
        normal_std[feature] = float(np.std(normal))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True)
    for ax, feature in zip(axes, FEATURES):
        group = pooled.groupby(pd.cut(pooled["CBD"], edges), observed=True)
        stats = group[feature].agg(["mean", "std", "count"]).reset_index()
        stats = stats[stats["count"] >= args.min_count]
        positions = np.asarray([
            interval.mid for interval in stats["CBD"]
        ])
        ax.errorbar(positions, stats["mean"], yerr=stats["std"], fmt="o-",
                    color="#1f4e9a", capsize=3, label="Pooled mean")
        for well in wells:
            sub = well[well["CBD"] <= 5.0]
            per_well = sub.groupby(pd.cut(sub["CBD"], edges),
                                   observed=True)[feature].agg(["mean", "count"])
            per_well = per_well[per_well["count"] >= args.min_count_per_well]
            ax.plot([i.mid for i in per_well.index], per_well["mean"].to_numpy(),
                    "--", linewidth=0.8, alpha=0.55,
                    label=well["WELL"].iloc[0])
        ax.axhspan(normal_mean[feature] - normal_std[feature],
                   normal_mean[feature] + normal_std[feature],
                   color="0.85", alpha=0.6, label="Non-enlarged mean +/- 1 s.d.")
        ax.set_title(LABELS[feature], fontsize=10)
        ax.set_xlabel("CBD (in)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Binned response of the five base inputs to CBD "
                 "(Well 1-Well 3, synthetic demonstration data)", fontsize=11)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"fig1_cbd_response.{suffix}", dpi=300)
    plt.close(fig)
    print(f"Figure written to: {args.out_dir}")


if __name__ == "__main__":
    main()
