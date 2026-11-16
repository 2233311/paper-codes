"""Generate synthetic demonstration wells with the same schema as field data.

The generator produces four wells with a shared stratigraphic column, smooth
petrophysical trends, borehole-enlargement intervals, and enlargement-induced
contamination of CNCF, resistivity, DEN and DTC. Both resistivities decrease
in common mode, with a slightly stronger decrease in the shallow curve, so
RD/RS stays near its non-enlarged level. The data are entirely synthetic and
contain no field measurements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .physics import rhg_dtc

WELL_SPECS = {
    "Well1": {
        "bit": 12.25, "threshold": 1.25, "phase": 0.3,
        "enlarged": [(3180, 3215, 2.8), (3300, 3325, 2.0)],
    },
    "Well2": {
        "bit": 8.50, "threshold": 1.50, "phase": 1.7,
        "enlarged": [(3070, 3105, 3.2), (3260, 3280, 2.2)],
    },
    "Well3": {
        "bit": 8.50, "threshold": 1.00, "phase": 3.1,
        "enlarged": [(3140, 3155, 1.8)],
    },
    "Well4": {
        "bit": 12.25, "threshold": 0.50, "phase": 4.4,
        "enlarged": [(3095, 3130, 3.5), (3200, 3235, 2.2), (3340, 3355, 1.0)],
    },
}


def _smooth_noise(rng: np.random.RandomState, n: int, scale: float,
                  kernel: int = 81) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, n)
    window = np.ones(kernel) / kernel
    smoothed = np.convolve(raw, window, mode="same")
    std = smoothed.std()
    return smoothed / std * scale if std > 0 else smoothed


def _cbd_profile(depth: np.ndarray, intervals: list[tuple[float, float, float]],
                 rng: np.random.RandomState) -> np.ndarray:
    cbd = rng.normal(0.0, 0.015, len(depth))
    for start, end, peak in intervals:
        centre = 0.5 * (start + end)
        width = max(1e-6, 0.5 * (end - start))
        cbd += peak * np.exp(-((depth - centre) / (0.45 * width)) ** 2)
    return np.clip(cbd, 0.0, 8.0)


def generate_well(well: str, rng: np.random.RandomState,
                  start: float = 3000.0, stop: float = 3400.0,
                  step: float = 0.1) -> pd.DataFrame:
    spec = WELL_SPECS[well]
    depth = np.arange(start, stop + 0.5 * step, step)
    n = len(depth)
    phase = spec["phase"]

    vsh = (
        0.36
        + 0.20 * np.sin(depth / 23.0 + phase)
        + 0.12 * np.sin(depth / 6.1 + 2 * phase)
        + _smooth_noise(rng, n, 0.06)
    )
    vsh = np.clip(vsh, 0.03, 0.92)
    gr = np.clip(30.0 + 120.0 * vsh + rng.normal(0, 4.0, n), 5, 195)

    phi = (
        0.20
        - 0.14 * vsh
        + 0.020 * np.sin(depth / 8.7 + phase)
        + _smooth_noise(rng, n, 0.010)
    )
    phi = np.clip(phi, 0.02, np.minimum(0.24, 0.95 - vsh))

    zden = (
        phi * cfg.PHYS["rho_f"]
        + vsh * cfg.PHYS["rho_sh"]
        + (1.0 - phi - vsh) * cfg.PHYS["rho_ma"]
        + rng.normal(0, 0.012, n)
    )
    dtc = rhg_dtc(phi, cfg.PHYS["dtc_ma_default"]) + rng.normal(0, 1.0, n)
    cncf = 90.0 * phi + 7.0 * vsh + 1.5 + rng.normal(0, 0.7, n)

    log_rd = 1.15 - 5.5 * phi + 0.35 * vsh + rng.normal(0, 0.035, n)
    ratio = np.clip(1.15 + 0.08 * np.sin(depth / 11.0 + phase)
                    + rng.normal(0, 0.02, n), 0.85, 2.0)

    cbd = _cbd_profile(depth, spec["enlarged"], rng)
    bit = np.full(n, spec["bit"])
    cal = bit + cbd

    # Enlargement contaminates the sensitive inputs and both targets. The two
    # resistivities share a common-mode decrease, so their ratio stays close to
    # its non-enlarged level, with a small extra decrease in the shallow curve.
    log_common = -0.020 * cbd
    rd = np.power(10.0, log_rd + log_common)
    rs = np.power(10.0, log_rd - np.log10(ratio) + log_common - 0.010 * cbd)
    cncf = cncf + 2.2 * cbd + 0.35 * cbd ** 2
    zden = zden - 0.055 * cbd
    dtc = dtc + 1.40 * cbd

    return pd.DataFrame({
        "DEPTH": np.round(depth, 3),
        "GR": np.round(gr, 3),
        "CNCF": np.round(cncf, 3),
        "RD": np.round(rd, 6),
        "RS": np.round(rs, 6),
        "BIT": bit,
        "CAL": np.round(cal, 3),
        "ZDEN": np.round(zden, 4),
        "DTC": np.round(dtc, 3),
    })


def generate_core_porosity(wells: dict[str, pd.DataFrame],
                           rng: np.random.RandomState) -> pd.DataFrame:
    rows = []
    for well, frame in wells.items():
        threshold = WELL_SPECS[well]["threshold"]
        normal = frame.loc[frame["CAL"] - frame["BIT"] <= threshold].reset_index(drop=True)
        points = np.arange(40, len(normal) - 40, 47)
        for idx in points:
            row = normal.iloc[idx]
            vsh = np.clip((row["GR"] - 30.0) / 120.0, 0.0, 1.0)
            phi = 20.0 - 14.0 * vsh + rng.normal(0, 1.1)
            rows.append({
                "WELL": well,
                "DEPTH": round(float(row["DEPTH"]), 3),
                "PHI_CORE": round(float(np.clip(phi, 1.0, 28.0)), 3),
            })
    return pd.DataFrame(rows).sort_values(["WELL", "DEPTH"]).reset_index(drop=True)


def generate_dataset(output_dir: str | Path | None = None,
                     seed: int = cfg.SEED) -> Path:
    """Write four well CSVs and one core-porosity CSV. Returns the directory."""
    output_dir = Path(output_dir) if output_dir is not None else cfg.DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    wells = {}
    for well in cfg.WELLS:
        frame = generate_well(well, rng)
        wells[well] = frame
        frame.to_csv(output_dir / f"{well.lower()}.csv", index=False)

    core = generate_core_porosity(wells, rng)
    core.to_csv(output_dir / "core_porosity.csv", index=False)
    return output_dir
