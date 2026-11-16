"""Data loading and derivation of borehole-aware features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

REQUIRED_COLUMNS = [
    "DEPTH", "GR", "CNCF", "RD", "RS", "BIT", "CAL", "ZDEN", "DTC",
]


def load_well(well: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load one well from ``data/synthetic/<well>.csv``.

    The loader expects one row per depth sample sorted by depth. Column names
    are case-sensitive and listed in ``docs/data_dictionary.md``.
    """
    data_dir = Path(data_dir) if data_dir is not None else cfg.DATA_DIR
    path = data_dir / f"{well.lower()}.csv"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.sort_values("DEPTH").reset_index(drop=True)
    df["WELL"] = well
    return add_derived_columns(df)


def load_all_wells(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {well: load_well(well, data_dir) for well in cfg.WELLS}


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add CBD, log-resistivity and resistivity ratio columns."""
    df = df.copy()
    df["CBD"] = df["CAL"] - df["BIT"]
    df["LG_RD"] = np.log10(df["RD"].where(df["RD"] > 0))
    df["LG_RS"] = np.log10(df["RS"].where(df["RS"] > 0))
    df["R_ratio"] = df["RD"] / df["RS"].replace(0, np.nan)
    return df


def load_core_porosity(data_dir: Path | None = None) -> pd.DataFrame:
    """Load sparse core-plug porosity from ``core_porosity.csv``.

    Columns: WELL, DEPTH, PHI_CORE.
    """
    data_dir = Path(data_dir) if data_dir is not None else cfg.DATA_DIR
    path = data_dir / "core_porosity.csv"
    df = pd.read_csv(path)
    for col in ("WELL", "DEPTH", "PHI_CORE"):
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column: {col}")
    return df.sort_values(["WELL", "DEPTH"]).reset_index(drop=True)
