"""Physics-inspired feature construction (volume model and RHG)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def vsh_from_gr(gr: np.ndarray) -> np.ndarray:
    """Linear gamma-ray shale-volume index, clipped to [0, 1]."""
    gr = np.asarray(gr, dtype=float)
    return np.clip(
        (gr - cfg.PHYS["gr_clean"]) / (cfg.PHYS["gr_shale"] - cfg.PHYS["gr_clean"]),
        0.0,
        1.0,
    )


def rhg_dtc(phi_fraction: np.ndarray, dtc_ma: float) -> np.ndarray:
    """Raymer-Hunt-Gardner forward model: porosity -> DTC (us/ft)."""
    phi = np.clip(np.asarray(phi_fraction, dtype=float), 0.0, 0.40)
    v_ma = 1.0 / dtc_ma
    v_f = 1.0 / cfg.PHYS["dtc_f"]
    slowness = (1.0 - phi) ** 2 * v_ma + phi * v_f
    return 1.0 / slowness


def forward_den(phi_pct: np.ndarray, vsh: np.ndarray) -> np.ndarray:
    """Three-component volume model: porosity + shale + matrix -> DEN."""
    phi = np.asarray(phi_pct, dtype=float) / 100.0
    vsh = np.asarray(vsh, dtype=float)
    v_matrix = np.clip(1.0 - phi - vsh, 0.0, 1.0)
    return (
        phi * cfg.PHYS["rho_f"]
        + vsh * cfg.PHYS["rho_sh"]
        + v_matrix * cfg.PHYS["rho_ma"]
    )


def calibrate_dtc_ma(zden: np.ndarray, dtc: np.ndarray) -> float:
    """Fit a single matrix transit time from RHG on non-enlarged samples."""
    mask = np.isfinite(zden) & np.isfinite(dtc)
    z = np.asarray(zden)[mask].astype(float)
    d = np.asarray(dtc)[mask].astype(float)
    if len(z) < 50:
        return cfg.PHYS["dtc_ma_default"]
    phi_d = (cfg.PHYS["rho_ma"] - z) / (cfg.PHYS["rho_ma"] - cfg.PHYS["rho_f"])
    phi_d = np.clip(phi_d, 0.001, 0.40)
    v_bulk = 1.0 / d
    v_fluid = 1.0 / cfg.PHYS["dtc_f"]
    v_matrix = (v_bulk - phi_d * v_fluid) / (1.0 - phi_d) ** 2
    ok = np.isfinite(v_matrix) & (v_matrix > 0)
    if ok.sum() < 20:
        return cfg.PHYS["dtc_ma_default"]
    return float(1.0 / np.median(v_matrix[ok]))


@dataclass
class PorosityModel:
    """Ridge regression from enlargement-immune logs to core porosity."""

    model: RidgeCV | None = None
    scaler: StandardScaler | None = None

    def fit(self, gr: np.ndarray, lg_rd: np.ndarray, phi_pct: np.ndarray) -> "PorosityModel":
        x = np.column_stack([gr, lg_rd])
        self.scaler = StandardScaler().fit(x)
        self.model = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(
            self.scaler.transform(x), phi_pct
        )
        return self

    def predict(self, gr: np.ndarray, lg_rd: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("PorosityModel is not fitted")
        x = np.column_stack([gr, lg_rd])
        return self.model.predict(self.scaler.transform(x))


def build_vm_training_set(
    wells: dict[str, pd.DataFrame],
    core: pd.DataFrame,
    training_wells: list[str] | None = None,
) -> pd.DataFrame:
    """Match core porosity to the nearest log sample in the training wells."""
    training_wells = training_wells or cfg.TRAIN_WELLS
    rows = []
    for well in training_wells:
        log = wells[well][["DEPTH", "GR", "LG_RD", "CBD"]].sort_values("DEPTH")
        cores = core[core["WELL"] == well].sort_values("DEPTH")
        if cores.empty:
            continue
        matched = pd.merge_asof(
            cores[["DEPTH", "PHI_CORE"]],
            log,
            on="DEPTH",
            direction="nearest",
            tolerance=0.05,
        )
        matched = matched[matched["CBD"] <= cfg.CBD_THRESHOLDS[well]]
        matched["WELL"] = well
        rows.append(matched)
    if not rows:
        raise ValueError("no core porosity samples matched the log data")
    return pd.concat(rows, ignore_index=True).dropna(subset=["GR", "LG_RD", "PHI_CORE"])


def add_physics_features(
    df: pd.DataFrame,
    vm: PorosityModel,
    dtc_ma: float,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Attach VM porosity, ZDEN_VM and DTC_VM to a window-feature table."""
    out = df.copy()
    out["VM_PHI"] = np.clip(
        vm.predict(out["GR"].to_numpy(), out["LG_RD"].to_numpy()), 0.0, 35.0
    )
    vsh = vsh_from_gr(out["GR"].to_numpy())
    out["ZDEN_VM"] = forward_den(out["VM_PHI"].to_numpy(), vsh)
    out["DTC_VM"] = rhg_dtc(out["VM_PHI"].to_numpy() / 100.0, dtc_ma)
    out["DTC_MA"] = dtc_ma
    return out, feature_columns + ["ZDEN_VM", "DTC_VM"]
