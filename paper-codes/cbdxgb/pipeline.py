"""End-to-end data preparation, model fitting and prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .data import load_all_wells, load_core_porosity
from .evaluation import direction_rates, regression_metrics
from .features import prepare_well, risky_indices
from .learners import make_learner
from .physics import (
    PorosityModel,
    add_physics_features,
    build_vm_training_set,
    calibrate_dtc_ma,
)


@dataclass
class Dataset:
    wells: dict[str, pd.DataFrame]
    core: pd.DataFrame
    feature_columns: list[str]
    vm: PorosityModel | None
    dtc_ma: float
    train: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: pd.DataFrame = field(default_factory=pd.DataFrame)
    blind: pd.DataFrame = field(default_factory=pd.DataFrame)
    enlarged_train_wells: pd.DataFrame = field(default_factory=pd.DataFrame)
    enlarged_blind: pd.DataFrame = field(default_factory=pd.DataFrame)


def prepare_dataset(
    data_dir: str | Path | None = None,
    use_vm: bool = True,
    seed: int = cfg.SEED,
) -> Dataset:
    """Load all wells, fit the volume model, and build train/evaluation splits."""
    data_dir = Path(data_dir) if data_dir is not None else cfg.DATA_DIR
    raw = load_all_wells(data_dir)
    core = load_core_porosity(data_dir)

    prepared: dict[str, pd.DataFrame] = {}
    feature_columns: list[str] | None = None
    for well in cfg.WELLS:
        table, cols = prepare_well(raw[well], well)
        prepared[well] = table
        feature_columns = cols
    if feature_columns is None:
        raise RuntimeError("no well features were constructed")

    vm = None
    dtc_ma = cfg.PHYS["dtc_ma_default"]
    if use_vm:
        vm_set = build_vm_training_set(raw, core)
        vm = PorosityModel().fit(
            vm_set["GR"].to_numpy(),
            vm_set["LG_RD"].to_numpy(),
            vm_set["PHI_CORE"].to_numpy(),
        )
        normal_train = pd.concat(
            [prepared[w].loc[prepared[w]["LABEL"] == "normal"] for w in cfg.TRAIN_WELLS],
            ignore_index=True,
        )
        dtc_ma = calibrate_dtc_ma(
            normal_train["ZDEN"].to_numpy(), normal_train["DTC"].to_numpy()
        )
        for well in cfg.WELLS:
            prepared[well], _ = add_physics_features(
                prepared[well], vm, dtc_ma, feature_columns
            )

    dataset = Dataset(
        wells=prepared,
        core=core,
        feature_columns=feature_columns,
        vm=vm,
        dtc_ma=dtc_ma,
    )

    train_pool = pd.concat(
        [prepared[w].loc[prepared[w]["LABEL"] == "normal"] for w in cfg.TRAIN_WELLS],
        ignore_index=True,
    )
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train_pool))
    n_train = int(cfg.TRAIN_FRACTION * len(order))
    dataset.train = train_pool.iloc[order[:n_train]].reset_index(drop=True)
    dataset.validation = train_pool.iloc[order[n_train:]].reset_index(drop=True)
    dataset.blind = prepared[cfg.BLIND_WELL].loc[
        prepared[cfg.BLIND_WELL]["LABEL"] == "normal"
    ].reset_index(drop=True)
    dataset.enlarged_train_wells = pd.concat(
        [prepared[w].loc[prepared[w]["LABEL"] == "enlarged"] for w in cfg.TRAIN_WELLS],
        ignore_index=True,
    )
    dataset.enlarged_blind = prepared[cfg.BLIND_WELL].loc[
        prepared[cfg.BLIND_WELL]["LABEL"] == "enlarged"
    ].reset_index(drop=True)
    return dataset


class CorrectionModel:
    """CBD-PIF-XGBoost style model with optional gating and physics features."""

    def __init__(
        self,
        learner: str = "xgb",
        target: str = "ZDEN",
        use_vm: bool = True,
        use_gate: bool = True,
        seed: int = cfg.SEED,
        **learner_kwargs,
    ):
        if target not in {"ZDEN", "DTC"}:
            raise ValueError("target must be 'ZDEN' or 'DTC'")
        self.learner_name = learner
        self.target = target
        self.use_vm = use_vm
        self.use_gate = use_gate
        self.seed = seed
        self.learner_kwargs = learner_kwargs
        self.model = None
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] = []
        self.risky_idx: np.ndarray = np.array([], dtype=int)
        self.fill_values: np.ndarray | None = None

    def _columns(self, dataset: Dataset) -> list[str]:
        cols = list(dataset.feature_columns)
        if self.use_vm:
            cols.append("ZDEN_VM" if self.target == "ZDEN" else "DTC_VM")
        return cols

    def fit(self, dataset: Dataset) -> "CorrectionModel":
        self.feature_columns = self._columns(dataset)
        self.risky_idx = risky_indices(self.feature_columns)
        x = dataset.train[self.feature_columns].to_numpy(dtype=float)
        y = dataset.train[self.target].to_numpy(dtype=float)
        self.fill_values = np.nanmean(x, axis=0)
        self.scaler = StandardScaler().fit(x)
        self.model = make_learner(self.learner_name, **self.learner_kwargs)
        self.model.fit(self.scaler.transform(x), y)
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        apply_gate: bool | None = None,
        multiplier: float = 1.0,
        thresholds: dict[str, float] | None = None,
        threshold_override: float | None = None,
    ) -> np.ndarray:
        """Predict one target.

        ``multiplier`` scales the calibrated per-well thresholds. A single
        ``threshold_override`` (in inches) replaces all thresholds, which is
        used for the blind-well scan.
        """
        if self.model is None or self.scaler is None or self.fill_values is None:
            raise RuntimeError("model is not fitted")
        apply_gate = self.use_gate if apply_gate is None else apply_gate
        x = frame[self.feature_columns].to_numpy(dtype=float).copy()

        if apply_gate:
            if threshold_override is not None:
                gate = frame["CBD"].to_numpy() > threshold_override
            else:
                if thresholds is None:
                    limits = frame["WELL"].map(
                        lambda w: cfg.CBD_THRESHOLDS[w] * multiplier
                    ).to_numpy()
                else:
                    limits = frame["WELL"].map(thresholds).to_numpy() * multiplier
                gate = frame["CBD"].to_numpy() > limits
            if gate.any():
                x[np.ix_(gate, self.risky_idx)] = self.fill_values[self.risky_idx]
        return self.model.predict(self.scaler.transform(x))


def evaluate_target(
    dataset: Dataset,
    model: CorrectionModel,
    frame: pd.DataFrame,
    split: str = "normal",
) -> dict[str, float]:
    """Evaluate a fitted model on a normal or enlarged reference set."""
    prediction = model.predict(frame)
    if split == "normal":
        return regression_metrics(frame[model.target].to_numpy(), prediction)

    raw = frame[model.target].to_numpy()
    rates = direction_rates(raw, prediction)
    normal_mean = {
        well: float(dataset.wells[well].loc[
            dataset.wells[well]["LABEL"] == "normal", model.target
        ].mean())
        for well in cfg.WELLS
    }
    reference = frame["WELL"].map(normal_mean).to_numpy(dtype=float)
    return {
        "n": rates["n"],
        "up_pct": rates["up_pct"],
        "down_pct": rates["down_pct"],
        "mean_pred": float(np.mean(prediction)),
        "mean_raw": float(np.mean(raw)),
        "reference_mean": float(np.mean(reference)),
        "mean_abs_dev_from_reference": float(np.mean(np.abs(prediction - reference))),
        "coverage": float(
            np.mean(
                frame["CBD"].to_numpy()
                > frame["WELL"].map(cfg.CBD_THRESHOLDS).to_numpy()
            )
        ),
    }


def predict_with_cbd_bins(
    dataset: Dataset,
    model: CorrectionModel,
    frame: pd.DataFrame,
    bins: list[tuple[float, float]],
) -> pd.DataFrame:
    """Mean raw and predicted values inside CBD bins."""
    prediction = model.predict(frame)
    work = frame.copy()
    work["_prediction"] = prediction
    rows = []
    for lo, hi in bins:
        part = work[(work["CBD"] >= lo) & (work["CBD"] < hi)]
        if part.empty:
            continue
        rows.append({
            "CBD bin (in)": f"[{lo:g}, {hi:g})",
            "n": int(len(part)),
            f"raw {model.target}": float(part[model.target].mean()),
            f"predicted {model.target}": float(part["_prediction"].mean()),
        })
    return pd.DataFrame(rows)
