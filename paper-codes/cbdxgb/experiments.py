"""Experiment drivers used to generate the summary tables in ``results/``."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .evaluation import direction_rates, markdown_table, regression_metrics
from .learners import LEARNER_NAMES, make_learner
from .pipeline import (
    CorrectionModel,
    Dataset,
    evaluate_target,
    prepare_dataset,
    predict_with_cbd_bins,
)


def _write(out_dir: Path, name: str, title: str, sections: list[tuple[str, pd.DataFrame]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, table in sections:
        lines += [f"## {heading}", "", markdown_table(table), ""]
        safe_heading = "".join(
            ch if ch.isalnum() else "_" for ch in heading.lower()
        ).strip("_")
        csv_name = name.replace(".md", "") + "_" + safe_heading + ".csv"
        table.to_csv(out_dir / csv_name, index=False)
    (out_dir / name).write_text("\n".join(lines), encoding="utf-8")


def _normal_row(model_name: str, target: str, split: str,
                metrics: dict[str, float]) -> dict:
    return {
        "model": model_name,
        "target": target,
        "set": split,
        "n": metrics["n"],
        "R2": metrics["r2"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "MAPE (%)": metrics["mape"],
    }


def _enlarged_row(model_name: str, target: str, split: str,
                  metrics: dict[str, float]) -> dict:
    return {
        "model": model_name,
        "target": target,
        "set": split,
        "n": metrics["n"],
        "up (%)": metrics["up_pct"],
        "down (%)": metrics["down_pct"],
        "reference gate coverage": metrics["coverage"],
        "mean predicted": metrics["mean_pred"],
        "mean raw": metrics["mean_raw"],
        "reference mean": metrics["reference_mean"],
        "mean |dev|": metrics["mean_abs_dev_from_reference"],
    }


def run_main(data_dir: str | Path | None = None,
             out_dir: str | Path | None = None) -> pd.DataFrame:
    """CBD-PIF-XGBoost versus plain XGBoost, both targets."""
    dataset = prepare_dataset(data_dir=data_dir, use_vm=True)
    out_dir = Path(out_dir) if out_dir is not None else cfg.TABLE_DIR

    frameworks = {
        "CBD-PIF-XGBoost": CorrectionModel("xgb", use_vm=True, use_gate=True),
        "XGBoost (25D, no gate)": CorrectionModel("xgb", use_vm=False, use_gate=False),
    }
    normal_rows, enlarged_rows = [], []
    bin_frames = {}
    for name, model in frameworks.items():
        for target in ("ZDEN", "DTC"):
            model.target = target
            model.fit(dataset)
            for split_name, frame in (
                ("validation", dataset.validation),
                ("blind well (normal)", dataset.blind),
            ):
                metric = evaluate_target(dataset, model, frame, split="normal")
                normal_rows.append(_normal_row(name, target, split_name, metric))
            for split_name, frame in (
                ("training wells (enlarged)", dataset.enlarged_train_wells),
                ("blind well (enlarged)", dataset.enlarged_blind),
            ):
                metric = evaluate_target(dataset, model, frame, split="enlarged")
                enlarged_rows.append(_enlarged_row(name, target, split_name, metric))
            bins = predict_with_cbd_bins(
                dataset, model, dataset.enlarged_blind,
                [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 10.0)],
            )
            bins = bins.rename(columns={
                f"predicted {target}": f"{name} {target}"
            }).set_index(["CBD bin (in)", "n"])
            bin_frames[(name, target)] = bins

    normal_table = pd.DataFrame(normal_rows)
    enlarged_table = pd.DataFrame(enlarged_rows)

    sections = [
        ("Non-enlarged accuracy", normal_table),
        ("Enlarged direction and deviation", enlarged_table),
    ]
    for target in ("ZDEN", "DTC"):
        pieces = []
        for i, name in enumerate(frameworks):
            piece = bin_frames[(name, target)]
            if i > 0:
                piece = piece.drop(columns=[c for c in piece.columns
                                            if c.startswith("raw ")])
            pieces.append(piece)
        merged = pd.concat(pieces, axis=1).reset_index()
        sections.append((f"Blind-well CBD bins, {target}", merged))
    _write(out_dir, "main_results.md",
           "Main results: CBD-PIF-XGBoost versus plain XGBoost", sections)
    return normal_table


def run_model_comparison(data_dir: str | Path | None = None,
                         out_dir: str | Path | None = None,
                         learners: list[str] | None = None) -> pd.DataFrame:
    """Ten learners under the uniform 25-dimensional input protocol."""
    learners = learners or LEARNER_NAMES
    dataset = prepare_dataset(data_dir=data_dir, use_vm=False)
    out_dir = Path(out_dir) if out_dir is not None else cfg.TABLE_DIR
    rows = []
    for learner in learners:
        for target in ("ZDEN", "DTC"):
            try:
                model = CorrectionModel(learner, target=target, use_vm=False, use_gate=False)
                model.fit(dataset)
            except ImportError as exc:
                rows.append({
                    "model": learner, "target": target,
                    "set": f"skipped ({exc})", "n": 0,
                    "R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan,
                })
                break
            for split_name, frame in (
                ("validation", dataset.validation),
                ("blind well (normal)", dataset.blind),
            ):
                metric = evaluate_target(dataset, model, frame, split="normal")
                rows.append(_normal_row(learner, target, split_name, metric))
    table = pd.DataFrame(rows)
    _write(out_dir, "model_comparison.md",
           "Model comparison under a uniform 25-dimensional input", [("Results", table)])
    return table


def run_ablation(data_dir: str | Path | None = None,
                 out_dir: str | Path | None = None,
                 learners: list[str] | None = None) -> pd.DataFrame:
    """2 x 2 ablation: gating on/off against physics features on/off."""
    learners = learners or ["xgb", "rf"]
    dataset = prepare_dataset(data_dir=data_dir, use_vm=True)
    out_dir = Path(out_dir) if out_dir is not None else cfg.TABLE_DIR
    rows = []
    for learner in learners:
        for use_gate in (False, True):
            for use_vm in (False, True):
                label = (
                    f"{learner.upper()}, gate={'on' if use_gate else 'off'}, "
                    f"PIF={'on' if use_vm else 'off'}"
                )
                for target in ("ZDEN", "DTC"):
                    model = CorrectionModel(
                        learner, target=target, use_vm=use_vm, use_gate=use_gate
                    )
                    model.fit(dataset)
                    blind = evaluate_target(dataset, model, dataset.blind, "normal")
                    blind_enlarged = evaluate_target(
                        dataset, model, dataset.enlarged_blind, "enlarged"
                    )
                    tr_enlarged = evaluate_target(
                        dataset, model, dataset.enlarged_train_wells, "enlarged"
                    )
                    rows.append({
                        "model": label,
                        "target": target,
                        "blind R2": blind["r2"],
                        "blind MAE": blind["mae"],
                        "Well4 enlarged up (%)": blind_enlarged["up_pct"],
                        "Well4 enlarged down (%)": blind_enlarged["down_pct"],
                        "Well4 reference gate coverage": blind_enlarged["coverage"],
                        "training wells down (%)": tr_enlarged["down_pct"],
                    })
    table = pd.DataFrame(rows)
    _write(out_dir, "ablation.md",
           "Component ablation (gating x physics-inspired features)", [("Results", table)])
    return table


def _gated_fraction(frame: pd.DataFrame, thresholds: dict[str, float]) -> float:
    limits = frame["WELL"].map(thresholds).to_numpy(dtype=float)
    return float(np.mean(frame["CBD"].to_numpy() > limits))


@contextmanager
def _scaled_thresholds(multiplier: float):
    original = dict(cfg.CBD_THRESHOLDS)
    for well in cfg.CBD_THRESHOLDS:
        cfg.CBD_THRESHOLDS[well] = original[well] * multiplier
    try:
        yield
    finally:
        cfg.CBD_THRESHOLDS.clear()
        cfg.CBD_THRESHOLDS.update(original)


def run_threshold_sensitivity(data_dir: str | Path | None = None,
                              out_dir: str | Path | None = None,
                              closed_loop: bool = False) -> pd.DataFrame:
    """Open-loop multiplier scan, blind-well scan, and optional closed loop."""
    out_dir = Path(out_dir) if out_dir is not None else cfg.TABLE_DIR
    baseline = prepare_dataset(data_dir=data_dir, use_vm=True)
    baseline_thresholds = dict(cfg.CBD_THRESHOLDS)
    models = {}
    for target in ("ZDEN", "DTC"):
        models[target] = CorrectionModel("xgb", target=target,
                                         use_vm=True, use_gate=True).fit(baseline)

    open_rows = []
    for multiplier in cfg.MULTIPLIERS:
        row = {"multiplier m": multiplier}
        scaled = {w: baseline_thresholds[w] * multiplier for w in baseline_thresholds}
        for target in ("ZDEN", "DTC"):
            model = models[target]
            val = regression_metrics(
                baseline.validation[target].to_numpy(),
                model.predict(baseline.validation, multiplier=multiplier),
            )
            blind = regression_metrics(
                baseline.blind[target].to_numpy(),
                model.predict(baseline.blind, multiplier=multiplier),
            )
            blind_enl = evaluate_target_with_multiplier(
                baseline, model, baseline.enlarged_blind, multiplier, scaled
            )
            row[f"val {target} R2"] = val["r2"]
            row[f"blind {target} R2"] = blind["r2"]
            if target == "DTC":
                row["Well4 enlarged DTC down (%)"] = blind_enl["down_pct"]
            else:
                row["Well4 enlarged DEN up (%)"] = blind_enl["up_pct"]
            row["Well4 coverage"] = blind_enl["coverage"]
        row["false gate, validation (%)"] = _gated_fraction(baseline.validation, scaled) * 100
        row["false gate, blind normal (%)"] = _gated_fraction(baseline.blind, scaled) * 100
        open_rows.append(row)

    blind_rows = []
    for tau in cfg.BLIND_THRESHOLD_SCAN:
        row = {"threshold (in)": tau}
        for target in ("ZDEN", "DTC"):
            model = models[target]
            normal = regression_metrics(
                baseline.blind[target].to_numpy(),
                model.predict(baseline.blind, threshold_override=tau),
            )
            enlarged = evaluate_target_threshold(
                baseline, model, baseline.enlarged_blind, tau
            )
            row[f"blind normal {target} R2"] = normal["r2"]
            row[f"blind normal {target} MAE"] = normal["mae"]
            if target == "DTC":
                row["Well4 enlarged DTC down (%)"] = enlarged["down_pct"]
            else:
                row["Well4 enlarged DEN up (%)"] = enlarged["up_pct"]
            row["Well4 coverage"] = enlarged["coverage"]
        row["normal mis-gate (%)"] = float(
            np.mean(baseline.blind["CBD"].to_numpy() > tau)
        ) * 100
        blind_rows.append(row)

    sections = [
        ("Open-loop global multiplier", pd.DataFrame(open_rows)),
        ("Blind-well threshold scan", pd.DataFrame(blind_rows)),
    ]

    if closed_loop:
        closed_rows = []
        for multiplier in cfg.MULTIPLIERS:
            scaled = {w: baseline_thresholds[w] * multiplier for w in baseline_thresholds}
            with _scaled_thresholds(multiplier):
                retrained = prepare_dataset(data_dir=data_dir, use_vm=True)
                row = {"multiplier m": multiplier}
                for target in ("ZDEN", "DTC"):
                    model = CorrectionModel("xgb", target=target,
                                            use_vm=True, use_gate=True).fit(retrained)
                    val = regression_metrics(
                        baseline.validation[target].to_numpy(),
                        model.predict(baseline.validation),
                    )
                    blind = regression_metrics(
                        baseline.blind[target].to_numpy(),
                        model.predict(baseline.blind),
                    )
                    blind_enl = evaluate_target(baseline, model, baseline.enlarged_blind, "enlarged")
                    row[f"fixed val {target} R2"] = val["r2"]
                    row[f"fixed blind {target} R2"] = blind["r2"]
                    if target == "DTC":
                        row["Well4 enlarged DTC down (%)"] = blind_enl["down_pct"]
                    else:
                        row["Well4 enlarged DEN up (%)"] = blind_enl["up_pct"]
                    row["Well4 coverage"] = blind_enl["coverage"]
                row["normal training rows"] = int(len(retrained.train))
                row["delta vs m=1"] = (
                    int(len(retrained.train)) - int(len(baseline.train))
                )
                row["false gate, validation (%)"] = _gated_fraction(
                    baseline.validation, scaled
                ) * 100
                closed_rows.append(row)
        sections.append(("Closed-loop relabelling and retraining", pd.DataFrame(closed_rows)))

    _write(out_dir, "threshold_sensitivity.md",
           "CBD threshold sensitivity", sections)
    return sections[0][1]


def evaluate_target_with_multiplier(dataset: Dataset, model: CorrectionModel,
                                    frame: pd.DataFrame, multiplier: float,
                                    thresholds: dict[str, float]) -> dict:
    prediction = model.predict(frame, multiplier=multiplier)
    gated = frame["CBD"].to_numpy() > frame["WELL"].map(thresholds).to_numpy()
    return _enlarged_dict(model, frame, prediction, gated)


def evaluate_target_threshold(dataset: Dataset, model: CorrectionModel,
                              frame: pd.DataFrame, threshold: float) -> dict:
    prediction = model.predict(frame, threshold_override=threshold)
    gated = frame["CBD"].to_numpy() > threshold
    return _enlarged_dict(model, frame, prediction, gated)


def _enlarged_dict(model: CorrectionModel, frame: pd.DataFrame,
                   prediction: np.ndarray, gated: np.ndarray) -> dict:
    raw = frame[model.target].to_numpy()
    rates = direction_rates(raw, prediction)
    return {
        "n": rates["n"],
        "up_pct": rates["up_pct"],
        "down_pct": rates["down_pct"],
        "coverage": float(np.mean(gated)),
        "mean_pred": float(np.mean(prediction)),
        "mean_raw": float(np.mean(raw)),
    }
