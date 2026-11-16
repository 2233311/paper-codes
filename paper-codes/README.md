# CBD-PIF-XGBoost: DEN and DTC log correction under borehole enlargement

This repository is a self-contained reference implementation of the
CBD-PIF-XGBoost workflow for correcting density (DEN) and acoustic transit
time (DTC) logs in enlarged borehole intervals. It runs end to end on the
synthetic demonstration wells shipped in `data/synthetic/`, so no proprietary
well-log data are required.

## Method in brief

1. **Input construction.** Five base inputs — GR, CNCF, log10 deep
   resistivity, log10 shallow resistivity, and their ratio — are expanded into
   a five-sample depth window, giving a 25-dimensional feature vector per
   depth point.
2. **CBD conditional gating.** The caliper–bit difference
   `CBD = CAL - BIT` is compared with a per-well threshold. During prediction,
   points above the threshold have every `CNCF*` and `LG_RS*` window feature
   replaced by its training-set mean. The mask is applied per row: at a gated
   depth all ten contaminated components are replaced, including the components
   that come from neighbouring depth samples. This removes the contaminated
   inputs without changing the tree structure of the regressor.
3. **Physics-inspired features.** A Ridge model maps GR and log10 deep
   resistivity to core-plug porosity. A three-component volume model converts
   porosity to `ZDEN_VM`, and the Raymer–Hunt–Gardner relation converts
   porosity to `DTC_VM`. The corresponding physical estimate is appended as a
   26th input feature.
4. **Regressor.** XGBoost with 300 trees, maximum depth 6, learning rate 0.05,
   subsample and column-subsample ratios of 0.8, and random seed 42.

The model is trained only on non-enlarged samples. Labels are expanded by
±3 depth samples around each enlarged run, with expansion stopped across
depth gaps larger than 0.5 m. Expanded samples are used for labelling and
evaluation and are never used as training samples.

Evaluation follows two regimes:

- **Non-enlarged intervals** have measured targets; R², MAE, RMSE and MAPE are
  reported on a held-out 20% split of the training wells and on the blind well.
- **Enlarged intervals** have no reliable point-by-point ground truth. Only
  direction consistency (upward for DEN, downward for DTC), the share of
  expanded labels whose CBD exceeds the calibrated threshold (reported as
  reference gate coverage; it is a property of the labels and is therefore
  the same for gated and ungated models), and the deviation from the
  non-enlarged reference level are reported.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # PyTorch, for LSTM/RNN/GRU rows

python scripts/generate_synthetic_data.py
python scripts/run_all.py --closed-loop
python scripts/make_fig1.py
```

`run_all.py` regenerates every table under `results/tables/`. On a four-core
CPU the full sequence takes roughly 5–10 minutes; most of that time is the
closed-loop sweep, which can be skipped by omitting `--closed-loop`.

Individual experiments can be run separately:

```bash
python scripts/run_main.py
python scripts/run_model_comparison.py
python scripts/run_ablation.py
python scripts/run_threshold_sensitivity.py            # open loop
python scripts/run_threshold_sensitivity.py --closed-loop
python scripts/make_fig1.py                            # Fig. 1 response analysis
```

If PyTorch is not installed, `run_model_comparison.py` records the LSTM, RNN
and GRU entries as skipped and continues with the remaining learners.

## Repository layout

```
paper-codes/
├── cbdxgb/
│   ├── config.py          # thresholds, bins, hyperparameters
│   ├── data.py            # CSV loading and derived columns
│   ├── features.py        # enlargement labels and 25-D windows
│   ├── physics.py         # porosity model, volume model, RHG
│   ├── pipeline.py        # dataset preparation and CorrectionModel
│   ├── learners.py        # ten regressors
│   ├── evaluation.py      # metrics and Markdown rendering
│   ├── experiments.py     # main, comparison, ablation, threshold sweeps
│   └── synthetic.py       # synthetic well generator
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_main.py
│   ├── run_model_comparison.py
│   ├── run_ablation.py
│   ├── run_threshold_sensitivity.py
│   ├── make_fig1.py
│   └── run_all.py
├── data/synthetic/        # four wells + core porosity (generated)
├── docs/data_dictionary.md
├── results/
│   ├── tables/            # generated Markdown and CSV summaries
│   └── figures/           # generated Fig. 1 (PNG and PDF)
├── requirements.txt
├── requirements-optional.txt
└── LICENSE
```

## Generated outputs

`results/tables/` contains:

| File | Content |
| --- | --- |
| `main_results.md` | CBD-PIF-XGBoost versus plain XGBoost: non-enlarged accuracy, enlarged direction and deviation, blind-well CBD bins |
| `model_comparison.md` | Ten learners under a uniform 25-dimensional input |
| `ablation.md` | 2 × 2 ablation of gating and physics-inspired features for XGBoost and Random Forest |
| `threshold_sensitivity.md` | Open-loop multiplier scan, blind-well threshold scan, and optional closed-loop relabelling/retraining |

Each Markdown file is accompanied by one CSV per table. `results/figures/`
contains `fig1_cbd_response.png` and `fig1_cbd_response.pdf`, produced by
`scripts/make_fig1.py`. The numeric values depend on the synthetic generator
seed and on the installed library versions, and are not intended to reproduce
the figures reported for the field data set.

## Synthetic demonstration data

`synthetic.py` generates four wells on a common 3000–3400 m column at 0.1 m
spacing. Each well has

- a shared stratigraphic trend in GR, porosity, resistivity and the two
  targets;
- one to three smooth enlargement intervals whose peak CBD differs by well;
- enlargement-induced contamination: CNCF increases, both resistivities
  decrease in common mode (a little more strongly for the shallow curve, so
  RD/RS stays near its non-enlarged level), DEN decreases and DTC increases
  with CBD;
- sparse core-plug porosity samples in `core_porosity.csv`.

The per-well thresholds are 1.25, 1.50, 1.00 and 0.50 in for Well1–Well4. The
synthetic data contain no field measurements and must not be used to draw
geological conclusions. They exist only to exercise the code path and the
table generation.

## Reproducibility notes

- Random seed 42 controls the train/validation split, XGBoost, Random Forest,
  the neural networks and the synthetic generator.
- The 80/20 split is random at the sample level, so adjacent 0.1 m samples can
  appear in both parts. The validation set therefore measures interpolation
  within the training wells; cross-well generalization is tested only on the
  blind well.
- Exact metrics also depend on the installed library versions; the committed
  tables were produced with the versions resolved from `requirements.txt` and
  a CPU build of PyTorch.
- Per-well CBD thresholds are fixed in `cbdxgb/config.py`. The sensitivity
  experiments scale them through `CorrectionModel.predict(multiplier=...)` or
  replace them with `threshold_override=...`; editing the config changes the
  calibrated baseline itself.
- `CorrectionModel.predict(..., multiplier=m)` scales every calibrated
  threshold by `m`; `threshold_override=tau` replaces all thresholds with a
  single value for the blind-well scan.
- The gating mask is applied to raw window features before standardization,
  using training-set means. The scaler itself is fitted on ungated training
  features.
- The closed-loop sweep relabels and retrains the model for each threshold,
  but always evaluates on the reference sets fixed at the calibrated label.

## Data availability

The field well logs used in the associated study are subject to a
confidentiality agreement and are not included in this repository. This
repository provides the complete method implementation, the synthetic
demonstration data with an identical field structure, and the summary outputs
produced by this code on that synthetic data. The values reported for the
field-data tables cannot be reproduced exactly without the original logs; the
demonstration verifies the workflow, not the field-data values.

## License

MIT. See `LICENSE`.
