# Data dictionary

The schema below is shared by the field-data workflow and the synthetic
demonstration data. Each well is one CSV file with one row per depth sample,
sorted by depth. All files are UTF-8 encoded with a comma separator and a
header row.

## Well-log files (`data/synthetic/well1.csv` … `well4.csv`)

The ranges below describe the synthetic demonstration data shipped with this
repository. Field logs follow the same schema but may extend beyond these
bounds.

| Column | Unit | Range (synthetic data) | Description |
| --- | --- | --- | --- |
| `DEPTH` | m | 3000–3400 | Measured depth, sampled every 0.1 m |
| `GR` | API | 20–140 | Natural gamma ray |
| `CNCF` | % | 10–32 | Compensated neutron porosity; increases in enlarged intervals |
| `RD` | ohm·m | 0.5–12 | Deep laterolog resistivity; decreases weakly in enlarged intervals (common mode with `RS`) |
| `RS` | ohm·m | 0.5–10 | Shallow laterolog resistivity; decreases more strongly in enlarged intervals |
| `BIT` | in | 8.5 or 12.25 | Bit diameter |
| `CAL` | in | BIT to BIT + 4 | Caliper borehole diameter |
| `ZDEN` | g/cm³ | 2.20–2.60 | Compensated bulk density target; biased low in enlarged intervals |
| `DTC` | μs/ft | 70–100 | Compressional transit time target; biased high in enlarged intervals |

## Core-porosity file (`data/synthetic/core_porosity.csv`)

| Column | Unit | Description |
| --- | --- | --- |
| `WELL` | — | Well identifier (`Well1` … `Well4`) |
| `DEPTH` | m | Core-plug depth, matched to the nearest log sample within 0.05 m |
| `PHI_CORE` | % | Measured core porosity (8–22% in the synthetic data), used only to fit the porosity model |

## Derived columns (computed at load time)

| Column | Unit | Definition |
| --- | --- | --- |
| `CBD` | in | `CAL - BIT`; the caliper–bit difference used for conditional gating |
| `LG_RD` | log10(ohm·m) | `log10(RD)` |
| `LG_RS` | log10(ohm·m) | `log10(RS)` |
| `R_ratio` | — | `RD / RS`; stays near its non-enlarged level because both resistivities share the common-mode decrease |
| `GATE` | 0/1 | 1 when `CBD` exceeds the calibrated per-well threshold |
| `LABEL` | — | `normal` or `enlarged`; `enlarged` includes the ±3-sample boundary expansion |
| `VM_PHI` | % | Porosity predicted by the Ridge model from `GR` and `LG_RD` |
| `ZDEN_VM` | g/cm³ | Volume-model density from `VM_PHI` and GR-derived shale volume |
| `DTC_VM` | μs/ft | Raymer–Hunt–Gardner transit time from `VM_PHI` |
| `DTC_MA` | μs/ft | Calibrated matrix transit time (65.7 μs/ft in the demonstration run) |

## Window-feature naming

For each base feature in `GR`, `CNCF`, `LG_RD`, `LG_RS`, `R_ratio`, five
columns are created:

```
FEATURE_-2, FEATURE_-1, FEATURE, FEATURE_+1, FEATURE_+2
```

The suffix is the depth offset relative to the current sample. Together the
five base features produce 25 columns. `ZDEN_VM` or `DTC_VM` is appended as the
26th feature according to the target.

Window components are taken from adjacent rows of the same well. When the
spacing between the current row and a source row exceeds 1.5 times the median
sampling step per offset, the component is set to NaN and the incomplete row
is dropped, so a window never spans a depth gap.

## Calibrated parameters

| Parameter | Value | Role |
| --- | --- | --- |
| CBD threshold, Well1 | 1.25 in | Gating and labelling |
| CBD threshold, Well2 | 1.50 in | Gating and labelling |
| CBD threshold, Well3 | 1.00 in | Gating and labelling |
| CBD threshold, Well4 | 0.50 in | Gating and labelling; Well4 is the blind well |
| Matrix density | 2.65 g/cm³ | Volume model |
| Shale density | 2.70 g/cm³ | Volume model |
| Fluid density | 1.00 g/cm³ | Volume model |
| Fluid transit time | 189.0 μs/ft | Raymer–Hunt–Gardner |
| Matrix transit time | 65.7 μs/ft | Calibrated on non-enlarged training samples in the demonstration run; the field-data workflow yields 64.40 μs/ft as reported in the paper |
| GR endpoints | 30 / 150 API | Linear shale-volume index |
| Random seed | 42 | Split and model fitting |
