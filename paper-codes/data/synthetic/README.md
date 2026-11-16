# Synthetic demonstration data

This folder is populated by `python scripts/generate_synthetic_data.py`.

- `well1.csv` … `well4.csv`: four synthetic wells, 3000–3400 m at 0.1 m spacing.
- `core_porosity.csv`: sparse synthetic core-plug porosity samples.

The files contain **no field measurements**. They reproduce the column
structure, units and contamination behaviour of the field-data workflow so
that every script can run without proprietary data. See
`docs/data_dictionary.md` for the schema.

The generator is deterministic for a fixed seed (`42` by default) and can be
re-run at any time:

```bash
python scripts/generate_synthetic_data.py --seed 42
```
