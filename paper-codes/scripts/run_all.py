#!/usr/bin/env python
"""Regenerate the synthetic data and every summary table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbdxgb.config import DATA_DIR, TABLE_DIR  # noqa: E402
from cbdxgb.experiments import (  # noqa: E402
    run_ablation,
    run_main,
    run_model_comparison,
    run_threshold_sensitivity,
)
from cbdxgb.synthetic import generate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--closed-loop", action="store_true")
    args = parser.parse_args()

    generate_dataset(args.data_dir)
    run_main(args.data_dir, args.out_dir)
    run_model_comparison(args.data_dir, args.out_dir)
    run_ablation(args.data_dir, args.out_dir)
    run_threshold_sensitivity(args.data_dir, args.out_dir, args.closed_loop)
    print(f"All outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
