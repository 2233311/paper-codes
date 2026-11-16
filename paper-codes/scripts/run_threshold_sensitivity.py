#!/usr/bin/env python
"""Run open-loop, blind-well and optionally closed-loop threshold sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbdxgb.config import DATA_DIR, TABLE_DIR  # noqa: E402
from cbdxgb.experiments import run_threshold_sensitivity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--closed-loop", action="store_true",
                        help="also relabel and retrain the model for each threshold")
    args = parser.parse_args()
    table = run_threshold_sensitivity(args.data_dir, args.out_dir, args.closed_loop)
    print(table.to_string(index=False))
    print(f"\nTables written to: {args.out_dir}")


if __name__ == "__main__":
    main()
