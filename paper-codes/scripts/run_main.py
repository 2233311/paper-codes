#!/usr/bin/env python
"""Run the main CBD-PIF-XGBoost versus plain XGBoost comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbdxgb.config import DATA_DIR, TABLE_DIR  # noqa: E402
from cbdxgb.experiments import run_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=TABLE_DIR)
    args = parser.parse_args()
    table = run_main(args.data_dir, args.out_dir)
    print(table.to_string(index=False))
    print(f"\nTables written to: {args.out_dir}")


if __name__ == "__main__":
    main()
