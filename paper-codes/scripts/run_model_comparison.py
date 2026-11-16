#!/usr/bin/env python
"""Compare ten regressors under a uniform 25-dimensional input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbdxgb.config import DATA_DIR, TABLE_DIR  # noqa: E402
from cbdxgb.experiments import run_model_comparison  # noqa: E402
from cbdxgb.learners import LEARNER_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--learners", type=str, default=",".join(LEARNER_NAMES),
                        help="comma-separated subset of: " + ", ".join(LEARNER_NAMES))
    args = parser.parse_args()
    learners = [item.strip() for item in args.learners.split(",") if item.strip()]
    table = run_model_comparison(args.data_dir, args.out_dir, learners)
    print(table.to_string(index=False))
    print(f"\nTables written to: {args.out_dir}")


if __name__ == "__main__":
    main()
