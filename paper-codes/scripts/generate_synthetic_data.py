#!/usr/bin/env python
"""Generate the synthetic demonstration wells and core-porosity samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cbdxgb.synthetic import generate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = generate_dataset(args.out_dir, seed=args.seed)
    print(f"Synthetic data written to: {target}")


if __name__ == "__main__":
    main()
