"""CLI: generate the deterministic sample dataset."""
from __future__ import annotations

import argparse

from src.sample import SAMPLE_NOTICE, make_sample_data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate VISUELLE-shaped sample data")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--stores", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = make_sample_data(args.out, args.items, args.stores, args.seed,
                              overwrite=args.overwrite)
    print(SAMPLE_NOTICE)
    for path in result["files"]:
        print(f"[sample] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())