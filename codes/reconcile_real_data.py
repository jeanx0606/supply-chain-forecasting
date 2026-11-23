"""CLI: reconcile the cleaned pair contract against the signed Step 1 figures.

Read-only. It does not run the pipeline and does not publish a figure; it states
which declared counts the cleaned contract settles and which need the raw files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from src.errors import PrototypeError
from src.preprocess.reconcile import reconcile_cleaned_pairs

DEFAULT_PAIRS = os.environ.get("VISUELLE_CLEANED_PAIRS", "data/cleaned/cleaned_pairs.csv")
DEFAULT_OUT = "reports/reconcile"


def _write(out_dir, report) -> tuple:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "cleaned_pairs_reconciliation.json"
    md_path = target / "cleaned_pairs_reconciliation.md"
    json_path.write_text(json.dumps(report.as_dict(), indent=2, default=str),
                         encoding="utf-8")
    md_path.write_text(report.markdown(), encoding="utf-8")
    return json_path, md_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile the cleaned pair contract")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    pairs = Path(args.pairs)
    if not pairs.exists():
        print(f"cleaned pairs file not found: {pairs} "
              f"(set VISUELLE_CLEANED_PAIRS or pass --pairs)", file=sys.stderr)
        return 2

    try:
        frame = pd.read_csv(pairs)
        report = reconcile_cleaned_pairs(frame)
    except PrototypeError as exc:
        print(exc.render(), file=sys.stderr)
        return 2
    except pd.errors.ParserError as exc:
        print(f"could not read {pairs}: {exc}", file=sys.stderr)
        return 2

    written = _write(args.out, report)
    if args.as_json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(f"pairs            : {report.pair_count}")
        for item in report.figures:
            measured = "n/a" if item.measured is None else item.measured
            print(f"{item.status:<17} {item.name:<30} declared={item.declared} "
                  f"measured={measured}")
        if report.structural:
            print("structural checks:")
            for item in report.structural:
                print(f"{item.status:<17} {item.name:<30} measured={item.measured}")
        print(f"result           : {'PASS' if report.ok else 'FAIL'}")
        if report.unsettled:
            print("not settled here: " + ", ".join(item.name for item in report.unsettled))
        print(f"reports          : {written[0]} / {written[1]}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())