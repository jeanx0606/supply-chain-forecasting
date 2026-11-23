"""CLI: verify the trained-model handoff. Read-only; never fits and never switches a default."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.errors import PrototypeError
from src.handoff.forecast import load_manifest
from src.handoff.published import PUBLISHED_TABLE_7_1
from src.handoff.verify import verify_handoff

DEFAULT_HANDOFF = os.environ.get("VISUELLE_HANDOFF_DIR", "data/handoff")
DEFAULT_PAIRS = os.environ.get("VISUELLE_CLEANED_PAIRS", "data/cleaned/cleaned_pairs.csv")
DEFAULT_OUT = "reports/handoff"


def _write(out_dir, report) -> tuple:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "handoff_verification.json"
    md_path = target / "handoff_verification.md"
    json_path.write_text(json.dumps(report.as_dict(), indent=2, default=str),
                         encoding="utf-8")
    md_path.write_text(report.markdown(), encoding="utf-8")
    return json_path, md_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify the trained-model handoff")
    parser.add_argument("--handoff-dir", default=DEFAULT_HANDOFF)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--reference-dir", default=None,
                        help="directory holding predict_forecast.py / replenishment.py")
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--no-published", action="store_true",
                        help="do not compare against the published Chapter 7 table")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    handoff = Path(args.handoff_dir)
    if not handoff.exists():
        print(f"handoff directory not found: {handoff} "
              f"(set VISUELLE_HANDOFF_DIR or pass --handoff-dir)", file=sys.stderr)
        return 2
    pairs = Path(args.pairs) if args.pairs else None
    if pairs is not None and not pairs.exists():
        print(f"warning: pairs file not found: {pairs}; input checks will be skipped",
              file=sys.stderr)
        pairs = None

    try:
        manifest = load_manifest(handoff)
        report = verify_handoff(handoff, pairs_path=pairs,
                                reference_dir=args.reference_dir,
                                sample_rows=args.sample_rows, manifest=manifest,
                                published=None if args.no_published else PUBLISHED_TABLE_7_1)
    except PrototypeError as exc:
        print(exc.render(), file=sys.stderr)
        return 2

    written = _write(args.out, report)
    if args.as_json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        print(f"handoff          : {report.handoff_dir}")
        print(f"features         : {report.manifest.get('n_features', 0)}")
        for check in report.checks:
            print(f"{check.status:<8} {check.name:<26} {check.detail}")
        counts = report.counts
        print(f"counts           : pass={counts.get('pass', 0)} fail={counts.get('fail', 0)} "
              f"skipped={counts.get('skipped', 0)} info={counts.get('info', 0)}")
        if report.metrics:
            for name, values in report.metrics.items():
                line = (f"{name:<8} recomputation: WAPE {values['wape']:.4f} "
                        f"MAE {values['mae']:.4f} RMSE {values['rmse']:.4f} "
                        f"bias {values['bias']:+.4f}")
                if values.get('cum_err') is not None:
                    line += f" cum {values['cum_err']:.4f}"
                print(line)
        print(f"result           : {'PASS' if report.ok else 'FAIL'}")
        print(f"reports          : {written[0]} / {written[1]}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())