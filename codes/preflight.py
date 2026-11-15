"""CLI: read-only data preflight (A10). Never runs a stage or a fit."""
from __future__ import annotations

import argparse
import json
import sys

from src.config.loader import load_config
from src.errors import PrototypeError
from src.io.preflight import preflight_exit_code, run_preflight


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only data preflight")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, root=args.root)
        report = run_preflight(cfg)
    except PrototypeError as exc:
        print(exc.render(), file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"data class       : {report['data_class']['data_class']}")
        for label, entry in report["files"].items():
            print(f"{label:<17}: {entry.get('status')} ({entry.get('file')})")
        if "n_rows" in report:
            print(f"rows             : {report['n_rows']}")
            print(f"items / stores   : {report['n_items']} / {report['n_stores']}")
            print(f"pair coverage    : {report['pair_coverage']:.1%}")
            print(f"negative cells   : {report['negative_cells']} "
                  f"(reference {report['negative_reference']})")
            print(f"dates            : {report['date_min']} .. {report['date_max']}")
            print(f"monday violations: {report['monday_violations']}")
        print(f"optional files   : {[o['file'] for o in report['optional_files']]}")
        print(f"conditional      : {report['conditional']}")
        print(f"readiness        : {report['readiness']}")
    return preflight_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())