"""CLI: run the end-to-end pipeline."""
from __future__ import annotations

import argparse
import json
import sys

from src.errors import PrototypeError
from src.pipeline import run_pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the prototype pipeline")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        bundle = run_pipeline(args.config, args.root, verbose=not args.quiet)
    except PrototypeError as exc:
        print(exc.render(), file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(bundle.to_dict(), indent=2, default=str))
    else:
        print(f"main model      : {bundle.main_model}")
        print(f"metrics         : {', '.join(bundle.metrics)}")
        print(f"WAPE (item)     : {bundle.metrics[bundle.main_model].wape_item_level}")
        print(f"reorder point   : {bundle.replenishment.reorder_point}")
        print(f"target level    : {bundle.replenishment.target_level}")
        print(f"recommended qty : {bundle.replenishment.recommended_qty}")
        print(f"risk            : stockout={bundle.risk.stockout_proxy} "
              f"overstock={bundle.risk.overstock}")
        print(f"artefacts       : {bundle.artefact_dir}")
        if bundle.self_test:
            print("SELF-TEST DATA - figures are not results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())