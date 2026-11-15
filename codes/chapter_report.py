"""CLI: write the chapter figure pack from one published bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config.loader import load_config
from src.errors import PrototypeError
from src.report.chapter import write_pack


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the chapter figure pack")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--bundle", default=None,
                        help="path to bundle.json; defaults to the newest run")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, root=args.root)
        if not cfg.report.enabled:
            print("report.enabled is false - nothing written")
            return 0
        if args.bundle:
            bundle_path = Path(args.bundle)
        else:
            runs = sorted(cfg.output.dir.glob("run-*/bundle.json"))
            if not runs:
                print("no bundle found; run python -m codes.run_pipeline first",
                      file=sys.stderr)
                return 1
            bundle_path = runs[-1]
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        path = write_pack(bundle, cfg.report.dir, bundle.get("run_id", "run"))
    except (OSError, ValueError, PrototypeError) as exc:
        print(f"chapter pack failed: {exc}", file=sys.stderr)
        return 2
    print(f"chapter pack written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())