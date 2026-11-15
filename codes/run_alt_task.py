"""CLI: exercise the alternative 4-in/1-out rolling task (D8, REP-16)."""
from __future__ import annotations

import argparse
import sys

from src.config.loader import load_config
from src.errors import PrototypeError
from src.features.tasks import alt_task_summary
from src.io.loader import load_sales
from src.preprocess.cleaning import clean_sales
from src.preprocess.spine import ItemStoreGrain
from src.sample import SAMPLE_NOTICE, is_sample_dataset
from src.seams.factory import build_seams


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the 4-in / 1-out rolling task")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, root=args.root)
        seams = build_seams(cfg)
        if not cfg.tasks.alt_4in1out_enabled:
            print("tasks.alt_4in1out.enabled is false - nothing to run")
            return 0
        sales, _ = load_sales(cfg.data.raw_dir / cfg.data.files.sales,
                              cfg.data.n_weeks, cfg.data.sales_week_prefix)
        frame = clean_sales(sales, cfg.data.n_weeks, cfg.data.sales_week_prefix,
                        policy=seams.sign_policy).frame
        frame = ItemStoreGrain().build(frame).reset_index(drop=True)
        points = seams.alt_task.build(frame, cfg.data.n_weeks)
    except PrototypeError as exc:
        print(exc.render(), file=sys.stderr)
        return 2

    summary = alt_task_summary(points)
    for point in summary:
        print(f"origin week {point['origin_week']:2d} -> {point['labels']} "
              f"({point['n_rows']} rows)")
    print(f"{len(summary)} evaluation points (4-in / 1-out)")
    if is_sample_dataset(cfg.data.raw_dir):
        print(SAMPLE_NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())