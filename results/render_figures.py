"""Render screenshot-ready PNGs from one published bundle. Evidence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOOTER = "prototype self-test (sample data, run.seed 42) - not results"


def _load(bundle_path: Path) -> dict:
    return json.loads(bundle_path.read_text(encoding="utf-8"))


def _finish(fig, axes, path: Path, unit_note: str) -> Path:
    for axis in axes:
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
    fig.text(0.01, 0.012, f"{FOOTER} | unit: {unit_note}", fontsize=7.5,
             color="#555555")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def figure_model_selection(bundle: dict, out: Path) -> Path:
    figures = bundle["selection"]["figures"]
    chosen = bundle["selection"]["chosen_model"]
    names = [f["candidate"] for f in figures if f["available"]]
    values = [f["value"] for f in figures if f["available"]]
    colors = ["#2e6b4f" if n == chosen else "#9aa4ad" for n in names]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(names, values, color=colors, edgecolor="#4a4a4a",
                  linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0015,
                f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("WAPE item-level (ratio, lower is better)")
    ax.set_title("Model selection - validation season (WAPE item-level)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", labelrotation=20)
    ax.axhline(min(values), color="#2e6b4f", linewidth=1.0, linestyle=":")
    ax.text(0.99, 0.94, f"chosen: {chosen} (REP-02, REP-18 vs B2 category_curve)",
            transform=ax.transAxes, ha="right", fontsize=9, color="#2e6b4f")
    return _finish(fig, [ax], out / "fig1_model_selection.png", "ratio")


def figure_sigma_e(bundle: dict, out: Path) -> Path:
    scales = bundle["error_scale_by_model"]
    names = list(scales)
    values = [scales[n]["sigma_e"] for n in names]
    note = scales[names[0]]["unit"]
    fig, ax = plt.subplots(figsize=(10, 5.0))
    bars = ax.bar(names, values, color="#6b7d8f", edgecolor="#3f4a55",
                  linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03,
                f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("sigma_e - standard deviation of weekly forecast error")
    ax.set_title("Sigma_e per model - validation season, standard deviation "
                 "(REP-03, REP-10, REP-19)")
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(axis="x", labelrotation=20)
    return _finish(fig, [ax], out / "fig2_sigma_e_standard_deviation.png", note)


def figure_error_slices(bundle: dict, out: Path) -> Path:
    slices = [s for s in bundle["error_slices"]["slices"]
              if s["status"] == "reported"]
    slices = sorted(slices, key=lambda s: s["wape"])
    worst_entry = bundle["error_slices"]["worst_groups"][0]
    worst = f"{worst_entry['slice']}:{worst_entry['group']}"
    labels = [f"{s['slice']}: {s['group']} (n={s['n']})" for s in slices]
    values = [s["wape"] for s in slices]
    colors = ["#a83232" if f"{s['slice']}:{s['group']}" == worst else "#8f9aa6"
              for s in slices]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.barh(labels, values, color=colors, edgecolor="#3f4a55",
                   linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(value + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=8.5)
    ax.set_xlabel("WAPE (ratio, lower is better)")
    ax.set_title("Forecast-error slices - test season, min slice size 10 "
                 "(REP-03)")
    ax.set_xlim(0, max(values) * 1.18)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.text(0.99, 0.02, f"worst slice: {worst}", transform=ax.transAxes,
            ha="right", fontsize=9, color="#a83232")
    return _finish(fig, [ax], out / "fig3_error_slices.png", "ratio")


def figure_scenario_ladder(bundle: dict, out: Path) -> Path:
    rows = bundle["scenarios"]["rows"]
    names = [r["scenario"] for r in rows]
    fill = [r["fill_rate"] * 100 for r in rows]
    stockout = [r["stockout_rate"] * 100 for r in rows]
    overstock = [r["overstock_ratio"] * 100 for r in rows]
    cover = [r["weeks_of_cover"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    axes[0].bar(names, fill, color="#2e6b4f", edgecolor="#3f4a55",
                linewidth=0.5)
    axes[0].set_title("Fill rate (%) - REP-10, REP-11")
    axes[0].set_ylim(80, 102)
    axes[1].bar(names, stockout, color="#a83232", edgecolor="#3f4a55",
                linewidth=0.5)
    axes[1].set_title("Stock-out rate (%) - REP-10, REP-11")
    axes[2].bar(names, cover, color="#6b7d8f", edgecolor="#3f4a55",
                linewidth=0.5)
    axes[2].set_title("Weeks of cover (weeks)")
    for axis, values in zip(axes, (fill, stockout, cover)):
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center",
                      va="bottom", fontsize=7.5)
        axis.tick_params(axis="x", labelrotation=25, labelsize=8)
    axes[2].set_ylim(0, max(cover) * 1.28)
    axes[2].text(0.02, 0.98, f"overstock ratio max "
                 f"{max(overstock):.2f}%", transform=axes[2].transAxes,
                 ha="left", va="top", fontsize=8)
    fig.suptitle("Scenario ladder at the main setting (L=1, R=1, z=95%)",
                 fontsize=11)
    return _finish(fig, list(axes), out / "fig4_scenario_ladder.png",
                   "percent / weeks")


def figure_demo_forecast(bundle: dict, out: Path) -> Path:
    demo = bundle["demo_forecast"]
    replenishment = bundle["replenishment"]
    weeks = [f"week {w}" for w in demo["weeks"]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(weeks, demo["actual"], marker="o", color="#3f4a55",
            linewidth=1.4, label="actual (test weeks 3-12)")
    ax.plot(weeks, demo["predicted"], marker="s", color="#2e6b4f",
            linewidth=1.4, linestyle="--",
            label=f"forecast - {bundle['main_model']}")
    ax.axhline(replenishment["reorder_point"], color="#a83232",
               linestyle=":", linewidth=1.2,
               label=f"reorder point {replenishment['reorder_point']:.2f}")
    ax.axhline(replenishment["target_level"], color="#6b2f8f",
               linestyle="-.", linewidth=1.2,
               label=f"target level {replenishment['target_level']:.2f}")
    ax.set_ylabel("units (raw_sales_units)")
    ax.set_title("Demonstration item - forecast vs actual with replenishment "
                 "thresholds (REP-07, REP-10, REP-17)")
    ax.set_ylim(0, max(max(demo["actual"]), replenishment["target_level"]) * 1.15)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    return _finish(fig, [ax], out / "fig5_demo_forecast.png",
                   "raw_sales_units")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render evidence PNGs")
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)
    bundle_path = (Path(args.bundle) if args.bundle
                   else sorted(Path("output").glob("run-*/bundle.json"))[-1])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = _load(bundle_path)
    written = [
        figure_model_selection(bundle, out),
        figure_sigma_e(bundle, out),
        figure_error_slices(bundle, out),
        figure_scenario_ladder(bundle, out),
        figure_demo_forecast(bundle, out),
    ]
    print(f"bundle: {bundle_path}")
    for path in written:
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())