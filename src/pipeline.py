"""End-to-end orchestration: S0 configure -> S13 publish."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConfigLedger, effective_config_dict, load_config
from .errors import StageError
from .evaluate.aggregate import evaluate_split
from .evaluate.error_scale import measure as measure_error_scale
from .evaluate.secondary import compare_on_preset
from .evaluate.slices import error_slices
from .features.build import TaskDataset, build_task
from .features.tasks import alt_task_summary
from .io.channels import probe_channels
from .io.loader import (load_price, load_restocks, load_sales, load_weather)
from .io.preflight import classify_dataset
from .io.unit_contract import RAW_UNIT
from .models.registry import get_model
from .models.selection import SelectionGuard
from .preprocess.cleaning import clean_sales, quality_report
from .preprocess.inventory import (ReconstructionPolicy, audit_reconstruction,
                                    build_trajectory)
from .preprocess.expected_counts import (EXPECTED_A4_PAIRS,
                                         EXPECTED_A4_PAIRS_REQUIRING_A3,
                                         EXPECTED_NEGATIVE_PAIRS,
                                         EXPECTED_NEGATIVE_PAIRS_WITH_A4,
                                         EXPECTED_STOCKOUT_PAIRS)
from .preprocess.restock import make_retention_policy
from .preprocess.time_index import (attach_week_index, check_window,
                                    monday_violations, window_summary)
from .preprocess.weather import interpolate_localities, to_weekly
from .replenish.demo import DemoCandidate
from .replenish.policy import (ReplenishmentContext, recommend, recommend_baseline)
from .replenish.scenario import (ScenarioPair, ScenarioSettings, ladder_payload,
                                 run_ladder)
from .replenish.risk import assess
from .report.assumptions import as_dicts as assumption_records
from .report.bundle import (QualityReport, ResultBundle, SplitProvenance,
                             UnitProvenance)
from .report.provisional import load_register, provisional_refs, register_version
from .sample import SAMPLE_NOTICE, is_sample_dataset
from .seams.factory import build_seams

STAGES = (
    "S0 configure", "S1 ingest", "S2 clean", "S3 index", "S4 spine",
    "S5 restock", "S6 weather", "S7 features", "S8 split", "S9 fit",
    "S10 evaluate", "S11 inventory", "S12 recommend", "S13 publish",
)
SAMPLE_NEXT = "python -m codes.sample_data"


def _weekly_sales(row, n_weeks: int) -> tuple:
    values = []
    for week in range(1, int(n_weeks) + 1):
        value = row.get(f"q{week}", 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if value != value:
            value = 0.0
        values.append(value)
    return tuple(values)


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _consume_ledger(cfg, ledger: ConfigLedger) -> None:
    ledger.consume(
        "run.seed", "run.verbose", "time.week_convention", "spine.granularity",
        "spine.input_weeks", "spine.output_weeks", "split.policy",
        "split.train_ratio", "split.val_ratio", "split.test_ratio",
        "split.explicit_cut_dates", "split.use_preset_files", "model.baseline",
        "model.selected", "model.backend", "model.params", "baseline.naive_reference",
        "metrics.primary", "metrics.supporting", "channels.weather",
        "channels.gtrends", "weather.join_mode", "inventory.initial_stock_method",
        "inventory.lead_time_weeks", "inventory.safety_factor",
        "inventory.opening_stock_alpha", "inventory.negative_correction",
        "inventory.floor_at_zero",
        "truncation.restock_policy", "truncation.restock_date_max",
        "replenish.policy", "replenish.safety_stock_method",
        "replenish.service_level", "replenish.order_cycle_weeks",
        "replenish.risk_high_weeks", "replenish.risk_medium_weeks",
        "replenish.overstock_high_ratio", "replenish.overstock_medium_ratio",
        "demo.selection_rule", "demo.pin_item", "demo.pin_store",
        "tasks.alt_4in1out.enabled", "output.enabled", "output.dir",
        "output.format", "cleaning.negative_sales_policy",
        "split.preset_scale_factor", "split.preset_files",
        "model.candidates", "model.selection_rule", "model.selection_metric",
        "model.qualification.mode", "model.qualification.tolerance",
        "model.qualification.on_none", "features.lags",
        "features.history_aggregates", "features.price_discount_interaction",
        "features.weather",
    )


def run(cfg, verbose: bool | None = None) -> ResultBundle:
    say = (lambda *a: print(*a)) if (cfg.run.verbose if verbose is None else verbose) else (lambda *a: None)
    ledger = ConfigLedger()
    _consume_ledger(cfg, ledger)
    check_window(cfg.spine.input_weeks, cfg.spine.output_weeks, cfg.data.n_weeks)
    seams = build_seams(cfg)
    seams.preset.require()
    say(f"[S0] {window_summary(cfg.spine.input_weeks, cfg.spine.output_weeks, cfg.data.n_weeks)}")

    # ---- S1 ingest ----------------------------------------------------
    raw = cfg.data.raw_dir
    names = cfg.data.files
    mandatory = {"sales": names.sales, "restocks": names.restocks,
                 "price_discount": names.price_discount}
    for label, fname in mandatory.items():
        if not (raw / fname).exists():
            raise StageError("ingest", f"missing mandatory input '{fname}' ({label})",
                             next_command=SAMPLE_NEXT)
    ledger.consume("data.raw_dir", "data.files.sales", "data.files.restocks",
                   "data.files.price_discount", "data.files.gtrends",
                   "data.files.weather", "data.sales_week_prefix", "data.n_weeks")

    sales, sales_stats = load_sales(raw / names.sales, cfg.data.n_weeks,
                                    cfg.data.sales_week_prefix)
    restocks, _ = load_restocks(raw / names.restocks)
    price, _ = load_price(raw / names.price_discount, cfg.data.n_weeks)
    registry = probe_channels(cfg)
    preset_files = cfg.split.preset_files
    preset_present = bool(preset_files) and all(
        (raw / name).exists() for name in preset_files)
    say(f"[S1] {len(sales)} sales rows, {len(restocks)} restock events, "
        f"{len(price)} price rows")

    # ---- S2 clean -----------------------------------------------------
    cleaned = clean_sales(sales, cfg.data.n_weeks, cfg.data.sales_week_prefix,
                          policy=seams.sign_policy)
    sign = cleaned.sign_outcome
    frame = cleaned.frame
    violations = monday_violations(frame["release_date"])
    if violations:
        print(f"[S2] warning: {violations} release dates are not Monday")
    say(f"[S2] kept {len(frame)} rows, excluded {cleaned.excluded}; "
        f"sign_policy={sign.policy_name}, negative_cells={sign.negative_cells}")

    # ---- S3 index -----------------------------------------------------
    frame = attach_week_index(frame, cfg.data.n_weeks)
    price = price.drop_duplicates(subset=["external_code", "retail"], keep="first")
    disc_columns = [c for c in price.columns
                    if c.startswith("disc_w") and c != "disc_w1"]
    price_columns = ["external_code", "retail", "price", "disc_w1"] + disc_columns
    frame = frame.merge(price[price_columns],
                        on=["external_code", "retail"], how="left").reset_index(drop=True)
    say(f"[S3] week index attached through week_{cfg.data.n_weeks}_date")

    # ---- S4 spine -----------------------------------------------------
    frame = seams.grain.build(frame).reset_index(drop=True)
    say(f"[S4] grain={seams.grain.describe().implementation}, {len(frame)} series")

    # ---- S5 restock retention ----------------------------------------
    release_map = {
        (int(row.external_code), int(row.retail)): row.release_date.date()
        for row in frame[["external_code", "retail", "release_date"]].itertuples(index=False)
    }
    outcome = seams.retention.retain(restocks, release_map, cfg.data.n_weeks)
    say(f"[S5] policy={outcome.policy_name}, excluded={outcome.excluded}, "
        f"{outcome.restock_date_max_state['parameter']}={outcome.restock_date_max_state['state']}")

    # ---- S5b reconstruction audit (A3-on only, both figures) -----------
    reconstruction = ReconstructionPolicy(
        alpha=cfg.inventory.opening_stock_alpha,
        negative_correction=cfg.inventory.negative_correction,
        floor_at_zero=cfg.inventory.floor_at_zero)
    audit = audit_reconstruction(
        frame, outcome.events_by_key, outcome.opening_by_key,
        seams.initial_stock, cfg.data.n_weeks)
    reconstruction_audit = {
        **audit.as_dict(),
        "a3_switch": "on" if cfg.inventory.negative_correction else "off",
        "reported_a3_configuration": "on",
        "alpha": float(cfg.inventory.opening_stock_alpha),
        "definition": ("stock <= 0 in weeks 0-11 before the A3 adjustment; "
                       "the reported configuration is A3-on"),
        "partitions": "all cleaned pairs",
        "declared": {
            "pairs_negative_excluding_a4": EXPECTED_NEGATIVE_PAIRS,
            "pairs_negative_including_a4": EXPECTED_NEGATIVE_PAIRS_WITH_A4,
            "a4_pairs": EXPECTED_A4_PAIRS,
            "a4_pairs_requiring_a3": EXPECTED_A4_PAIRS_REQUIRING_A3,
            "pairs_zero_or_below": EXPECTED_STOCKOUT_PAIRS,
            "source": ("declared by Rue against the cleaned data; the real "
                       "dataset is not present in this run"),
        },
    }
    say(f"[S5b] pairs={reconstruction_audit['pairs']} "
        f"zero<=0={reconstruction_audit['zero_or_below_pairs']} "
        f"neg_exA4={reconstruction_audit['negative_pairs_excluding_a4']} "
        f"neg_inclA4={reconstruction_audit['negative_pairs_including_a4']} "
        f"A4={reconstruction_audit['a4_pairs']} "
        f"A4&A3={reconstruction_audit['a4_pairs_requiring_a3']} (A3-on only)")

    # ---- S6 weather ---------------------------------------------------
    wx_path = raw / names.weather
    if getattr(cfg.channels.weather, "value", cfg.channels.weather) == "consume" and wx_path.exists():
        weather, _ = load_weather(wx_path)
        weather, gaps = interpolate_localities(weather, cfg.weather.max_interp_days)
        if gaps:
            print(f"[S6] warning: {len(gaps)} localities still have gaps after interpolation")
        weekly = to_weekly(weather, dict(cfg.weather.agg))
        before = len(frame)
        frame = seams.weather_join.join(frame, weekly).reset_index(drop=True)
        if len(frame) != before:
            raise StageError("weather", "weather join changed record count")
        ledger.consume("weather.max_interp_days", "weather.agg")
        say(f"[S6] weather joined as {seams.weather_join.describe().implementation}, "
            f"{sum(c.startswith('wx_') for c in frame.columns)} wx columns")
    else:
        ledger.waive("weather.max_interp_days", "weather.agg")
        print(f"[S6] weather channel skipped (file_absent or out_of_scope)")

    # ---- S7 features --------------------------------------------------
    main_task: TaskDataset = build_task(
        frame, cfg.spine.input_weeks, cfg.spine.output_weeks, cfg.data.n_weeks,
        origin_week=cfg.spine.input_weeks, prefix=cfg.data.sales_week_prefix,
        spec=seams.features)
    alt_points = seams.alt_task.build(frame, cfg.data.n_weeks, spec=seams.features)
    say(f"[S7] features={list(main_task.features.columns)[:6]}..., "
        f"targets={main_task.horizon}, alt_points={len(alt_points)}")

    # ---- S8 split -----------------------------------------------------
    assignment = seams.split.resolve(frame["release_date"], frame.get("season"))
    preset_provenance = seams.preset.provenance(preset_present)
    train_mask = assignment.train.to_numpy()
    test_mask = assignment.test.to_numpy()
    say(f"[S8] train={assignment.counts['train']} val={assignment.counts['validation']} "
        f"test={assignment.counts['test']} cuts={assignment.cut_train_end}..{assignment.cut_validation_end}")

    # ---- S9 select on validation, refit, one test evaluation -----------
    X = main_task.features
    y = main_task.targets
    validation_mask = assignment.validation.to_numpy()
    if not validation_mask.any():
        raise StageError(
            "split", "validation partition is empty - the selection rule cannot "
                     "run; adjust split ratios")
    X_train, y_train = X.loc[train_mask], y.loc[train_mask].to_numpy()
    X_val, y_val = X.loc[validation_mask], y.loc[validation_mask].to_numpy()
    X_test, y_test = X.loc[test_mask], y.loc[test_mask].to_numpy()
    if len(X_test) == 0:
        raise StageError("split", "test set is empty - adjust split ratios")
    meta_val = main_task.metadata.loc[validation_mask].reset_index(drop=True)
    meta_test = main_task.metadata.loc[test_mask].reset_index(drop=True)

    model_candidates = list(cfg.model.candidates)
    fitted_train = {}
    figures = []
    validation_predictions = {}
    for name in model_candidates:
        model = get_model(name, seams.backend, cfg.run.seed,
                          cfg.baseline.naive_reference)
        model.fit(X_train, y_train)
        fitted_train[name] = model
        validation_predictions[name] = model.predict(X_val)
        figures.append(seams.selection_policy.score_validation(
            validation_predictions[name], y_val, meta_val, seams.metrics, name))
    decision = seams.selection_policy.choose(figures, model_candidates,
                                             cfg.model.selected)
    union_mask = train_mask | validation_mask
    chosen = get_model(decision.chosen_model, seams.backend, cfg.run.seed,
                       cfg.baseline.naive_reference)
    chosen.fit(X.loc[union_mask], y.loc[union_mask].to_numpy())
    fitted_for_test = dict(fitted_train)
    fitted_for_test[decision.chosen_model] = chosen

    guard = SelectionGuard()
    guard.seal()
    predictions: dict = {}
    for name in model_candidates:
        guard.evaluate_test(name)
        predictions[name] = fitted_for_test[name].predict(X_test)
    main_model = decision.chosen_model
    if decision.outcome == "qualified":
        main_model_status = "qualified"
    elif decision.on_none == "least_bad_provisional":
        main_model_status = "unqualified_least_bad"
    else:
        main_model_status = "unqualified_retained_default"
    fit_span = {name: ("train+validation" if name == decision.chosen_model else "train")
                for name in model_candidates}
    say(f"[S9] candidates={len(model_candidates)}; chosen={main_model}; "
        f"outcome={decision.outcome}")
    error_scale = measure_error_scale(
        y_val, validation_predictions[decision.chosen_model])
    error_scale_by_model = {
        name: measure_error_scale(y_val, prediction).as_dict()
        for name, prediction in validation_predictions.items()}
    say(f"[S9] sigma_e={error_scale.sigma_e:.4f} (validation, "
        f"n={error_scale.n_observations})")

    # ---- S10 evaluate --------------------------------------------------
    register = load_register(cfg.root / "config/replaceability.yaml")
    refs = provisional_refs(register)
    register_revision = register_version(register)
    metrics = {}
    for name, prediction in predictions.items():
        metrics[name] = evaluate_split(
            seams.metrics, y_test, prediction, meta_test, name,
            assignment.cut_train_end, assignment.cut_validation_end,
            provisional=refs, unit=RAW_UNIT,
            sign_policy=sign.policy_name, partition="test")
    say(f"[S10] { {n: round(m.wape_item_level, 4) if m.wape_item_level is not None else None for n, m in metrics.items()} }")

    # ---- S10a error slices (Record D) ----------------------------------
    test_positions = np.flatnonzero(test_mask)
    error_slice_payload = error_slices(
        y_test, predictions[main_model], X_test,
        frame.iloc[test_positions], meta_test)
    worst = error_slice_payload["worst_groups"]
    if worst:
        worst_note = (f"{worst[0]['slice']}:{worst[0]['group']} "
                      f"(WAPE {worst[0]['wape']:.3f})")
    else:
        worst_note = "n/a"
    say(f"[S10a] slices={len(error_slice_payload['slices'])} "
        f"reported={error_slice_payload['reported_count']} worst={worst_note}")

    # ---- S10b scenario ladder (main setting only) ----------------------
    ladder_pairs = []
    for index, position in enumerate(test_positions):
        key = (int(meta_test.loc[index, "external_code"]),
               int(meta_test.loc[index, "retail"]))
        pair_row = frame.iloc[int(position)]
        pair_events = outcome.events_by_key.get(key, {})
        pair_opening = outcome.opening_by_key.get(key, 0.0)
        pair_trajectory = build_trajectory(
            pair_row, pair_events, seams.initial_stock, cfg.data.n_weeks,
            key=key, opening=pair_opening, policy=reconstruction)
        ladder_pairs.append(ScenarioPair(
            key=key, sales=_weekly_sales(pair_row, cfg.data.n_weeks),
            restocks=dict(pair_events),
            opening=float(pair_trajectory.initial_stock.value)))

    naive_model = {"last_input": "seasonal_naive",
                   "mean_input": "moving_average"}[cfg.baseline.naive_reference.value]
    forecast_sources = {}
    if naive_model in predictions:
        forecast_sources["R1"] = (f"{naive_model} (B0/B1 naive)",
                                  predictions[naive_model])
    if "category_curve" in predictions:
        forecast_sources["R2"] = ("category_curve (B2)",
                                  predictions["category_curve"])
    if main_model in predictions:
        forecast_sources["R3"] = (f"{main_model} (chosen)", predictions[main_model])
    forecast_sources["R4"] = ("actual future sales (perfect foresight)", y_test)
    sigma_by_scenario = {
        "R1": float(error_scale_by_model.get(naive_model, {}).get("sigma_e", 0.0)),
        "R2": float(error_scale_by_model.get("category_curve", {}).get("sigma_e", 0.0)),
        "R3": float(error_scale.sigma_e),
        "R4": 0.0,
    }
    ladder_settings = ScenarioSettings(
        lead_time_weeks=cfg.inventory.lead_time_weeks,
        order_cycle_weeks=cfg.replenish.order_cycle_weeks,
        service_level=cfg.replenish.service_level,
        n_weeks=cfg.data.n_weeks,
        forecast_origin_week=int(main_task.origin_week))
    scenarios = ladder_payload(
        run_ladder(ladder_pairs, ladder_settings, forecast_sources,
                   sigma_by_scenario),
        ladder_settings, forecast_sources)
    say("[S10b] " + ", ".join(
        f"{row['scenario']}: fill={_fmt(row['fill_rate'])} "
        f"stockout={_fmt(row['stockout_rate'])} "
        f"overstock={_fmt(row['overstock_ratio'])}"
        for row in scenarios["rows"]))

    # ---- S11 inventory --------------------------------------------------
    positions = test_positions
    candidates = [
        DemoCandidate(key=(int(meta_test.loc[i, "external_code"]),
                           int(meta_test.loc[i, "retail"])),
                      horizon_demand=float(y_test[i].sum()), row_index=int(positions[i]))
        for i in range(len(meta_test))
    ]
    demo = seams.demo.select(candidates)
    demo_row = frame.iloc[demo.row_index]
    events = outcome.events_by_key.get(demo.key, {})
    opening = outcome.opening_by_key.get(demo.key, 0.0)
    trajectory = build_trajectory(demo_row, events, seams.initial_stock,
                                  cfg.data.n_weeks, key=demo.key,
                                  opening=opening, policy=reconstruction)
    say(f"[S11] demo={demo.key}, I0={trajectory.initial_stock.value:.2f} "
        f"({trajectory.initial_stock.strategy_name}, opening={opening:.2f}, "
        f"A3+={trajectory.opening_stock_correction:.2f}), "
        f"sellout={trajectory.sellout_week}")

    # ---- S12 recommend and grade ---------------------------------------
    decision_week = int(cfg.spine.input_weeks)
    snapshot = trajectory.at(decision_week)
    local_index = candidates.index(demo)
    demand_slice = np.asarray(predictions[main_model][local_index], dtype=float)
    variability = float(error_scale.sigma_e)

    context = ReplenishmentContext(
        key=demo.key, decision_week=decision_week,
        forecast_origin_week=int(main_task.origin_week),
        inventory_at_decision_week=snapshot.reconstructed_stock,
        demand_slice=tuple(demand_slice.tolist()),
        lead_time_weeks=cfg.inventory.lead_time_weeks,
        order_cycle_weeks=cfg.replenish.order_cycle_weeks)
    actual_slice = y_test[local_index]
    weeks = [decision_week + 1 + i for i in range(len(demand_slice))]
    demo_forecast = {
        "weeks": weeks,
        "predicted": [float(v) for v in demand_slice],
        "actual": [float(v) for v in actual_slice],
        "labels": list(main_task.labels),
    }

    recommendation = recommend(context, seams.safety_stock, variability)
    baseline = recommend_baseline(context, target_weeks=2)
    risk = assess(snapshot.reconstructed_stock, demand_slice,
                  cfg.replenish.risk_high_weeks, cfg.replenish.risk_medium_weeks,
                  cfg.replenish.overstock_high_ratio,
                  cfg.replenish.overstock_medium_ratio)
    say(f"[S12] s={recommendation.reorder_point} S={recommendation.target_level} "
        f"qty={recommendation.recommended_qty} risk={risk.stockout_proxy}/{risk.overstock}")

    # ---- S13 publish ----------------------------------------------------
    quality = QualityReport(**quality_report(
        frame, cleaned.excluded, cleaned.duplicate_count, violations,
        outcome.policy_name, outcome.restock_date_max_state, outcome.excluded,
        negative_sales_policy=sign.policy_name,
        negative_cells=sign.negative_cells,
        negative_cells_by_week=sign.negative_cells_by_week,
        unit=RAW_UNIT, applied_scale_factor=None))
    split = SplitProvenance(assignment.cut_train_end, assignment.cut_validation_end,
                            assignment.counts, assignment.method,
                            preset=preset_provenance.as_dict())
    selection_record = decision.as_dict()
    selection_record.update({
        "candidate_count": len(model_candidates),
        "candidates": model_candidates,
        "test_reads": len(predictions),
        "fit_span": fit_span,
        "main_model_status": main_model_status,
        "seed": cfg.run.seed,
        "unit": RAW_UNIT,
        "sign_policy": sign.policy_name,
        "applied_scale_factor": None,
        "primary_metric": cfg.metrics.primary.value,
        "tolerance": cfg.model.qualification.tolerance,
        "mode": cfg.model.qualification.mode.value,
    })
    data_class = classify_dataset(raw)["data_class"]
    conditional = {
        "data_class": data_class,
        "b27_real_data": ("ready" if data_class == "real"
                          else "not evaluated - precondition absent"),
        "b29_secondary": ("ready" if preset_present
                          else "not evaluated - precondition absent"),
        "preset_files_present": bool(preset_present),
    }
    self_test = is_sample_dataset(raw)
    if self_test:
        print(SAMPLE_NOTICE)

    bundle = ResultBundle(
        effective_config=effective_config_dict(cfg),
        quality=quality,
        metrics=metrics,
        main_model=main_model,
        split=split,
        demonstration=trajectory,
        replenishment=recommendation,
        replenishment_baseline=baseline,
        risk=risk,
        channels=registry.all(),
        seams=seams.describe_all(),
        provisional=refs,
        available_models=seams.models.available(),
        self_test=self_test,
        run_id=f"run-{cfg.run.seed}",
        seed=cfg.run.seed,
        alt_task=alt_task_summary(alt_points),
        n_evaluated_series=metrics[main_model].n_evaluated_series,
        demo_forecast=demo_forecast,
        unit_provenance=UnitProvenance(
            unit=RAW_UNIT, applied_scale_factor=None,
            scale_factor_source=preset_provenance.factor_state,
            sign_policy=sign.policy_name, negative_cells=sign.negative_cells,
            register_version=register_revision),
        register_version=register_revision,
        secondary_comparison=None,
        feature_set=main_task.feature_set.as_dict()
        if main_task.feature_set else None,
        error_scale=error_scale.as_dict(),
        error_scale_by_model=error_scale_by_model,
        assumptions=assumption_records(),
        scenarios=scenarios,
        error_slices=error_slice_payload,
        reconstruction_audit=reconstruction_audit,
        selection=selection_record,
        conditional=conditional,
        main_model_status=main_model_status,
        main_model_note=decision.note,
    )
    if preset_provenance.used:
        bundle.secondary_comparison = _build_secondary(cfg, seams, main_model)
    bundle.validate_complete()
    bundle.artefact_dir = seams.artefacts.write(bundle, bundle.run_id)
    if bundle.artefact_dir:
        say(f"[S13] artefacts written to {bundle.artefact_dir}")
    else:
        say("[S13] artefact writer disabled (output.enabled=false)")

    ledger.assert_consumed()
    return bundle


def _build_secondary(cfg, seams, model_name):
    """Optional, labelled comparison on the benchmark partition (T11.7)."""
    factor = seams.preset.require_scale_factor()
    names = cfg.split.preset_files
    prepared = []
    for name in names:
        frame, _ = load_sales(cfg.data.raw_dir / name, cfg.data.n_weeks,
                              cfg.data.sales_week_prefix)
        cleaned = clean_sales(frame, cfg.data.n_weeks, cfg.data.sales_week_prefix,
                              policy=seams.sign_policy)
        indexed = attach_week_index(cleaned.frame, cfg.data.n_weeks)
        task = build_task(indexed, cfg.spine.input_weeks, cfg.spine.output_weeks,
                          cfg.data.n_weeks, origin_week=cfg.spine.input_weeks,
                          prefix=cfg.data.sales_week_prefix, spec=seams.features)
        prepared.append(task)
    comparison = compare_on_preset(
        prepared[0], prepared[1], factor, seams.metrics, model_name,
        seams.backend, cfg.run.seed, cfg.baseline.naive_reference)
    return comparison.as_dict()


def run_pipeline(cfg_path: str = "config/settings.yaml", root: str = ".",
                 verbose: bool | None = None) -> ResultBundle:
    cfg = load_config(cfg_path, root=root)
    return run(cfg, verbose=verbose)


if __name__ == "__main__":
    result = run_pipeline()
    print(result.to_dict())