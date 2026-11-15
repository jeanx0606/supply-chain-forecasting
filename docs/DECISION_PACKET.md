# Decision Packet — provisional seams

- register_version: 3
- register_sha256: 42d3d09a510f26b558008c833bed081b375ba10e6f5b6594f12e0c7f081308b6
- This packet records options only. It records **no decision and no date**.
- The prototype cannot decide these; they belong to the named confirmer.

| REP | Decision | Current default | Options | Confirmer | Affected outputs | Status |
|---|---|---|---|---|---|---|
| REP-04 | Whether restock events are retained by lifecycle membership or by a hard calendar cutoff | lifecycle_membership, so late-2019 releases keep their 2020 shipments | lifecycle_membership; calendar_cutoff | Rue (Chapter 5) | inventory.entries, inventory.sellout_week, replenishment, risk | Provisional |
| REP-10 | Whether safety stock is implemented from the service level, or the service level is removed from scope | normal_approx (normal approximation from the configured service level) | normal_approx; none | Linda (Chapter 7) | replenishment.reorder_point, replenishment.target_level, replenishment.recommended_qty, risk.stockout_proxy, risk.overstock | Provisional |
| REP-17 | Whether negative weekly sales are clipped at zero or retained as returns | clip_at_zero - negative cells are counted first, then set to zero | clip_at_zero; retain_as_returns | Rue (Chapter 5) | quality.negative_cells, inventory.entries, replenishment.reorder_point, metrics | Provisional |
| REP-18 | How the main model is chosen among the declared candidates | min_primary_metric on validation; must_not_be_worse_than_naive; retained_default when none qualifies | min_primary_metric; must_not_be_worse_than_naive | strictly_better; retained_default | least_bad_provisional | fail_run | Linda (Chapter 7) | selection, metrics, replenishment.recommended_qty, risk | Provisional |
| REP-19 | Which feature families the closed matrix contains | input_window lags and discount lags, item/store input-window aggregates, price x discount, weather as supplied | input_window; item_mean_input / store_mean_input; price_discount_interaction true|false; as_supplied|off | Linda (Chapter 7) | feature_set, metrics | Provisional |

## What changes when a decision is made

### REP-04
- Configuration keys: truncation.restock_policy, truncation.restock_date_max
- Options: lifecycle_membership; calendar_cutoff
- Gates affected: B11, B2, B5
- Replacement: configuration only — set lifecycle_membership | calendar_cutoff; restock_date_max is applied or echoed inactive; call sites affected: 0

### REP-10
- Configuration keys: replenish.safety_stock_method, replenish.service_level, replenish.order_cycle_weeks
- Options: normal_approx; none
- Gates affected: B4, B5, B23
- Replacement: configuration only — normal_approx | none; none declares the service level out of scope; call sites affected: 0

### REP-17
- Configuration keys: cleaning.negative_sales_policy
- Options: clip_at_zero; retain_as_returns
- Gates affected: B23, B14
- Replacement: configuration only - clip_at_zero | retain_as_returns; call sites affected: 0

### REP-18
- Configuration keys: model.candidates, model.selection_rule, model.selection_metric, model.qualification.mode, model.qualification.tolerance, model.qualification.on_none
- Options: min_primary_metric; must_not_be_worse_than_naive | strictly_better; retained_default | least_bad_provisional | fail_run
- Gates affected: B25, B26
- Replacement: configuration only - rule, mode, tolerance or on_none; call sites affected: 0

### REP-19
- Configuration keys: features.lags, features.history_aggregates, features.price_discount_interaction, features.weather
- Options: input_window; item_mean_input / store_mean_input; price_discount_interaction true|false; as_supplied|off
- Gates affected: B26, B24
- Replacement: configuration or adapter - change the declared families; an undeclared column is an error; call sites affected: 0

## Exact update steps (performed only after a dated decision)

1. Set `status: Confirmed`, `confirmed_by` and `confirmed_on` in `config/replaceability.yaml`.
2. Change the affected default in `config/settings.yaml` if the decision differs.
3. Run `python -m codes.check_docs --fix` to regenerate R9.
4. Rerun the affected gates and the full suite.
5. Regenerate this packet: `python -m codes.decision_packet`.

<!-- register_sha256: 42d3d09a510f26b558008c833bed081b375ba10e6f5b6594f12e0c7f081308b6 -->
