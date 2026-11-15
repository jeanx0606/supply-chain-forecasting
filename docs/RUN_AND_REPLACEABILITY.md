# Run and Replaceability — VISUELLE 2.0 Prototype

**Version block**
- `register_version: 3`
- specification version: 1.4
- document revision: 2026-11-15
- prototype version: 1.0.0
- verified runtime: Python 3.13.14, pandas 3.0.6, numpy 2.2.6, scikit-learn 1.9.1,
  streamlit 1.66.0, altair 6.3.0, PyYAML 6.0.2, pytest 9.2.0
- `lightgbm` is **not installed** on the verified runtime; the default backend is
  `sklearn_hgb` and `lightgbm` remains an opt-in adapter (REP-14)

This document is the operator document required by spec §9. The authoritative
register is `config/replaceability.yaml`; section R9 is generated from it and must
not be hand-edited.

## R1. Purpose and status

The prototype answers one question: can weekly demand be forecast and converted
into a replenishment quantity and a risk grade from this dataset? It is a proof of
concept on external data, not a SHEIN system and not a production claim.

**Self-test notice.** `data/raw/` currently holds generated sample data. Every
figure produced from it is a **self-test artefact**, never a finding. A run on
sample data prints `[sample] SAMPLE DATA - SELF-TEST ONLY`.

**Non-claims.** No daily resolution; no measured stock-out; inventory is
reconstructed; prices are relative; lead time, service level and order cycle are
assumed; external channels are optional; no monetary figure is computed.

**Open REPs.** Every entry except REP-05 is `Provisional` at delivery, including
REP-18 (model selection) and REP-19 (feature set). Figures resting on a provisional
REP must carry its identifier. REP-05 (benchmark preset partition) is `Confirmed` by
Rue on 2026-11-15 with no default change.

## R2. Environment and installation

Python 3.11–3.13 is supported; the verified runtime is 3.13.14 (Windows 11).
Create and activate a virtual environment from the project root:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS use `source .venv/bin/activate`. Expected exit status: 0.
Dependency-audit outcome: `python -m codes.check_deps` reports every declared
package imported and every imported third-party package declared; `lightgbm` is the
one optional, undeclared backend. No environment variables and no network access.

## R3. Five-command quick start

Run from the project root, in order. Each line states the expected exit status and
one expected output line.

1. `pip install -r requirements.txt` — exit 0; `Successfully installed` (or
   `Requirement already satisfied`).
2. `python -m codes.sample_data` — exit 0; `[sample] SAMPLE DATA - SELF-TEST ONLY`.
3. `python -m codes.run_pipeline` — exit 0; `recommended qty`.
4. `python -m pytest` — exit 0; `236 passed`.
5. `streamlit run app.py` — server starts; `You can now view your Streamlit app`.

## R4. Sample data

`python -m codes.sample_data --items 200 --stores 20 --seed 42` writes five CSVs to
`data/raw/` plus `_sample_marker.json`: `sales.csv`, `restocks.csv`,
`price_discount_series.csv`, `vis2_gtrends_data.csv`, `vis2_weather_data.csv`.
Same seed ⇒ byte-identical files. Regeneration over existing files is refused unless
`--overwrite` is passed. Delete `_sample_marker.json` before using real data.

## R5. Real data onboarding

Place the five files in `data/raw/`, named as declared in `config/settings.yaml`:
`sales.csv`, `restocks.csv`, `price_discount_series.csv`,
`vis2_gtrends_data.csv`, `vis2_weather_data.csv`. Weekly headers are numeric
`0..11` and are renamed to `q1..q12`; `release_date` is required and parsed.

**Which files are optional.** `vis2_weather_data.csv` and
`vis2_gtrends_data.csv` are optional (a missing file is reported `skipped`, the run
continues). The benchmark pair `stfore_train.csv` / `stfore_test.csv` is **never
required**: the main experiments use `sales.csv` with the chronological 70/15/15
split by release date. The pair is the benchmark's own partition of the same
106,850 item-store pairs, at most a labelled secondary comparison.

**The constant 53.** The preset files hold the raw values divided by 53 for every
numeric column, including restock. A model trained on them therefore predicts in
units of 1/53 and every prediction must be multiplied by 53 before comparison with
raw quantities. This prototype never applies that factor on the main path: figures
are reported in raw sales units (`unit_provenance.applied_scale_factor: null`). A
scaled read without a declared `split.preset_scale_factor` aborts at S0 rather than
reporting silent 1/53 units.

**Healthy load.** A good quality report shows `n_rows > 0`, every file's mandatory
fields present, `monday_violations` at or near 0, duplicate keys counted rather than
dropped silently, and `negative_cells` reported with the active
`negative_sales_policy`.

## R6. Expected outputs

`output/<run_id>/` contains `bundle.json`, `quality.csv`, `metrics.csv`,
`inventory.csv`, `replenishment.json` and `run_meta.json`; `secondary_comparison.json`
appears only when the secondary path is enabled. The bundle carries `quality`,
`metrics`, `main_model`, `split` (including `preset.statement`), `demonstration`,
`replenishment`, `replenishment_baseline`, `risk`, `channels`, `seams`,
`provisional`, `available_models`, `self_test`, `unit_provenance`,
`register_version`.

The chapter figure pack is written to `reports/chapter_pack/<run>-<digest>/`
(`figures.json`, `figures.csv`, `summary.md`) and lies outside `output/` so a re-run
does not delete it. Every figure in it traces to a bundle field and carries its unit
and responsible REPs.

Panels: P1 Overview (item/store/coverage counters, negative-cell policy, unit and
preset captions); P2 Forecast (predicted-vs-actual chart, ten weeks); P3 Metrics
(per-model WAPE/MAE/RMSE); P4 Inventory (reconstructed trajectory, assumed I0,
sell-out week); P5 Replenishment (s, S, quantity, assumptions block); P6 Risk
(stock-out proxy and overstock grades). Sample figures are illustrative only.

## R7. Verification commands

```
python -m pytest                      # full suite, ~50 s, exit 0
python -m pytest tests/integration -q # end-to-end and failure paths
python -m codes.check_docs            # documentation contract
python -m codes.check_seams           # 17 markers vs the register
python -m codes.check_units           # no scale constant in business logic
python -m codes.run_alt_task          # optional 4-in/1-out task, exit 0
```

## R8. Troubleshooting matrix

| # | Symptom | Cause | Corrective command |
|---|---|---|---|
| 1 | `missing mandatory input 'sales.csv'` | no raw data | `python -m codes.sample_data` |
| 2 | `failed the field contract` | missing required column or too few week columns | inspect the named file, then rerun |
| 3 | `input(4) + output(10) > n_weeks(12)` | window arithmetic | restore `spine.input_weeks: 2` |
| 4 | `ratios must sum to 1.0 +/- 0.001` | split ratios edited | restore 0.70/0.15/0.15 |
| 5 | `need at least 3 distinct release dates` | tiny dataset | `python -m codes.sample_data --items 200 --overwrite` |
| 6 | `test set is empty` | ratios or cut dates too wide | adjust `split.*` |
| 7 | weather or trends `skipped` | optional file absent | expected; add the file only if needed |
| 8 | `backend 'lightgbm' unavailable` | optional wheel missing | set `model.backend: sklearn_hgb` |
| 9 | `port 8501 is already in use` | console already running | `streamlit run app.py --server.port 8502` |
| 10 | run exceeds the budget | oversized input | reduce `data.n_weeks` inputs; report the dominant stage |
| 11 | week attribution off by one at New Year | non-ISO week arithmetic | use `time.week_convention: iso8601` (default) |
| 12 | metric shown as `n/a` | zero denominator | expected; no division-by-zero figure is invented |
| 13 | no artefacts written | writer disabled | set `output.enabled: true` |
| 14 | provisional-figure warning on a number | REP not confirmed | read the REP id and confirmer in the caption |
| 15 | negative weekly values present | data property, not an error | see `quality.negative_cells`; switch `cleaning.negative_sales_policy` |
| 16 | figure moved after switching the sign policy | expected once; the count must not move | compare `quality.negative_cells` across runs |

Informational: `stfore_train.csv` / `stfore_test.csv` present but unused is the
normal state — `split.preset.statement` says so explicitly. A scaled read without a
declared factor aborts at S0; values from the preset files must be multiplied by 53.

## R9. Replaceable parts pending confirmation

Generated from `config/replaceability.yaml`. Status `Provisional` means the current
default is a runnable placeholder awaiting the named confirmer.

<!-- R9:BEGIN -->
| REP | Decision | Current default | Seam | Confirmer | Status |
|---|---|---|---|---|---|
| REP-01 | Where the chronological train/validation/test cut points fall | chronological with 70/15/15 ratios on distinct release dates | SEAM-SPLIT-CUT | Linda (Chapter 7) | Provisional |
| REP-02 | Which model is the reported main model and which is the baseline | gradient_boosting as main, seasonal_naive as baseline | SEAM-MODEL-SELECT | Linda (Chapter 7) | Provisional |
| REP-03 | Which metric is primary and which are supporting | wape_item_level primary; pooled WAPE, MAE, RMSE, store-weighted and small-store WAPE supporting | SEAM-METRIC-SET | Linda (Chapter 7) | Provisional |
| REP-04 | Whether restock events are retained by lifecycle membership or by a hard calendar cutoff | lifecycle_membership, so late-2019 releases keep their 2020 shipments | SEAM-RESTOCK-TRUNC | Rue (Chapter 5) | Provisional |
| REP-05 | Whether the preset train/test files replace the chronological split | false — preset files are not supplied by this dataset | SEAM-PRESET-FILES | Rue (Chapter 5) | Confirmed |
| REP-06 | Where and in what format run artefacts are persisted | json+csv under output/<run_id>/ | SEAM-OUTPUT-ARTEFACT | Ashley (Chapter 6) | Provisional |
| REP-07 | How the unobserved opening stock I0 is back-calculated | lead_time_demand (first lead-time demand times the safety factor) | SEAM-INITIAL-STOCK | team (Chapter 6 review) | Provisional |
| REP-08 | How locality-level weather is attached to an item-store row | pooled_representative (one aggregate per ISO week, record count preserved) | SEAM-WEATHER-JOIN | Rue (Chapter 5) | Provisional |
| REP-09 | Whether Google Trends is consumed as features or declared out of scope | deferred — file present, series not joined, status reported | SEAM-TRENDS | Ashley (Chapter 6) | Provisional |
| REP-10 | Whether safety stock is implemented from the service level, or the service level is removed from scope | normal_approx (normal approximation from the configured service level) | SEAM-SAFETY-STOCK | Linda (Chapter 7) | Provisional |
| REP-11 | Which item-store row is used for the replenishment demonstration | max_horizon_demand with ties broken by (external_code, retail) | SEAM-DEMO-SELECT | Ashley (Chapter 6) | Provisional |
| REP-12 | At which grain the forecast task is built | item_store — the only registered grain | SEAM-GRAIN | Linda (Chapter 7) | Provisional |
| REP-13 | Which week convention governs restock, weather and lifecycle arithmetic | iso8601 with Monday start, via date.fromisocalendar | SEAM-WEEK-CONVENTION | Rue (Chapter 5) | Provisional |
| REP-14 | Which boosting backend builds the horizon ensemble | sklearn_hgb — installed on the verified runtime; lightgbm is opt-in | SEAM-MODEL-BACKEND | Ashley (Chapter 6) | Provisional |
| REP-15 | Which observed week the naive baseline repeats | last_input — the last week of the input window, never a target week | SEAM-BASELINE-DEF | Linda (Chapter 7) | Provisional |
| REP-16 | Whether the alternative 4-in/1-out rolling task is exposed | enabled — eight rolling evaluation points q1..q4 to q12 | SEAM-ALT-TASK | Ashley (Chapter 6) | Provisional |
| REP-17 | Whether negative weekly sales are clipped at zero or retained as returns | clip_at_zero - negative cells are counted first, then set to zero | SEAM-NEGATIVE-SALES | Rue (Chapter 5) | Provisional |
| REP-18 | How the main model is chosen among the declared candidates | min_primary_metric on validation; must_not_be_worse_than_naive; retained_default when none qualifies | SEAM-MODEL-SELECTION | Linda (Chapter 7) | Provisional |
| REP-19 | Which feature families the closed matrix contains | input_window lags and discount lags, item/store input-window aggregates, price x discount, weather as supplied | SEAM-FEATURE-SET | Linda (Chapter 7) | Provisional |
<!-- R9:END -->

## R10. Replacement procedure

1. Find the seam in the table above; every seam has exactly one `SEAM-*` marker in
   one source module.
2. Configuration-only seam: change the key in `config/settings.yaml`. No call-site
   changes.
3. Adapter seam: implement the declared protocol in the owning module and register
   the new selector value in `src/seams/factory.py`.
4. Update `config/replaceability.yaml`: status, `confirmed_by`, `confirmed_on`.
5. Regenerate R9 with `python -m codes.check_docs --fix`.
6. Re-run `python -m pytest`, `codes.check_seams`, `codes.check_docs`,
   `codes.check_units`; the suite must stay green.

Recorded drills: `SEAM-SAFETY-STOCK: normal_approx -> none` and
`SEAM-MODEL-BACKEND: lightgbm -> sklearn_hgb` (reverse direction: the wheel is
unavailable). A third, configuration-only drill covers
`SEAM-NEGATIVE-SALES: clip_at_zero -> retain_as_returns`.

## R11. Known limitations and non-claims

- Weekly resolution only; no daily output exists in the data.
- No measured stock-out; the risk grade is an explicit proxy.
- Inventory is reconstructed from restock minus sales; I0 is assumed, so only the
  trajectory is observable.
- Prices are normalised; no monetary value is computed.
- Lead time, service level and order cycle are assumed, not measured.
- Sample data is not evidence; external channels are optional.
- **All reported quantities are in raw sales units.** No scale factor is applied on
  the main path, so a prediction produced from the scaled preset files must be
  multiplied by 53 before use.
- The negative-weekly-sales policy is a declared interpretation, not a data
  correction and not imputation.
- The benchmark preset partition is not the main experimental basis.

## R12. Confirmation and change log

| Date | REP | Decision | Confirmed by | Changes | Gates rerun |
|---|---|---|---|---|---|
| 2026-11-15 | REP-05 | Preset partition is optional; main path stays `sales.csv` + 70/15/15 | Rue | status → `Confirmed`; default unchanged; runbook R5/R9 updated | full suite, check_docs, check_seams |
| 2026-11-15 | REP-18 / REP-19 | new seams: selection rule and feature set | pending Linda | register v3; R9 regenerated | pending |
| — | REP-01…REP-19 (except REP-05) | not confirmed at delivery | Linda / Rue / team | — | — |

When a REP becomes `Confirmed`, update the register and this log in the same
revision, regenerate R9, and rerun the gates that the REP affects.

## R13. Additional operational commands

These four commands support onboarding and reporting; none of them changes a
figure.

| Command | Expected result |
|---|---|
| `python -m codes.preflight` | exit 0; `data class`, per-file contract status, counts, coverage, negative cells, readiness line |
| `python -m codes.decision_packet` | exit 0; `docs/DECISION_PACKET.md` regenerated from the register |
| `python -m codes.chapter_report` | exit 0; `reports/chapter_pack/<run>-<digest>/` written from the newest bundle |
| `python -m codes.check_units` | exit 0; no scale constant in business logic |

`python -m codes.preflight --json` emits the same report as JSON. Preflight is
read-only: it never imports the pipeline and never fits a model.
