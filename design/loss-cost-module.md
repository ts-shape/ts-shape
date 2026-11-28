# Design: loss cost module (`ts_shape.cost`)

Status: proposal, not implemented. Planned on 28 November 2026.

## 1. Why

Every ts-shape detector reports losses in minutes, parts or kWh. Continuous
improvement teams rank and approve work in money: "micro-stops on press 3
cost 41 000 € last quarter" gets a project funded, "312 minutes of
micro-stops" does not. Today the library has only two narrow money methods,
`ScrapEvents.scrap_cost` and `ReworkEvents.rework_cost`. Each takes a plain
dict, and both cost a part that has no price at **0**: `scrap_cost` warns
about it, `rework_cost` does not even warn. That breaks the library's own
rule against placeholder values ([naming.md](../docs/naming.md#errors-and-warnings)).

The goal is one module that:

1. holds the plant's money master data (machine rates, labour, part costs,
   prices and margins, energy tariffs, CO₂ price) in one validated model;
2. prices the results of the existing detectors in a single cost ledger,
   without double counting;
3. answers the questions a plant actually asks: what each loss costs per
   shift, machine, part and reason; what one OEE point is worth; what a
   planned improvement would save per year.

### Non-goals

- No accounting system, no booking to an ERP, no tax handling.
- No currency conversion: a model has exactly one currency.
- No new detection: the module only prices what detectors already found.
- No `Decimal` arithmetic. Costs are float estimates, like every other KPI
  in ts-shape, and rounding happens in reports only.

## 2. Where it sits

A new layer between Detect and Reports, with its own package (like
`ts_shape.eventlog`):

```
Load → Transform → Detect ─┬─ event table ──→ Event log
                           └─ loss tables ──→ Cost (ts_shape.cost) ──→ cost ledger ──→ reports
                                                   ▲
                                         CostModel (master data)
```

| Name | Kind | Job |
|---|---|---|
| `CostModel` | master data | Holds and validates the money inputs: machines, labour groups, parts, changeovers, energy tariffs, global settings. |
| `LossCosting` | transform (role noun, per [naming.md](../docs/naming.md)) | Turns detector outputs into ledger rows, and aggregates the ledger. |
| `cost_workbook_template()` | helper | Writes an empty `.xlsx` with every sheet and column, for plants to fill in. |

## 3. Money inputs (the `CostModel`)

Each table is a DataFrame. A model loads from dicts, DataFrames, a YAML/JSON
file, or an Excel workbook with one sheet per table (read with the existing
standard-library `read_workbook`, so there is no new dependency).

Conventions that apply to every table:

- Every rate row may carry `valid_from` and `valid_to` (both optional). That
  is how a wage increase in April or a new energy contract in January is
  handled: a loss spanning the change is split and each part priced at its
  own rate.
- Money columns are in the model's single `currency`. Rate columns carry
  their basis in the name: `_per_hour`, `_per_part`, `_per_kwh`, and so on.
- A value left blank means "unknown", never 0. Section 6 covers what that
  does to the result.

### 3.1 `settings` (one row)

| Column | Required | Meaning |
|---|---|---|
| `currency` | yes | ISO 4217 code, for example `EUR`. It is written into every ledger row. |
| `timezone` | yes | Plant time zone. Tariff windows and validity dates are read in it. |
| `valuation_basis` | no, default `cost` | `cost`, `margin` or `both`. See section 4.1. |
| `include_overhead` | no, default `true` | Whether time losses carry the overhead rate. |
| `planned_hours_per_year` | no | Used to annualize results and to value one OEE point. |

### 3.2 `machines` (one row per machine, line or station)

| Column | Required | Meaning |
|---|---|---|
| `machine_id` | yes | Key, for example `press_03`. |
| `machine_rate_per_hour` | yes | Machine hour rate: depreciation, interest, space, maintenance, insurance. |
| `labour_group` | no | Key into `labour`. |
| `crew_size` | no, default 1 | Operators tied to the machine while it is stopped. |
| `overhead_rate_per_hour` | no | Fixed overhead allocated per machine hour. |
| `ideal_cycle_seconds` | no | Converts lost time into lost parts. Needed for the margin basis and for speed loss in parts. |
| `parts_per_cycle` | no, default 1 | Multi-cavity tools and similar. |
| `is_constraint` | no, default `false` | Bottleneck in a sold-out market. Lost time here is lost sales (margin basis). |
| `cost_center` | no | Copied into the ledger so results can be grouped by it. |
| `signal_uuids` | no | List of uuids that belong to this machine. Detector rows are mapped to a machine through it. |

### 3.3 `labour` (one row per labour group)

| Column | Required | Meaning |
|---|---|---|
| `labour_group` | yes | Key, for example `operator`, `setter`, `technician`. |
| `rate_per_hour` | yes | Fully loaded rate: wage plus social charges. |
| `shift_premium_pct` | no | Per shift name, as `shift_premium_pct.<shift>` columns, for example night shift +25 %. |
| `overtime_premium_pct` | no | Applied when a loss is flagged as recovered in overtime. |

### 3.4 `parts` (one row per part number)

| Column | Required | Meaning |
|---|---|---|
| `part_number` | yes | Key. Matched with the same text normalization the detectors use (`id_text`). |
| `material_cost_per_part` | for scrap | Raw material and purchased components. |
| `conversion_cost_per_part` | no | Standard value added up to the machine that scraps the part. If it is missing, it is derived as cycle time × hourly rate (`derived` is recorded in the ledger). |
| `sales_price_per_part` | for margin | Net sales price. |
| `contribution_margin_per_part` | for margin | Overrides price minus variable cost when given. |
| `salvage_value_per_part` | no | Money recovered from scrap: recycling, regrind. It is subtracted. |
| `disposal_cost_per_part` | no | Cost of getting rid of scrap. It is added. |
| `rework_minutes_per_part` | for rework | Rework effort per part. |
| `rework_labour_group` | no | Who reworks. Defaults to the machine's labour group. |
| `rework_material_cost_per_part` | no | Extra material used in rework. |
| `rework_cost_per_part` | no | Flat override. Replaces the three columns above. |

### 3.5 `changeovers` (optional, one row per product transition)

| Column | Required | Meaning |
|---|---|---|
| `machine_id`, `from_product`, `to_product` | keys | `*` matches any product. |
| `material_cost_per_changeover` | no | Purge material, first-off parts, consumables. |
| `crew_size` | no | Setters on top of the machine crew during setup. |
| `labour_group` | no | Labour group of those setters. |

### 3.6 `energy` (one row per medium and tariff window)

| Column | Required | Meaning |
|---|---|---|
| `medium` | yes | `electricity`, `gas`, `compressed_air`, `steam`, `water`, ... |
| `unit` | yes | `kwh`, `m3`, `t`. It must match the detector output. |
| `price_per_unit` | yes | Energy price. |
| `weekdays`, `time_from`, `time_to` | no | Time-of-use window in plant local time, for example peak 06:00–22:00 Mon–Fri. Rows without a window are the default price. |
| `demand_charge_per_kw` | no | Price per kW of peak demand above the contracted level. |
| `co2_price_per_t` | no | CO₂ certificate price. Emissions come from `CarbonIntensityEvents`. |

### 3.7 `repairs` (optional, phase 2)

| Column | Required | Meaning |
|---|---|---|
| `machine_id` | yes | Key. |
| `technician_labour_group` | no | Who repairs. |
| `spare_parts_cost_per_failure` | no | Average spare parts cost per failure. |
| `external_service_cost_per_failure` | no | Average cost of outside service per failure. |

## 4. How each loss is priced

Notation: *h* = lost hours, *n* = parts, *C_cap* = capacity cost per hour =
`machine_rate_per_hour` + `crew_size` × labour rate (with any shift premium)
+ `overhead_rate_per_hour` (when overhead is included).

| Loss type | Typical source (existing detector) | `cost` basis | Extra under `margin` basis (constraint machines only) | Category |
|---|---|---|---|---|
| `downtime` | `MicroStopEvents`/`LongDowntimeEvents` intervals, `ReliabilityEvents.failure_events` | *h* × *C_cap* | lost parts = *h* × 3600 / ideal cycle × parts per cycle, × margin | availability |
| `micro_stop` | `MicroStopEvents.detect_micro_stops` | *h* × *C_cap* | same | performance |
| `speed_loss` | `PerformanceLossEvents.slow_periods` (`loss_seconds`) | *h* × *C_cap* | same | performance |
| `setup` / `changeover` | `SetupTimeEvents.setup_durations`, `ChangeoverEvents.changeover_window` | *h* × (*C_cap* + setter crew × rate) + material per changeover | same | availability |
| `scrap` | `ScrapEvents`, `QualityEvents.quality_by_part`, `ScrapProfile` | *n* × (material + conversion + disposal − salvage) | lost margin when the part must be made again on a constraint | quality (internal failure) |
| `rework` | `ReworkEvents` | *n* × (rework minutes / 60 × labour rate + rework material), or the flat override | none | quality (internal failure) |
| `idle_energy` | `IdleEnergyDetectionEvents.idle_energy_by_window` | energy × tariff for its time window | none | energy |
| `energy` | `EnergyConsumptionEvents.consumption_by_window` | energy × tariff for its time window | none | energy |
| `peak_demand` (phase 2) | `EnergyConsumptionEvents.peak_demand_detection` | kW above the limit × demand charge | none | energy |
| `emissions` (phase 2) | `CarbonIntensityEvents.emissions_by_window` | kg CO₂e / 1000 × CO₂ price | none | energy |
| `repair` (phase 2) | `ReliabilityEvents.failure_events` | repair hours × technician rate + spare parts + external service | none | maintenance |

### 4.1 Valuation basis: what a lost hour is worth

This is the most important decision in the module, and the one most often
done wrong in spreadsheets.

- **`cost`** (default): a lost hour costs the capacity that was paid for and
  not used (machine, crew, overhead). This is right for any machine that is
  not the bottleneck, because the lost output can be made up later.
- **`margin`**: on a constraint in a sold-out market, a lost hour is lost
  sales. The cost is the contribution margin of the parts not made. Fixed
  machine cost is *not* added on top, because it is paid either way, so
  adding it would count the loss twice. Directly incurred costs (scrap
  material, rework labour, energy) are still added.
- **`both`**: both totals are reported in separate columns
  (`total_cost_capacity`, `total_cost_margin`), and `total_cost` follows the
  basis set for each machine.

The basis applies per machine: `margin` only takes effect where
`is_constraint` is true, so one plant-wide ledger is correct for both kinds
of machine.

### 4.2 Attribution: which machine and which part

- **Machine:** an explicit `machine_id=` argument, or the row's
  `source_uuid` looked up in `machines.signal_uuids`.
- **Part:** an explicit `part_number=` argument, or a column already in the
  detector output (`part_number`, `product`, `to_product`), or, for
  interval losses, the value of a product signal at the loss start
  (`part_signal=` a frame plus `part_uuid=`, using the existing state
  interval utilities). A loss without a part is still priced with the
  machine-level rates; only part-dependent components (margin, scrap
  material) are left unknown.

## 5. The cost ledger (output)

Every pricing method returns rows of one schema, in the **interval** shape
from [naming.md](../docs/naming.md#output-columns), so ledgers from
different sources can be concatenated, filtered and turned into event logs.

| Column | Content |
|---|---|
| `start`, `end`, `duration_seconds`, `uuid`, `source_uuid` | Interval shape. `uuid` is `cost:<loss_type>`, `source_uuid` is the detector's signal. |
| `date`, `shift` | Calendar labels, when shift definitions are given. |
| `machine_id`, `cost_center`, `part_number`, `reason` | Attribution. `reason` is filled when the source has one. |
| `loss_category` | `availability`, `performance`, `quality`, `energy`, `maintenance`. |
| `loss_type` | One of the loss types in section 4. |
| `coq_category` | Cost-of-quality bucket for quality losses (`internal_failure`), otherwise empty. |
| `quantity`, `quantity_unit` | The basis: seconds, parts, kWh, kgCO2e. This follows the rule that every KPI carries its basis. |
| `machine_cost`, `labour_cost`, `overhead_cost`, `material_cost`, `energy_cost`, `emission_cost`, `other_cost`, `lost_margin` | Components. |
| `total_cost` | Sum of the components that count under the basis in use. |
| `valuation_basis` | `cost` or `margin`, per row. |
| `currency` | From settings. |
| `rate_source` | How the rate was found: `model`, `derived`, `argument`. |
| `cost_complete` | `False` when a component needed for this row had no rate. |

Naming rule to add to `naming.md`: money columns end in `_cost` or
`_margin`, and the currency is stated in the `currency` column. This follows
the "unit in the column name" rule, but the actual currency is
configuration, so it goes in a column rather than in the name.

## 6. Correctness rules

These follow the existing library rules, applied to money:

1. **Unknown is not zero.** A missing rate makes that component `NaN`, sets
   `cost_complete=False`, and emits one `DataQualityWarning` that lists the
   missing keys (machines, parts, media). `errors="raise"` turns it into an
   error. `total_cost` is the sum of the known components, so a report shows
   a lower bound and flags it as incomplete, instead of silently showing 0.
2. **No double counting.** `LossCosting.ledger([...])` merges several sources
   and resolves time losses that overlap on the same machine. Overlapping
   time is priced once, by precedence: `downtime` > `setup`/`changeover` >
   `micro_stop` > `speed_loss`. The precedence is configurable. Removed
   overlap is reported in `overlap_seconds` so it can be audited. Quality and
   energy losses never overlap with time losses, because they price
   different resources.
3. **Rates change over time.** Intervals are split at `valid_from`/`valid_to`
   and at tariff window bounds (`split_intervals` in
   `utils/time_accounting.py`), so each part is priced at the rate in force.
4. **Time zones.** Tariff windows and validity dates are plant local time.
   A tz-naive detector output is read in the model's `timezone`.
5. **Full precision.** Results are never rounded.
6. **Empty in, empty out.** An empty result keeps every ledger column.
7. **Shift-only inputs.** Outputs with only `date` and `shift` and no
   `start`/`end` (for example `DowntimeEvents.downtime_by_shift`) are
   accepted when `shift_definitions` is passed, which rebuilds the window
   bounds. Otherwise the call raises an error saying which interval method
   to use instead.

## 7. Reports on the ledger

All are methods of `LossCosting` and return DataFrames:

| Method | Answers |
|---|---|
| `cost_by_window(ledger, window="1D", by=[...])` | Cost per day, week or month, optionally per machine, loss type or part. |
| `cost_by_shift(ledger, shift_definitions)` | Cost per shift, with the shift premium already applied. |
| `cost_pareto(ledger, by="reason")` | Money Pareto (reuses `ParetoEvents.rank_by_sum`). |
| `cost_of_poor_quality(ledger)` | Quality losses grouped by cost-of-quality bucket. |
| `oee_point_value(machine_id)` | What one OEE percentage point is worth per year, from planned hours and rates (margin basis on constraints). |
| `savings_scenario(ledger, reductions={"micro_stop": 0.3, ...})` | Yearly savings if a loss type shrinks by the given share. This is the business case for a CI project. |
| `rate_audit()` | Every rate used, with validity and source, so a controller can check it (the pattern of `CarbonIntensityEvents.emission_factor_audit`). |

## 8. API sketch

```python
from ts_shape.cost import CostModel, LossCosting

model = CostModel.from_workbook("cost_master.xlsx")  # or from_dict / from_yaml
costing = LossCosting(model, shift_definitions=shifts)

ledger = costing.ledger([
    costing.time_loss_cost(micro_stops, loss_type="micro_stop", machine_id="press_03",
                           part_signal=df, part_uuid="press_03:part_number"),
    costing.time_loss_cost(setups, loss_type="setup", machine_id="press_03"),
    costing.speed_loss_cost(slow_periods, machine_id="press_03"),
    costing.scrap_cost(scrap_by_part, machine_id="press_03"),
    costing.rework_cost(rework_by_part, machine_id="press_03"),
    costing.energy_cost(idle_energy, medium="electricity", loss_type="idle_energy"),
])

costing.cost_pareto(ledger, by="loss_type")
costing.savings_scenario(ledger, reductions={"micro_stop": 0.3})
```

## 9. Integration with the rest of the library

- **Existing money methods.** `ScrapEvents.scrap_cost` and
  `ReworkEvents.rework_cost` keep their signatures but delegate to the new
  module. A part without a price becomes `NaN` with a warning instead of 0.
  This is a behavior fix and gets a `CHANGES.txt` entry. A `cost_model=`
  argument is added to both, and the bare dicts are deprecated per the
  [deprecation policy](../docs/naming.md#deprecation-policy).
- **Event log.** Ledger rows convert to events. A `ts_shape:cost` attribute
  and the `cost.*` taxonomy entries replace the two existing
  `production.*.cost` entries.
- **Pipeline.** `LossCosting` methods work as pipeline steps.
- **Period summaries.** Add `cost` and `margin` to `_SUM_TOKENS` in
  `period_summary.py`, so that cost columns are summed. Today they would be
  averaged.
- **Catalog and docs.** `list_detectors("cost")`, a guide page `guides/cost.md`
  with a worked plant example, the module reference, and a landing page
  example ("What did micro-stops cost us last month?").

## 10. Tests

- **Reference tests** with hand-computed cases: one shift with known stops,
  rates and tariffs, priced to the cent. These cover a rate change in
  mid-interval, a peak/off-peak split, and margin basis versus cost basis on
  a constraint machine.
- **Overlap tests:** a micro-stop inside a downtime interval is priced once.
- **Missing-rate tests:** `NaN` plus a warning, and `errors="raise"`.
- **Invariant harness:** empty, tz-aware, nullable and pyarrow inputs, the
  same as every detector.
- **Property tests (hypothesis):** cost scales linearly with rates and
  durations; splitting an interval never changes its total.
- **Workbook round trip:** the template writes, fills, loads and validates.

## 11. Phases

**Phase 1: core (one release)**
1. `CostModel`: `settings`, `machines`, `labour`, `parts`, `energy` (flat
   price), with validation, `from_dict`, `from_frames`, `from_workbook`, the
   workbook template and `rate_audit`.
2. `LossCosting`: `time_loss_cost`, `speed_loss_cost`, `scrap_cost`,
   `rework_cost`, `energy_cost`, `ledger` with overlap resolution.
3. Reports: `cost_by_window`, `cost_by_shift`, `cost_pareto`.
4. The missing-rate rules, reference tests, a guide page and a
   `CHANGES.txt` entry.

**Phase 2: realism**
5. Validity periods and time-of-use tariffs (interval splitting).
6. Part attribution from a product signal.
7. Margin basis and `is_constraint`.
8. Changeover matrix, shift and overtime premiums.
9. Peak demand charges, CO₂ cost, repair cost (`repairs` table).

**Phase 3: decisions and integration**
10. `oee_point_value`, `savings_scenario`, `cost_of_poor_quality`.
11. `ScrapEvents.scrap_cost` and `ReworkEvents.rework_cost` delegate to the
    new module (the zero-cost fix), with deprecations.
12. Event log attribute and taxonomy, pipeline steps, a landing page example.

## 12. Open decisions

1. **Default valuation basis.** The proposal is `cost` by default, with
   `margin` per constraint machine. The alternative is to always report both.
2. **Package name.** The proposal is `ts_shape.cost`. The alternative is
   `ts_shape.events.finance`, but the module does not detect events from
   signals, so a separate layer fits better.
3. **Master data format.** The proposal is the Excel template as the main
   path, with dict and YAML for code. Is a database or API loader for cost
   data needed, for example from SAP CO?
4. **Fix of the existing zero-cost behavior.** The proposal is to fix it in
   phase 1 rather than wait for phase 3, because it is a correctness bug
   today.
