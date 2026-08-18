#!/usr/bin/env python3
"""
Cross-facility production reporting with ``ts_shape.reporting``.

Three plants on three continents, one consolidated report. The script is
self-contained: it synthesises raw UTC readings, turns them into per-site local
production days, and then walks the whole reporting layer:

1. Facility master data and its validation
2. Local production days and shifts (DST-correct, per site)
3. Coverage -- did every site actually report?
4. Consolidation with correct weighting (global / region / site)
5. Benchmarking and opportunity sizing
6. Variance: who moved the number, and was it rate or mix?
7. Currency normalisation
8. The management scorecard
"""

import os
import sys

import numpy as np
import pandas as pd

# Add parent directory to path to import ts_shape
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ts_shape.reporting import (  # noqa: E402
    Facility,
    FacilityBenchmark,
    FacilityCalendar,
    FacilityRegistry,
    FleetRollup,
    GlobalScorecard,
    KpiTarget,
    MetricNormalizer,
    MetricSpec,
    ReportingCoverage,
    VarianceBridge,
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def build_registry() -> FacilityRegistry:
    """Three sites: three timezones, three currencies, two shift patterns."""
    return FacilityRegistry(
        [
            Facility(
                "DE-STR",
                name="Stuttgart",
                timezone="Europe/Berlin",
                region="EMEA",
                country="DE",
                currency="EUR",
                capacity_per_day=1200.0,
            ),
            Facility(
                "US-DTW",
                name="Detroit",
                timezone="America/Detroit",
                region="AMER",
                country="US",
                currency="USD",
                capacity_per_day=900.0,
                shifts={"day": ("07:00", "19:00"), "night": ("19:00", "07:00")},
            ),
            Facility(
                "CN-SHA",
                name="Shanghai",
                timezone="Asia/Shanghai",
                region="APAC",
                country="CN",
                currency="CNY",
                capacity_per_day=2500.0,
            ),
        ]
    )


def synthesise_readings() -> pd.DataFrame:
    """Hourly UTC readings for two weeks, three sites, one silent day."""
    rng = np.random.default_rng(42)
    stamps = pd.date_range("2026-03-23", periods=24 * 14, freq="1h", tz="UTC")

    profiles = {
        # site: (parts/hour, mean OEE, scrap rate %, cost per scrap part)
        "DE-STR": (48, 72.0, 2.0, 12.5),
        "US-DTW": (34, 61.0, 4.5, 15.0),
        "CN-SHA": (96, 84.0, 3.0, 90.0),
    }

    frames = []
    for site, (rate, oee, scrap, unit_cost) in profiles.items():
        parts = rng.poisson(rate, len(stamps)).astype(float)
        frame = pd.DataFrame(
            {
                "facility": site,
                "systime": stamps,
                "parts_produced": parts,
                "good": np.round(parts * (1 - scrap / 100.0)),
                "planned_minutes": 60.0,
                "oee_pct": np.clip(rng.normal(oee, 6.0, len(stamps)), 0, 100),
                "scrap_cost": np.round(parts * scrap / 100.0 * unit_cost, 2),
            }
        )
        frames.append(frame)

    readings = pd.concat(frames, ignore_index=True)
    # Detroit's historian was down for a day: the rows simply are not there.
    outage = (readings["facility"] == "US-DTW") & (
        readings["systime"].dt.date == pd.Timestamp("2026-03-30").date()
    )
    return readings[~outage].reset_index(drop=True)


def main() -> None:
    registry = build_registry()
    readings = synthesise_readings()

    print("=" * 78)
    print("1. FACILITY MASTER DATA")
    print("=" * 78)
    print(registry.to_frame().to_string(index=False))
    issues = registry.validate()
    print(f"\nMaster-data issues: {len(issues)}")
    if not issues.empty:
        print(issues.to_string(index=False))

    print("\n" + "=" * 78)
    print("2. LOCAL PRODUCTION DAYS AND SHIFTS")
    print("=" * 78)
    calendar = FacilityCalendar(registry)
    labelled = calendar.assign_shifts(readings)
    sample = labelled.groupby("facility").head(1)[
        ["facility", "systime", "local_time", "utc_offset_hours", "shift", "production_date"]
    ]
    print("The same UTC instant, seen from each plant:")
    print(sample.to_string(index=False))

    # 29 March 2026 is the European spring-forward: a 23-hour local day.
    grid = calendar.period_grid("2026-03-28", "2026-03-30", facilities=["DE-STR"])
    grid = grid.assign(
        hours=(grid["period_end_utc"] - grid["period_start_utc"]).dt.total_seconds()
        / 3600.0
    )
    print("\nDST-correct local day lengths in Stuttgart:")
    print(grid[["period", "period_start_utc", "period_end_utc", "hours"]].to_string(index=False))

    # Per-site, per-day KPIs -- the grain the reporting layer consumes.
    site_kpis = (
        labelled.groupby(["facility", "production_date"], as_index=False)
        .agg(
            parts_produced=("parts_produced", "sum"),
            good=("good", "sum"),
            planned_minutes=("planned_minutes", "sum"),
            oee_pct=("oee_pct", "mean"),
            scrap_cost=("scrap_cost", "sum"),
        )
        .rename(columns={"production_date": "period"})
    )
    site_kpis["period"] = pd.to_datetime(site_kpis["period"])
    site_kpis["scrap_rate_pct"] = (
        (site_kpis["parts_produced"] - site_kpis["good"])
        / site_kpis["parts_produced"]
        * 100.0
    )

    print("\n" + "=" * 78)
    print("3. COVERAGE -- HOW MANY PLANTS ARE BEHIND THE NUMBER?")
    print("=" * 78)
    expected = calendar.period_grid("2026-03-24", "2026-04-04")
    coverage = ReportingCoverage(site_kpis, value_columns=["parts_produced", "oee_pct"])
    print(coverage.summary(expected).to_string(index=False))
    print("\nPer facility:")
    print(coverage.completeness(expected).to_string(index=False))
    gaps = coverage.missing(expected)
    print(f"\nMissing facility-days: {len(gaps)} (first few)")
    print(gaps.head().to_string(index=False))

    print("\n" + "=" * 78)
    print("4. CONSOLIDATION -- GLOBAL, REGION, SITE")
    print("=" * 78)
    specs = [
        MetricSpec.sum("parts_produced"),
        MetricSpec.ratio(
            "quality_pct", numerator="good", denominator="parts_produced", scale=100.0
        ),
        MetricSpec.weighted("oee_pct", weight="planned_minutes"),
    ]
    rollup = FleetRollup(site_kpis, registry=registry)
    print(rollup.consolidate(specs).to_string(index=False))

    naive = site_kpis["oee_pct"].mean()
    weighted = rollup.total(specs).loc[0, "oee_pct"]
    print(
        f"\nFleet OEE: {weighted:.2f} % weighted by planned minutes "
        f"vs {naive:.2f} % as a naive mean of site-days."
    )

    print("\nPareto of output by site:")
    print(rollup.contribution("parts_produced").to_string(index=False))

    print("\n" + "=" * 78)
    print("5. BENCHMARKING AND OPPORTUNITY SIZING")
    print("=" * 78)
    bench = FacilityBenchmark(site_kpis, period_column="period")
    print(bench.rank("oee_pct").to_string(index=False))
    print("\nGap to the top quartile, in planned minutes of output:")
    print(
        bench.opportunity("oee_pct", exposure_column="planned_minutes").to_string(
            index=False
        )
    )
    print("\nConsistency over the two weeks:")
    print(bench.stability("oee_pct").to_string(index=False))

    print("\n" + "=" * 78)
    print("6. VARIANCE -- WHO MOVED IT, AND WAS IT RATE OR MIX?")
    print("=" * 78)
    weekly = site_kpis.assign(
        week=site_kpis["period"].dt.to_period("W").dt.start_time
    )
    bridge = VarianceBridge(weekly, period_column="week")
    # Compare the two *complete* weeks. Local production days spill a few rows
    # into the partial weeks either side, and a bridge between partial weeks
    # compares different amounts of calendar rather than performance.
    base_week = pd.Timestamp("2026-03-23")
    compare_week = pd.Timestamp("2026-03-30")
    print(f"Contribution to the change in output, {base_week.date()} -> {compare_week.date()}:")
    print(
        bridge.facility_bridge(
            "parts_produced", base=base_week, compare=compare_week
        ).to_string(index=False)
    )
    print("\nWeighted fleet OEE: rate effect vs mix effect:")
    print(
        bridge.rate_bridge(
            "oee_pct",
            weight_column="planned_minutes",
            base=base_week,
            compare=compare_week,
        )[
            [
                "facility",
                "base_rate",
                "compare_rate",
                "rate_effect",
                "mix_effect",
                "interaction",
                "total_effect",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("7. ONE REPORTING CURRENCY")
    print("=" * 78)
    per_site = rollup.by_facility(
        [MetricSpec.sum("scrap_cost"), MetricSpec.sum("parts_produced")]
    )
    converted = MetricNormalizer(per_site, registry=registry).convert_currency(
        ["scrap_cost"],
        to_currency="EUR",
        rates={"EUR": 1.0, "USD": 0.92, "CNY": 0.13},
    )
    comparable = MetricNormalizer(converted, registry=registry).per_unit(
        ["scrap_cost_EUR"], denominator="parts_produced"
    )
    print(
        comparable[
            ["facility", "scrap_cost", "fx_rate", "scrap_cost_EUR", "scrap_cost_EUR_per_unit"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("8. MANAGEMENT SCORECARD")
    print("=" * 78)
    targets = [
        KpiTarget(
            "oee_pct",
            75.0,
            spec=MetricSpec.weighted("oee_pct", weight="planned_minutes"),
        ),
        KpiTarget("scrap_rate_pct", 2.5, higher_is_better=False),
        KpiTarget(
            "parts_produced",
            12000.0,
            spec=MetricSpec.sum("parts_produced"),
            label="output",
        ),
    ]
    board = GlobalScorecard(site_kpis, period_column="period", registry=registry)
    print("Fleet:")
    print(board.fleet_summary(targets).to_string(index=False))
    print("\nPer facility:")
    print(board.facility_summary(targets).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
