"""Contract: every summary-shape detector method returns its window bounds.

A method the taxonomy classifies as ``"summary"``
(:data:`ts_shape.eventlog.taxonomy.REGISTRY`) returns ``start``, ``end`` and
``duration_seconds`` (docs/naming.md, "Output columns") -- on real data and
on empty input, with a numeric ``duration_seconds`` and ``end`` never before
``start``. Window summaries that label rows only with a legacy
``window_start`` column must therefore add the bounds (they keep
``window_start`` as a deprecated twin).

The harness builds the calls (:func:`harness.sample_results`). The summary
methods it cannot call -- they need a mapping or the output of another
method -- are pinned in :data:`UNCALLABLE`, so a method that silently drops
out of the check fails here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.eventlog.taxonomy import REGISTRY

from . import harness

BOUNDS = ("start", "end", "duration_seconds")

#: Summary methods the harness cannot call, with the reason it gives.
UNCALLABLE = {
    "ContinuousProcessAlignmentEvents.align_to_reference": "line_config mapping",
    "ContinuousProcessAlignmentEvents.alignment_quality": "line_config mapping",
    "ContinuousProcessAlignmentEvents.lag_profile": "line_config mapping",
    "CycleTimeEvents.cycle_time_trend": "needs a part_number",
    "EnergyConsumptionEvents.normalize": "static helper on a raw frame",
    "EnergyEfficiencyEvents.normalize": "static helper on a raw frame",
    "FlowConstraintEvents.flow_constraint_analytics": "roles mapping",
    "GoldenBatchDeviationEvents.compare": "needs a batch frame",
    "GoldenBatchDeviationEvents.phase_breakdown": "needs a batch frame",
    "LongDowntimeEvents.count_events_between_gaps": "needs a production frame",
    "MultiProcessTraceabilityEvents.lead_time": "processes mapping",
    "MultiProcessTraceabilityEvents.station_statistics": "processes mapping",
    "OperatorPerformanceEvents.operator_efficiency": "needs target_per_shift",
    "PeriodSummaryEvents.compare_periods": "needs two date ranges",
    "PeriodSummaryEvents.from_daily_data": "classmethod on daily data",
    "ReworkEvents.rework_cost": "rework_costs mapping",
    "ScrapEvents.scrap_cost": "material_costs mapping",
    "ShiftHandoverEvents.from_shift_data": "classmethod on shift data",
    "ShiftReportEvents.shift_targets": "targets mapping",
    "StartupDetectionEvents.assess_startup_quality": "needs startup events",
    "TargetEvents.compare_to_target": "targets mapping",
    "TargetEvents.target_achievement_summary": "needs daily_target",
    "TargetEvents.target_hit_rate": "needs daily_target",
}

SUMMARIES = sorted(
    f"{cls}.{method}"
    for (cls, method), rule in REGISTRY.items()
    if rule.shape == "summary"
)

#: Summaries that were static before round 11 (labeled by ``window_start``).
WINDOW_START_SUMMARIES = [
    "CalibrationValidationEvents.calibration_stability",
    "CalibrationValidationEvents.detect_recalibration_need",
    "EnvironmentalCompensationEvents.detect_correlation",
    "EnvironmentalCompensationEvents.detect_env_exceedance",
    "EnvironmentalCompensationEvents.sensitivity",
    "MeasurementUncertaintyEvents.combined_uncertainty",
    "MeasurementUncertaintyEvents.type_a_uncertainty",
    "SamplingResolutionEvents.detect_aliasing",
    "SamplingResolutionEvents.detect_quantization",
    "SamplingResolutionEvents.effective_resolution",
    "SamplingResolutionEvents.sampling_jitter",
    "TimeToThresholdEvents.time_to_threshold_windows",
]


def test_uncallable_list_is_current():
    stale = sorted(set(UNCALLABLE) - set(SUMMARIES))
    assert not stale, f"not summary methods (any more): {stale}"


def test_window_start_methods_are_summaries():
    assert set(WINDOW_START_SUMMARIES) <= set(SUMMARIES)


@pytest.mark.parametrize("key", SUMMARIES)
def test_summary_has_window_bounds(key):
    cls_name, method = key.split(".")
    status, plain, empty = harness.sample_results(cls_name, method)
    if status.startswith("n/a: optional dependency"):
        pytest.skip(status)
    if key in UNCALLABLE:
        assert status.startswith(
            "n/a"
        ), f"{key} runs in the harness now; remove it from UNCALLABLE"
        return
    assert status == "ok", f"{key}: {status}"
    for name, result in (("result", plain), ("empty result", empty)):
        assert isinstance(result, pd.DataFrame), f"{key}: {name} is not a DataFrame"
        missing = [c for c in BOUNDS if c not in result.columns]
        assert not missing, f"{key}: {name} lacks {missing}"
    if len(plain):
        assert pd.api.types.is_numeric_dtype(plain["duration_seconds"]), key
        start = pd.to_datetime(plain["start"])
        end = pd.to_datetime(plain["end"])
        assert not bool((end < start).any()), f"{key}: end before start"
