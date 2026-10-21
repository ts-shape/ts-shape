"""Chunked processing: run a detector slice by slice on data larger than memory.

:func:`run_chunked` cuts the data into time slices (one day by default), runs
the detector on one slice at a time and stitches the slices together, so
memory holds about one slice while the result equals the in-memory run::

    Target(full_df, **init_kwargs).method(**method_kwargs)

(up to float rounding). It works for the methods in :data:`SUPPORT`, and for
those in :data:`CONDITIONAL` when the call passes the argument named there;
every other method raises :class:`~ts_shape.errors.NotChunkableError`.

``chunk`` is a fixed length (``"1D"``, ``"12h"``) or a calendar period:
``"W"`` (weeks from Monday 00:00), ``"W-<DAY>"`` (weeks ending on
``<DAY>``) or ``"MS"`` (calendar months), cut at local midnight. Weekly and
monthly summaries need the matching calendar chunk.

How the slices are stitched
---------------------------

Each slice is run on a *context* frame: the slice's own rows, plus for every
signal the last reading before the context (a held state starts where it
started) and the first reading at or after the slice end (the last interval
of the slice closes where the in-memory run closes it). Rolling methods also
get ``overlap`` of history before the slice.

What is kept of a slice's result depends on the shape of the method's output
(:data:`ts_shape.eventlog.taxonomy.REGISTRY`):

* **point** events are kept when their ``systime`` lies in the slice;
* **interval** events still open at the slice end (they touch its last
  sample) are not emitted: the rows from the start of the earliest open
  interval on are carried into the next slice, which computes the interval
  again with more data. Closed intervals that start in the slice are kept,
  so ``min_duration`` filters and merged runs come out exactly as in memory;
* **summary** windows are kept when they start in the slice; a window that
  ends after the slice end (an overnight shift) is carried like an open
  interval. The chunk must be a whole multiple of the method's ``window``.

A few outputs need the whole run and are completed at the end of the stream
(see ``_RUNNING_COUNTS``, ``_ZERO_FILL`` and ``_LAST_STEP``): sample counts
from the start / to the end of a signal, zero windows between two slices'
events, and the last window of a coverage report.

Methods that need the whole signal at once (``static`` outputs, whole-signal
sigma or quantiles, model fits, baselines taken from the first window) are
not chunkable.

Sources
-------

* any iterable of long-format DataFrames, or of ``(name, DataFrame)`` pairs
  such as the loaders' ``stream_by_time_range``. Frames must come in time
  order at slice granularity: once a row of a later slice has been seen, a
  row of an earlier slice raises ``ValueError``;
* :func:`time_range_source`, which calls ``load(start, end)`` once per slice
  (databases, TimescaleDB, any loader with a time-range query).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning, NotChunkableError
from ts_shape.pipeline import _param_names, _resolve
from ts_shape.utils.frames import to_ns

if TYPE_CHECKING:
    from ts_shape.pipeline import Pipeline, PipelineResult

__all__ = [
    "CONDITIONAL",
    "SUPPORT",
    "Condition",
    "iter_chunked",
    "run_chunked",
    "time_range_source",
]

#: Default bound for carried rows (open intervals) and for waiting on the
#: next reading of a signal.
DEFAULT_MAX_CARRY = pd.Timedelta(days=7)

#: ``"Class.method"`` names that run chunked with the in-memory result. The
#: list is data-driven: each member passes the equivalence check of
#: ``tests/test_stream_equivalence.py`` (a 3-day frame across a DST change,
#: run in memory and in daily slices) and does not depend on whole-signal
#: statistics.
SUPPORT: frozenset[str] = frozenset(
    {
        "AlarmManagementEvents.alarm_frequency",
        "AlarmManagementEvents.chattering_detection",
        "AlarmManagementEvents.standing_alarms",
        "BatchTrackingEvents.detect_batches",
        "BottleneckDetectionEvents.detect_bottleneck",
        "BottleneckDetectionEvents.station_utilization",
        "CapabilityTrendingEvents.capability_over_time",
        "CapabilityTrendingEvents.yield_estimate",
        "ChangeoverEvents.detect_changeover",
        "ControlLoopHealthEvents.detect_oscillation",
        "ControlLoopHealthEvents.error_integrals",
        "DegradationDetectionEvents.detect_trend_degradation",
        "DemandPatternEvents.demand_by_period",
        "DowntimeEvents.availability_trend",
        "DowntimeEvents.downtime_by_shift",
        "DutyCycleEvents.cycle_count",
        "DutyCycleEvents.duty_cycle_by_window",
        "DutyCycleEvents.excessive_cycling",
        "DutyCycleEvents.on_off_intervals",
        "EnergyConsumptionEvents.consumption_by_window",
        "EnergyConsumptionEvents.energy_per_unit",
        "EnergyEfficiencyEvents.idle_energy_waste",
        "EnergyPerformanceIndicatorEvents.enpi_by_hierarchy",
        "EnergyPerformanceIndicatorEvents.enpi_by_window",
        "EnvironmentalCompensationEvents.detect_correlation",
        "EnvironmentalCompensationEvents.detect_env_exceedance",
        "EnvironmentalCompensationEvents.sensitivity",
        "IdleEnergyDetectionEvents.idle_energy_by_window",
        "InventoryMonitoringEvents.consumption_rate",
        "InventoryMonitoringEvents.detect_low_stock",
        "InventoryMonitoringEvents.reorder_point_breach",
        "InventoryMonitoringEvents.stockout_prediction",
        "LineThroughputEvents.count_parts",
        "LongDowntimeEvents.detect_long_downtime",
        "MachineStateEvents.detect_rapid_transitions",
        "MachineStateEvents.detect_run_idle",
        "MaterialBalanceEvents.balance_check",
        "MaterialBalanceEvents.contribution_breakdown",
        "MaterialBalanceEvents.imbalance_trend",
        "MeasurementUncertaintyEvents.combined_uncertainty",
        "MeasurementUncertaintyEvents.type_a_uncertainty",
        "MicroStopEvents.detect_micro_stops",
        "MicroStopEvents.micro_stop_frequency",
        "MicroStopEvents.micro_stop_impact",
        "MultiSensorValidationEvents.detect_disagreement",
        "OEEEvents.availability",
        "OEEEvents.oee",
        "OEEEvents.performance",
        "OEEEvents.quality",
        "OperatingRangeEvents.detect_regime_change",
        "OperatingRangeEvents.operating_envelope",
        "OperatingRangeEvents.time_in_range",
        "PartProductionEvents.daily_production_summary",
        "PartProductionEvents.detect_resets",
        "PartProductionEvents.production_by_part",
        "PerformanceLossEvents.slow_periods",
        "PeriodSummaryEvents.monthly_summary",
        "PeriodSummaryEvents.weekly_summary",
        "ProcessWindowEvents.detect_mean_shift",
        "ProcessWindowEvents.detect_variance_change",
        "ProcessWindowEvents.windowed_statistics",
        "QualityEvents.nok_by_shift",
        "RateOfChangeEvents.detect_rapid_change",
        "RateOfChangeEvents.detect_step_changes",
        "RateOfChangeEvents.rate_statistics",
        "ReworkEvents.rework_by_shift",
        "ReworkEvents.rework_rate",
        "ReworkEvents.rework_trend",
        "RuntimeAccountingEvents.runtime_by_window",
        "SamplingResolutionEvents.detect_aliasing",
        "SamplingResolutionEvents.detect_quantization",
        "SamplingResolutionEvents.effective_resolution",
        "SamplingResolutionEvents.sampling_jitter",
        "ScrapEvents.scrap_by_shift",
        "ScrapEvents.scrap_trend",
        "SensorDriftEvents.calibration_health",
        "SetpointChangeEvents.control_quality_metrics",
        "SetpointChangeEvents.decay_rate",
        "SetpointChangeEvents.detect_setpoint_changes",
        "SetpointChangeEvents.detect_setpoint_steps",
        "SetpointChangeEvents.oscillation_frequency",
        "SetpointChangeEvents.overshoot_metrics",
        "SetpointChangeEvents.rise_time",
        "SetpointChangeEvents.time_to_settle",
        "SetpointChangeEvents.time_to_settle_derivative",
        "SetupTimeEvents.setup_durations",
        "ShiftReportEvents.shift_production",
        "SignalComparisonEvents.deviation_statistics",
        "SignalQualityEvents.data_completeness",
        "SignalQualityEvents.detect_missing_data",
        "SignalQualityEvents.detect_out_of_range",
        "SignalQualityEvents.sampling_regularity",
        "StartupDetectionEvents.detect_startup_by_threshold",
        "SteadyStateDetectionEvents.detect_steady_state",
        "SteadyStateDetectionEvents.detect_transient_periods",
        "ThresholdMonitoringEvents.multi_level_threshold",
        "ThresholdMonitoringEvents.threshold_with_hysteresis",
        "ThresholdMonitoringEvents.time_above_threshold",
        "TimeToThresholdEvents.time_to_threshold_windows",
        "ToleranceDeviationEvents.detect_deviations",
        "ValueDistributionEvents.normality_windows",
        "ValueDistributionEvents.percentile_tracking",
        "WarmUpCoolDownEvents.detect_cooldown",
        "WarmUpCoolDownEvents.detect_warmup",
    }
)

#: Keyword arguments holding a rolling window or a look-back span; their
#: largest value is the default ``overlap`` (history before each slice).
_HISTORY_KWARGS = (
    "window",
    "rolling_window",
    "baseline_window",
    "comparison_window",
    "slope_window",
    "oscillation_window",
    "coincidence_window",
    "consumption_rate_window",
    "lookback",
    "max_delay",
    "time_tolerance",
    "time_threshold",
)

#: Keyword arguments whose value is a span of data *after* an event that the
#: method looks at; events closer than that to the slice end are carried.
#: ``max_duration`` classifies a stop by its full length: a stop still open
#: at a slice end is a micro-stop or not depending on data up to that long
#: after its start, so windows ending that close to the slice end wait.
_HORIZON_KWARGS = ("lookahead", "max_completion_time", "max_recovery", "max_duration")

#: Minimum-duration filters of interval methods. An interval still open at a
#: slice end may be shorter than the filter so far; a second run with the
#: filter off finds it, so it is carried instead of lost.
_MIN_DURATION_KWARGS = (
    "min_duration",
    "min_gap",
    "min_above",
    "min_step_duration",
    "min_phase_duration",
)

#: Output columns that number the events over the whole run (renumbered
#: across slices).
_ORDINALS: dict[str, str] = {
    "LongDowntimeEvents.detect_long_downtime": "downtime_index",
}

#: Methods that look back over the current run of a state signal (named by
#: this argument), e.g. the run before each micro-stop.
_SINCE_CHANGE: dict[str, str] = {
    "MicroStopEvents.detect_micro_stops": "run_state_uuid",
}


class Condition(NamedTuple):
    """What a method of :data:`CONDITIONAL` needs to run chunked."""

    #: The keyword argument to pass explicitly.
    argument: str
    #: Why: what its default (or another value) takes from the whole signal,
    #: and what to pass.
    reason: str


#: ``"Class.method"`` names that run chunked with the in-memory result only
#: when one argument is given: their default takes something from the whole
#: signal (its median sampling step), which an explicit value replaces. Each
#: entry passes the equivalence check with that argument; without it,
#: :func:`run_chunked` raises :class:`~ts_shape.errors.NotChunkableError`
#: naming the argument.
CONDITIONAL: dict[str, Condition] = {
    "DataGapAnalysisEvents.detect_gaps": Condition(
        "min_gap",
        "pass min_gap=... (the default is twice the whole signal's median "
        "sampling step)",
    ),
    "DataGapAnalysisEvents.coverage_by_period": Condition(
        "min_gap",
        "pass min_gap=... (the default is twice the whole signal's median "
        "sampling step)",
    ),
    "AnomalyClassificationEvents.detect_flatline": Condition(
        "interval_end",
        "pass interval_end='last_sample' (with the default 'next_change' a "
        "one-sample run counts only when the gap after it is longer than the "
        "whole signal's median sampling step)",
    ),
    "ChangeoverEvents.changeover_window": Condition(
        "until",
        "pass until='fixed_window' (the default; with until='stable_band' "
        "the stability reference is an expanding median over all data after "
        "each change, however long)",
    ),
}

#: Which values of a :data:`CONDITIONAL` argument make the method chunkable,
#: ``fn(explicit, value) -> bool`` (``explicit``: the argument was passed).
#: Default: any explicit value other than None.
_ACCEPTS: dict[str, Callable[[bool, Any], bool]] = {
    "AnomalyClassificationEvents.detect_flatline": (
        lambda explicit, value: value == "last_sample"
    ),
    "ChangeoverEvents.changeover_window": (
        lambda explicit, value: value != "stable_band"
    ),
}

#: Output columns counting the samples of a signal from the start / to the
#: end of the whole run: ``"Class.method" -> (uuid argument, before column,
#: after column)``. A running per-signal count offsets the before column of
#: each slice; the after column is the total count minus the before column,
#: so these rows are held until the end of the stream.
_RUNNING_COUNTS: dict[str, tuple[str, str, str]] = {
    "DataGapAnalysisEvents.detect_gaps": (
        "signal_uuid",
        "samples_before_gap",
        "samples_after_gap",
    ),
}

#: Window summaries that report every window from the first to the last
#: event, with zeros in between: the zero windows between two slices'
#: results are inserted on merge (same window grid as the method). The
#: columns listed are set to zero; the others are copied.
_ZERO_FILL: dict[str, tuple[str, ...]] = {
    "MicroStopEvents.micro_stop_frequency": (
        "count",
        "micro_stop_count",
        "total_lost_time",
        "total_lost_time_seconds",
        "pct_of_window",
        "share_of_window_pct",
    ),
}

#: Methods whose runs restart at the carried frontier: a run anchored at its
#: first sample (flatline) would chain differently if the reading before the
#: carried rows were kept as context, so it is not.
_FRESH_START: frozenset[str] = frozenset(
    {"AnomalyClassificationEvents.detect_flatline"}
)

#: Methods whose last window counts one median sampling step of the whole
#: signal after its last sample (``coverage_by_period``): the engine counts
#: every sampling step over the stream and corrects that window at the end.
_LAST_STEP: frozenset[str] = frozenset({"DataGapAnalysisEvents.coverage_by_period"})

#: Methods reporting calendar periods; they need the matching calendar chunk.
_PERIODS: dict[str, str] = {
    "PeriodSummaryEvents.weekly_summary": "W-SUN",
    "PeriodSummaryEvents.monthly_summary": "MS",
}

#: Per-method extra history, ``"Class.method" -> fn(params) -> Timedelta``.
_OVERLAP: dict[str, Callable[[dict[str, Any]], pd.Timedelta]] = {}


def _config_hold(params: dict[str, Any]) -> pd.Timedelta:
    """The product hold of ``changeover_window``'s ``config``."""
    config = params.get("config") or {}
    hold = config.get("min_duration", config.get("min_hold"))
    delta = _as_delta(hold) if hold is not None else None
    return delta if delta is not None else pd.Timedelta(0)


#: Per-method extra horizon (data needed after an event), like ``_OVERLAP``.
_HORIZON: dict[str, Callable[[dict[str, Any]], pd.Timedelta]] = {
    "ChangeoverEvents.changeover_window": _config_hold,
}

#: Why a method is not chunkable, for the most common requests.
_REASONS: dict[str, str] = {}

_GLOBAL_SIGMA = (
    "it flags values against statistics of the whole signal (mean and sigma, "
    "quartiles, median absolute deviation or a model fitted on all values)"
)
_SPC = "its control limits are computed from the whole signal"
_FIRST_WINDOW = "its baseline is taken from the first window of the data"
_ANCHORED = "its windows are anchored at the first timestamp of the data"
_PREVIOUS_CYCLE = (
    "each cycle reaches back to the previous cycle trigger, however long ago"
)
_ROLLING_WINDOWS = "it averages over a number of earlier windows (a rolling trend)"
_WHOLE_RANGE = "it returns one summary row for the whole time range"
for _method in (
    "detect_outliers_zscore",
    "detect_outliers_iqr",
    "detect_outliers_mad",
    "detect_outliers_isolation_forest",
):
    _REASONS[f"OutlierDetectionEvents.{_method}"] = _GLOBAL_SIGMA
for _method in (
    "control_limits",
    "detect_violations",
    "detect_rule_violations",
    "detect_cusum_shifts",
    "dynamic_control_limits",
    "interpret_violations",
    *(f"rule_{i}" for i in range(1, 9)),
):
    _REASONS[f"StatisticalProcessControlEvents.{_method}"] = _SPC
for _method in ("detect_change_points", "segment_statistics"):
    _REASONS[f"ChangePointEvents.{_method}"] = (
        "its segmentation (PELT / BIC penalty) depends on the whole signal"
    )
for _method in ("hotelling_t2", "contributions", "detect_out_of_control"):
    _REASONS[f"MultivariateSPCEvents.{_method}"] = (
        "its reference mean and covariance come from the whole data"
    )
for _method in ("health_index", "health_events", "maintenance_forecast"):
    _REASONS[f"AssetHealthIndexEvents.{_method}"] = _FIRST_WINDOW
for _method in ("health_score", "detect_level_shift", "detect_variance_increase"):
    _REASONS[f"DegradationDetectionEvents.{_method}"] = _FIRST_WINDOW
for _method in ("detect_zero_drift", "detect_span_drift", "drift_trend"):
    _REASONS[f"SensorDriftEvents.{_method}"] = _FIRST_WINDOW
for _method in ("detect_drift", "detect_oscillation", "classify_anomalies"):
    _REASONS[f"AnomalyClassificationEvents.{_method}"] = _ANCHORED
for _method in ("cycle_time_by_part", "detect_slow_cycles", "cycle_time_trend"):
    _REASONS[f"CycleTimeEvents.{_method}"] = _PREVIOUS_CYCLE
for _method in ("cycle_quality_check", "takt_adherence"):
    _REASONS[f"LineThroughputEvents.{_method}"] = _PREVIOUS_CYCLE
for _method in ("balance_metrics", "station_cycle_times"):
    _REASONS[f"LineBalancingEvents.{_method}"] = _PREVIOUS_CYCLE
for _method in ("detect_disturbances", "recovery_time", "before_after_comparison"):
    _REASONS[f"DisturbanceRecoveryEvents.{_method}"] = (
        "it classifies disturbances against the average rolling sigma of the "
        "whole signal"
    )
for _detector in (
    "LineThroughputEvents.throughput_trends",
    "EnergyEfficiencyEvents.efficiency_trend",
    "IdleEnergyDetectionEvents.idle_energy_trend",
    "EnergyConsumptionEvents.consumption_baseline_deviation",
    "EnergyPerformanceIndicatorEvents.enpi_vs_baseline",
):
    _REASONS[_detector] = _ROLLING_WINDOWS
for _detector in (
    "TargetEvents.target_hit_rate",
    "TargetEvents.target_achievement_summary",
    "MachineStateEvents.state_quality_metrics",
    "RuntimeAccountingEvents.runtime_summary",
    "ReliabilityEvents.mtbf_mttr",
    "ToleranceDeviationEvents.capability_indices",
):
    _REASONS[_detector] = _WHOLE_RANGE
_REASONS.update(
    {
        "DataGapAnalysisEvents.interpolation_candidates": (
            "safe_to_interpolate compares each jump with twice the standard "
            "deviation of the whole signal"
        ),
        "MachineStateEvents.transition_events": (
            "time_since_last_transition_seconds reaches back to the previous "
            "transition, however long ago"
        ),
        "LongDowntimeEvents.count_events_between_gaps": (
            "a window between two long downtimes can span any length, and "
            "production_df is a second full frame"
        ),
        "QualityEvents.daily_quality_summary": (
            "a production day includes the night shift's hours after midnight, "
            "past the row's end"
        ),
        "RuntimeAccountingEvents.operating_hours_meter": (
            "the meter accumulates from the start of the data"
        ),
        "ReliabilityEvents.failure_events": (
            "operating_hours_before accumulates from the start of the data"
        ),
    }
)
del _method, _detector

_STATIC_REASON = (
    "it returns one summary of the whole data (rows without a time of their "
    "own), so per-slice results cannot be merged"
)
_DEFAULT_REASON = (
    "its result depends on data outside a slice (whole-signal statistics, a "
    "baseline or model fitted on all data, or a window grid taken from the "
    "data's first timestamp), so a slice-by-slice run does not reproduce the "
    "in-memory result"
)

#: Names of output-naming uuid parameters (not signals the method reads).
_OUTPUT_UUID_PARAMS = ("event_uuid", "output_uuid", "result_uuid")

_TIME_COLUMNS = {"point": "systime", "interval": "start", "summary": "start"}

_EMPTY_COLUMNS = (
    "uuid",
    "value_double",
    "value_integer",
    "value_bool",
    "value_string",
)


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------


_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


@dataclass(frozen=True)
class _Grid:
    """The slice grid: a fixed length, or calendar weeks / months.

    ``calendar`` is ``"W-<DAY>"`` (pandas' weeks ending on ``<DAY>``, so
    ``"W-SUN"`` slices run Monday 00:00 to Monday 00:00) or ``"MS"``
    (calendar months); ``fixed`` is the length of a fixed grid.
    """

    fixed: pd.Timedelta | None = None
    calendar: str | None = None

    @property
    def label(self) -> str:
        return self.calendar if self.calendar is not None else str(self.fixed)

    @property
    def whole_days(self) -> bool:
        if self.fixed is None:
            return True
        return bool(self.fixed % pd.Timedelta(days=1) == pd.Timedelta(0))

    def floor(self, ts: pd.Timestamp) -> pd.Timestamp:
        if self.fixed is not None:
            return _floor(ts, self.fixed)
        from ts_shape.utils.frames import localize_in

        day = _floor(ts, pd.Timedelta(days=1))
        wall = day.tz_localize(None) if day.tzinfo is not None else day
        if self.calendar == "MS":
            wall = wall.replace(day=1)
        else:
            first = (_WEEKDAYS.index(str(self.calendar)[2:]) + 1) % 7
            wall = wall - pd.Timedelta(days=(wall.weekday() - first) % 7)
        return localize_in(wall, day.tz)

    def next_boundary(self, start: pd.Timestamp) -> pd.Timestamp:
        if self.fixed is not None:
            return _next_boundary(start, self.fixed)
        from ts_shape.utils.frames import localize_in

        wall = start.tz_localize(None) if start.tzinfo is not None else start
        step = (
            pd.DateOffset(months=1) if self.calendar == "MS" else pd.DateOffset(days=7)
        )
        return localize_in(wall + step, start.tz)


def _chunk_grid(chunk: Any) -> _Grid:
    """``chunk`` as a slice grid: a positive fixed length or ``"W"`` / ``"MS"``."""
    if isinstance(chunk, str):
        code = chunk.strip().upper()
        if code == "W":
            code = "W-SUN"
        if code == "MS" or (code.startswith("W-") and code[2:] in _WEEKDAYS):
            return _Grid(calendar=code)
    try:
        delta = pd.Timedelta(chunk)
    except (TypeError, ValueError):
        raise ValueError(
            f"chunk must be a fixed length such as '1D', '12h' or a Timedelta, "
            f"or a calendar period: 'W' (weeks from Monday), 'W-<DAY>' (weeks "
            f"ending on <DAY>) or 'MS' (calendar months); got {chunk!r}"
        ) from None
    if delta <= pd.Timedelta(0):
        raise ValueError(f"chunk must be positive, got {chunk!r}")
    return _Grid(fixed=delta)


def _floor(ts: pd.Timestamp, chunk: pd.Timedelta) -> pd.Timestamp:
    """The slice start holding ``ts`` (wall-clock aligned, DST-safe)."""
    from ts_shape.events._windows import floor_local

    floored = floor_local(pd.Series([ts]), _freq(chunk)).iloc[0]
    return pd.Timestamp(floored)


def _freq(chunk: pd.Timedelta) -> str:
    """A floor frequency for ``chunk`` (whole days as ``"nD"``)."""
    day = pd.Timedelta(days=1)
    if chunk >= day and chunk % day == pd.Timedelta(0):
        return f"{chunk // day}D"
    return f"{int(chunk / pd.Timedelta(1, 'ns'))}ns"


def _next_boundary(start: pd.Timestamp, chunk: pd.Timedelta) -> pd.Timestamp:
    """The slice boundary after ``start`` (a calendar day on DST days)."""
    from ts_shape.utils.frames import window_offset

    probe = start + window_offset(_freq(chunk))
    step = min(chunk, pd.Timedelta(hours=1))
    for _ in range(4):
        end = _floor(probe, chunk)
        if end > start:
            return end
        probe = probe + step
    raise AssertionError(f"no slice boundary after {start}")  # pragma: no cover


def _times(frame: pd.DataFrame, time_column: str) -> pd.Series:
    if time_column not in frame.columns:
        from ts_shape.errors import ColumnNotFoundError

        raise ColumnNotFoundError(
            f"source frame has no time column {time_column!r}; "
            f"columns: {list(frame.columns)}"
        )
    times = pd.to_datetime(frame[time_column])
    return pd.Series(times.array, copy=False)


def _as_frames(source: Any) -> Iterator[pd.DataFrame]:
    """Each DataFrame of ``source`` (frames or ``(name, frame)`` pairs)."""
    if isinstance(source, pd.DataFrame):
        yield source
        return
    for item in source:
        if isinstance(item, tuple) and len(item) == 2:
            item = item[1]
        if not isinstance(item, pd.DataFrame):
            raise TypeError(
                "a chunked source yields DataFrames or (name, DataFrame) "
                f"pairs, got {type(item).__name__}"
            )
        box, item = [item], None
        # Yield without a local name, so no reference to the frame stays
        # here while suspended (the consumer may filter and drop it).
        yield box.pop()


def time_range_source(
    load: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    start: Any,
    end: Any,
    chunk: Any = "1D",
    *,
    time_column: str = "systime",
) -> Iterator[pd.DataFrame]:
    """Load ``[start, end)`` one slice at a time.

    Calls ``load(slice_start, slice_end)`` for each slice of the chunk grid
    (the first and last slice are clipped to ``start`` / ``end``) and yields
    the frames in time order, ready for :func:`run_chunked`. Rows outside the
    half-open ``[slice_start, slice_end)`` are dropped, so a loader with an
    inclusive end does not deliver a row twice.

    Args:
        load: ``(start, end) -> DataFrame``, e.g. a database query or
            ``lambda s, e: loader.load_by_time_range(s, e)``.
        start: First instant (anything ``pd.Timestamp`` accepts).
        end: End of the range, exclusive.
        chunk: Slice length, aligned like :func:`run_chunked`'s ``chunk``.
        time_column: Time column of the loaded frames.

    Yields:
        One DataFrame per slice.
    """
    grid = _chunk_grid(chunk)
    from ts_shape.utils.frames import localize_in

    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if lo.tzinfo is not None and hi.tzinfo is None:
        hi = localize_in(hi, lo.tz)
    elif hi.tzinfo is not None and lo.tzinfo is None:
        lo = localize_in(lo, hi.tz)
    current = lo
    while current < hi:
        boundary = min(grid.next_boundary(grid.floor(current)), hi)
        frame = load(current, boundary)
        if len(frame) and time_column in frame.columns:
            times = _times(frame, time_column)
            tz = times.dt.tz
            keep = (
                (times >= localize_in(current, tz))
                & (times < localize_in(boundary, tz))
            ).to_numpy()
            if not keep.all():
                frame = frame[keep]
        yield frame
        current = boundary


@dataclass
class _Slice:
    start: pd.Timestamp
    end: pd.Timestamp
    core: pd.DataFrame
    ahead: pd.DataFrame
    final: bool


class _Slicer:
    """Cut a source into time-ordered slices with each signal's next reading."""

    def __init__(
        self,
        source: Any,
        grid: _Grid,
        time_column: str,
        max_carry: pd.Timedelta,
        watch: frozenset[str] | None,
    ) -> None:
        self._it = _as_frames(source)
        self._grid = grid
        self._time_column = time_column
        self._max_carry = max_carry
        self._watch = watch
        self._pending: list[tuple[pd.DataFrame, pd.Series]] = []
        self._exhausted = False
        self._done: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self.template: pd.DataFrame | None = None

    def _pull(self) -> None:
        try:
            frame = next(self._it)
        except StopIteration:
            self._exhausted = True
            return
        if self.template is None:
            self.template = frame.iloc[:0].copy()
        if frame.empty:
            return
        if self._watch is not None and "uuid" in frame.columns:
            # Only the watched signals are kept: the detectors read no other.
            watched = frame["uuid"].isin(self._watch).to_numpy()
            if not watched.all():
                frame = frame[watched]
                if frame.empty:
                    return
        times = _times(frame, self._time_column)
        if times.isna().any():
            keep = times.notna().to_numpy()
            frame, times = frame[keep], times[keep].reset_index(drop=True)
            if frame.empty:
                return
        if self._done is not None and times.min() < self._done[1]:
            first = times.min()
            raise ValueError(
                f"rows at {first} arrived after the slice "
                f"[{self._done[0]}, {self._done[1]}) was already processed; "
                f"a chunked source must yield frames in time order at slice "
                f"granularity -- stream the files sequentially in time order "
                f"or load the data with time_range_source()"
            )
        self._pending.append((frame, times))

    def _uuids(self, frame: pd.DataFrame, mask: Any = None) -> set[Any]:
        """The watched signals among ``frame``'s rows (``mask`` selects rows)."""
        if "uuid" not in frame.columns:
            return {None}
        values = frame["uuid"] if mask is None else frame["uuid"][mask]
        found = set(values.unique())
        if self._watch is not None:
            found &= self._watch
        return found

    def _ahead_complete(self, end: pd.Timestamp) -> bool:
        core: set[Any] = set()
        later: set[Any] = set()
        for frame, times in self._pending:
            mask = (times < end).to_numpy()
            if mask.any():
                core |= self._uuids(frame, None if mask.all() else mask)
            if not mask.all():
                later |= self._uuids(frame, ~mask)
        return core <= later

    def _horizon(self) -> pd.Timestamp:
        return max(t.max() for _, t in self._pending)

    def __iter__(self) -> Iterator[_Slice]:
        while True:
            while not self._pending and not self._exhausted:
                self._pull()
            if not self._pending:
                return
            first = min(t.min() for _, t in self._pending)
            start = self._grid.floor(first)
            end = self._grid.next_boundary(start)
            waited = False
            while not self._exhausted:
                last = self._horizon()
                if last >= end and self._ahead_complete(end):
                    break
                if last >= end + self._max_carry:
                    waited = True
                    break
                self._pull()
            core_parts: list[pd.DataFrame] = []
            keep: list[tuple[pd.DataFrame, pd.Series]] = []
            for frame, times in self._pending:
                mask = (times < end).to_numpy()
                if mask.all():
                    core_parts.append(frame)
                elif mask.any():
                    core_parts.append(frame[mask])
                    rest = ~mask
                    keep.append((frame[rest], times[rest].reset_index(drop=True)))
                else:
                    keep.append((frame, times))
            self._pending = keep
            core = (
                core_parts[0]
                if len(core_parts) == 1
                else pd.concat(core_parts, ignore_index=True)
            )
            ahead = self._first_rows()
            if waited:
                missing = sorted(map(str, self._uuids(core) - self._uuids(ahead)))
                warnings.warn(
                    f"no reading after {end} within max_carry for {missing}; "
                    f"their last reading in the slice is treated as their "
                    f"last reading",
                    DataQualityWarning,
                    stacklevel=3,
                )
            self._done = (start, end)
            final = self._exhausted and not self._pending
            piece = _Slice(start, end, core, ahead, final)
            # Hold no reference to the slice's rows while suspended, so a
            # consumer that drops them frees them.
            del core, core_parts, frame, times, keep, ahead
            yield piece
            del piece

    def _first_rows(self) -> pd.DataFrame:
        """The first pending row of each signal (the readings after the slice)."""
        if not self._pending:
            template = self.template if self.template is not None else pd.DataFrame()
            return template.iloc[:0]
        picks: list[pd.DataFrame] = []
        for frame, times in self._pending:
            if "uuid" in frame.columns:
                keys = frame["uuid"].reset_index(drop=True)
                ns = pd.Series(to_ns(times))
                first = ns.groupby(
                    keys, sort=False, dropna=False, observed=True
                ).idxmin()
                picks.append(frame.iloc[np.sort(first.to_numpy())])
            else:
                picks.append(frame.iloc[[int(np.argmin(to_ns(times)))]])
        rows = pd.concat(picks, ignore_index=True) if len(picks) > 1 else picks[0]
        times = _times(rows, self._time_column)
        rows = rows.iloc[np.argsort(to_ns(times), kind="stable")]
        if "uuid" in rows.columns:
            rows = rows.drop_duplicates("uuid")
        else:
            rows = rows.iloc[:1]
        return rows.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plans: one chunked detector call
# ---------------------------------------------------------------------------


def _key(target: Any, method: str) -> tuple[str, str]:
    from ts_shape.eventlog.taxonomy import resolve

    return resolve(target.__name__, method)


def _class_support(class_name: str) -> list[str]:
    prefix = f"{class_name}."
    names = [m.split(".", 1)[1] for m in SUPPORT if m.startswith(prefix)]
    names += [
        f"{m.split('.', 1)[1]} (with {cond.argument}=...)"
        for m, cond in CONDITIONAL.items()
        if m.startswith(prefix)
    ]
    return sorted(names)


def _listing(name: str) -> str:
    return ", ".join(_class_support(name)) or "none"


def _not_chunkable(target: Any, method: str | None, reason: str) -> NotChunkableError:
    name = getattr(target, "__name__", repr(target))
    label = f"{name}.{method}" if method else name
    listing = _listing(name) if method else "none"
    call = f"{name}(df, ...).{method}(...)" if method else f"{name}(df, ...)"
    return NotChunkableError(
        f"{label} cannot run chunked: {reason}. Run it in memory on the full "
        f"frame instead ({call}), e.g. on a time range that fits in memory. "
        f"Chunkable methods of {name}: {listing}."
    )


def _check_condition(
    target: Any, key: str, explicit: dict[str, Any], params: dict[str, Any]
) -> None:
    """Raise unless the call passes the argument :data:`CONDITIONAL` names."""
    cond = CONDITIONAL[key]
    given = cond.argument in explicit
    value = params.get(cond.argument)
    accepts = _ACCEPTS.get(key)
    ok = accepts(given, value) if accepts else given and value is not None
    if ok:
        return
    name = target.__name__
    method = key.split(".", 1)[1]
    raise NotChunkableError(
        f"{key} cannot run chunked with {cond.argument}={value!r}: "
        f"{cond.reason}. Chunkable methods of {name}: {_listing(name)}. "
        f"Without it, run it in memory ({name}(df, ...).{method}(...))."
    )


def _signature_defaults(func: Any) -> dict[str, Any]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        return {}
    return {
        p.name: p.default
        for p in sig.parameters.values()
        if p.default is not inspect.Parameter.empty
    }


def _renames(func: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    obj, seen = func, set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        out.update(getattr(obj, "__deprecated_kwargs__", {}))
        obj = getattr(obj, "__wrapped__", None)
    return out


def _effective(target: type, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Every keyword value the call will see (defaults filled in)."""
    init, func = getattr(target, "__init__"), getattr(target, method)
    params = {**_signature_defaults(init), **_signature_defaults(func)}
    renames = {**_renames(init), **_renames(func)}
    for key, value in kwargs.items():
        params[renames.get(key, key)] = value
    return params


def _explicit(target: type, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """``kwargs`` under their current names (deprecated names resolved)."""
    init, func = getattr(target, "__init__"), getattr(target, method)
    renames = {**_renames(init), **_renames(func)}
    return {renames.get(key, key): value for key, value in kwargs.items()}


def _as_delta(value: Any) -> pd.Timedelta | None:
    if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
        return None
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(delta) else delta


def _window_delta(window: Any) -> pd.Timedelta | None:
    """A window as a fixed length, or None (no window, calendar offsets)."""
    if window is None:
        return None
    delta = _as_delta(window)
    if delta is not None:
        return delta
    try:
        from pandas.tseries.frequencies import to_offset

        delta = pd.Timedelta(to_offset(window))
    except (TypeError, ValueError):
        return None
    return None if pd.isna(delta) else delta


def _check_window(key: str, shape: str, params: dict[str, Any], grid: _Grid) -> None:
    """The method's windows must tile the slices (shifts: whole days)."""
    period = _PERIODS.get(key)
    if period is not None and grid.calendar != period:
        name = "calendar weeks from Monday" if period == "W-SUN" else "calendar months"
        chunk = "'W'" if period == "W-SUN" else "'MS'"
        raise ValueError(
            f"{key} reports {name}; pass chunk={chunk} so that every period "
            f"is one slice, got chunk={grid.label!r}"
        )
    if shape == "summary" and "shift_definitions" in params and not grid.whole_days:
        raise ValueError(
            f"{key} reports per shift; chunk must be a whole number of days, "
            f"got {grid.label}"
        )
    if "window" not in params or params["window"] is None:
        return
    window = params["window"]
    delta = _window_delta(window)
    day = pd.Timedelta(days=1)
    if grid.fixed is not None:
        tiles = delta is not None and delta > pd.Timedelta(0) and not grid.fixed % delta
    else:
        tiles = delta is not None and delta > pd.Timedelta(0) and not day % delta
    if not tiles:
        raise ValueError(
            f"{key}: chunk ({grid.label}) must be a whole multiple of the "
            f"method's window ({window!r}) so that every window lies in one "
            f"slice; pass a chunk such as '1D' for windows that divide a day"
        )


def _has_dst(tz: Any, when: pd.Timestamp) -> bool:
    """Whether ``tz`` changes its UTC offset during the year of ``when``."""
    year = when.year
    winter = pd.Timestamp(year=year, month=1, day=1, tz=tz).utcoffset()
    summer = pd.Timestamp(year=year, month=7, day=1, tz=tz).utcoffset()
    return bool(winter != summer)


def _check_dst_window(plan: _Plan, times: pd.Series) -> None:
    """Multi-hour windows drift off local midnight after a DST change.

    pandas places ``resample`` bins of e.g. ``"8h"`` in absolute time from
    the first day of the data, so after a DST change an in-memory run bins
    01:00-09:00 where a slice that starts later bins 00:00-08:00. A window
    that divides one hour, or is whole days, stays aligned.
    """
    window = plan.window
    tz = times.dt.tz
    if window is None or tz is None or len(times) == 0:
        return
    hour, day = pd.Timedelta(hours=1), pd.Timedelta(days=1)
    if hour % window == pd.Timedelta(0) or window % day == pd.Timedelta(0):
        return
    if _has_dst(tz, times.iloc[0]):
        raise ValueError(
            f"{plan.key}: a {window} window on data in a time zone with "
            f"daylight saving time ({tz}) cannot run chunked -- its bins start "
            f"at a different hour after each DST change; use a window that "
            f"divides one hour or whole days, or convert the data to UTC"
        )


def _history(key: str, shape: str, params: dict[str, Any]) -> pd.Timedelta:
    if key in _OVERLAP:
        return _OVERLAP[key](params)
    spans = [pd.Timedelta(0)]
    for name in _HISTORY_KWARGS:
        if name == "window" and shape == "summary":
            continue
        delta = _as_delta(params.get(name))
        if delta is not None and delta > pd.Timedelta(0):
            spans.append(delta)
    return max(spans)


def _horizon(key: str, params: dict[str, Any], shape: str) -> pd.Timedelta:
    spans = [_HORIZON[key](params) if key in _HORIZON else pd.Timedelta(0)]
    # A point event that must hold for a minimum duration (a changeover
    # whose new product persists ``min_duration``) needs that much data after
    # it.
    names = _HORIZON_KWARGS + (_MIN_DURATION_KWARGS if shape == "point" else ())
    for name in names:
        delta = _as_delta(params.get(name))
        if delta is not None and delta > pd.Timedelta(0):
            spans.append(delta)
    return max(spans)


def _relevant_uuids(params: dict[str, Any]) -> frozenset[str] | None:
    """The signals the call reads, or None when that is not known.

    Read from the uuid-named arguments (``run_state_uuid``, ``signal_uuids``
    ...). Unknown when there is none, or when one of them is ``None`` (which
    often means "all signals").
    """
    found: set[str] = set()
    seen = False
    for name, value in params.items():
        if "uuid" not in name.lower() or name in _OUTPUT_UUID_PARAMS:
            continue
        seen = True
        if value is None:
            return None
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if not all(isinstance(v, str) for v in value):
                return None
            found.update(value)
        elif isinstance(value, dict):
            found.update(str(k) for k in value)
        else:
            return None
    return frozenset(found) if seen and found else None


@dataclass(frozen=True)
class _Plan:
    key: str
    shape: str
    executor: Callable[[pd.DataFrame, pd.DataFrame], Any]
    overlap: pd.Timedelta
    horizon: pd.Timedelta
    uuids: frozenset[str] | None
    time_column: str
    # The same call with its minimum-duration filters off: finds intervals
    # still open at a slice end that the filter hides (too short so far).
    probe: Callable[[pd.DataFrame, pd.DataFrame], Any] | None = None
    # An output column numbering the events 0, 1, 2, ... over the whole run.
    ordinal: str | None = None
    # A state signal whose current run (since its last change) the method
    # looks back over: kept as history for the next slice.
    since_change: str | None = None
    # The method's ``window`` as a fixed length (None when it has none).
    window: pd.Timedelta | None = None
    # The method's ``window`` argument as given (for the zero-fill grid).
    window_arg: Any = None
    # The signal and the before / after columns of a running sample count.
    running: tuple[str, str, str] | None = None
    # Columns of zero windows to insert between slices' results.
    zero_fill: tuple[str, ...] = ()
    # Do not keep the reading before the carried rows as context.
    fresh_start: bool = False
    # The signal whose last window is corrected with the whole-signal median
    # sampling step at the end of the stream.
    last_step: str | None = None


def _probe_kwargs(params: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """``kwargs`` with every positive minimum-duration filter set to zero."""
    off: dict[str, Any] = {}
    for name in _MIN_DURATION_KWARGS:
        delta = _as_delta(params.get(name))
        if delta is not None and delta > pd.Timedelta(0):
            off[name] = "0s"
    return {**kwargs, **off} if off else {}


def _plan(
    target: Any,
    method: str | None,
    kwargs: dict[str, Any],
    *,
    chunk: _Grid,
    overlap: Any,
    time_column: str,
    check: bool = True,
) -> _Plan:
    from ts_shape.eventlog.taxonomy import REGISTRY

    if method is None or not isinstance(target, type):
        raise _not_chunkable(
            target,
            method,
            "only (class, 'method') detector steps with a registered output "
            "shape can run chunked",
        )
    if not hasattr(target, method):
        raise AttributeError(f"{target.__name__!r} has no method {method!r}")
    class_name, method_name = _key(target, method)
    key = f"{class_name}.{method_name}"
    rule = REGISTRY.get((class_name, method_name))
    if rule is None:
        raise _not_chunkable(target, method, "it has no registered output shape")
    if check and key in CONDITIONAL:
        _check_condition(
            target,
            key,
            _explicit(target, method, kwargs),
            _effective(target, method, kwargs),
        )
    elif check and key not in SUPPORT:
        reason = _REASONS.get(key)
        if reason is None:
            reason = _STATIC_REASON if rule.shape == "static" else _DEFAULT_REASON
        raise _not_chunkable(target, method, reason)
    call_kwargs = dict(kwargs)
    if (
        time_column != "systime"
        and "time_column" not in call_kwargs
        and (
            "time_column" in _param_names(target)
            or "time_column" in _param_names(getattr(target, method))
        )
    ):
        call_kwargs["time_column"] = time_column
    executor, _ = _resolve(target, method, call_kwargs)
    params = _effective(target, method, call_kwargs)
    _check_window(key, rule.shape, params, chunk)
    if overlap is None:
        history = _history(key, rule.shape, params)
    else:
        history = _as_delta(overlap)
        if history is None or history < pd.Timedelta(0):
            raise ValueError(
                f"overlap must be a non-negative Timedelta, got {overlap!r}"
            )
    probe = None
    if rule.shape == "interval":
        probe_kwargs = _probe_kwargs(params, call_kwargs)
        if probe_kwargs:
            probe, _ = _resolve(target, method, probe_kwargs)
    return _Plan(
        key=key,
        shape=rule.shape,
        executor=executor,
        overlap=history,
        horizon=_horizon(key, params, rule.shape),
        uuids=_relevant_uuids(params),
        time_column=time_column,
        probe=probe,
        ordinal=_ORDINALS.get(key),
        since_change=(params.get(_SINCE_CHANGE[key]) if key in _SINCE_CHANGE else None),
        window=_window_delta(params.get("window")),
        window_arg=params.get("window"),
        running=(
            (
                str(params.get(_RUNNING_COUNTS[key][0])),
                _RUNNING_COUNTS[key][1],
                _RUNNING_COUNTS[key][2],
            )
            if key in _RUNNING_COUNTS
            else None
        ),
        zero_fill=_ZERO_FILL.get(key, ()),
        fresh_start=key in _FRESH_START,
        last_step=params.get("signal_uuid") if key in _LAST_STEP else None,
    )


# ---------------------------------------------------------------------------
# Engine: run one plan slice by slice
# ---------------------------------------------------------------------------


def _column_times(result: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(result[column])
    return pd.Series(values, index=result.index)


@dataclass
class _Engine:
    plan: _Plan
    max_carry: pd.Timedelta
    retained: pd.DataFrame | None = None
    frontier: pd.Timestamp | None = None
    empty: pd.DataFrame | None = None
    counted: int = 0
    # Running sample count of the plan's ``running`` signal (all slices so
    # far) and the rows held until its total is known.
    samples: int = 0
    held: list[pd.DataFrame] = field(default_factory=list)
    # Start of the last window emitted (zero fill).
    last_window: pd.Timestamp | None = None
    # Positive sampling steps of the ``last_step`` signal (ns -> count), and
    # the last windows held for the whole-signal correction: (rows, the
    # signal's last sample, the median step the slice's run used).
    steps: dict[int, int] = field(default_factory=dict)
    tails: list[tuple[pd.DataFrame, pd.Timestamp, pd.Timedelta]] = field(
        default_factory=list
    )

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        uuids = self.plan.uuids
        if uuids is None or "uuid" not in frame.columns or frame.empty:
            return frame
        mask = frame["uuid"].isin(uuids).to_numpy()
        return frame if mask.all() else frame[mask]

    def run(self, sl: _Slice) -> pd.DataFrame:
        """:meth:`step` for the only engine on a slice.

        The slice keeps only this engine's rows, so the rest is freed before
        the detector runs.
        """
        sl.core = self._select(sl.core)
        sl.ahead = self._select(sl.ahead)
        return self.step(sl, sl.core, sl.ahead)

    def step(self, sl: _Slice, core: pd.DataFrame, ahead: pd.DataFrame) -> pd.DataFrame:
        core = self._select(core)
        ahead = self._select(ahead)
        parts = [p for p in (self.retained, core, ahead) if p is not None and len(p)]
        if not parts:
            return self._empty_result(core)
        # Samples of the counted signal before the context's first row.
        offset = self.samples - self._count(self.retained)
        context = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
        del parts
        result = self._call(self.plan.executor, context)
        if self.empty is None:
            self.empty = result.iloc[:0]
        probe = None
        if self.plan.probe is not None and not sl.final:
            probe = self._call(self.plan.probe, context)
        core_times = _times(core, self.plan.time_column) if len(core) else None
        if core_times is not None and self.frontier is None:
            _check_dst_window(self.plan, core_times)
        last = core_times.max() if core_times is not None else sl.start
        frontier = sl.end
        if not sl.final:
            if self.plan.shape == "interval" and self.plan.overlap > pd.Timedelta(0):
                # A windowed interval may start up to one window before the
                # data that reveals it: those starts wait for the next slice.
                frontier = sl.end - self.plan.overlap
            for frame in (result, probe):
                if frame is not None:
                    frontier = min(frontier, self._open_since(frame, sl, last))
            if sl.end - frontier > self.max_carry:
                raise ValueError(
                    f"{self.plan.key}: an event open since {frontier} is still "
                    f"open at {sl.end}, longer than max_carry "
                    f"({self.max_carry}); pass a larger max_carry= or run on a "
                    f"shorter time range in memory"
                )
        emit = self._emit(result, frontier, sl)
        if self.plan.last_step is not None:
            emit = self._hold_last_window(emit, context, ahead, sl)
        if self.plan.running is not None:
            emit = self._hold_counted(emit, offset)
            self.samples += self._count(core)
        if self.plan.zero_fill:
            emit = self._fill_zeros(emit)
        times = _times(context, self.plan.time_column)
        self.retained = self._retain(context, times, frontier, sl.end)
        self.frontier = frontier
        return emit

    def close(self) -> pd.DataFrame:
        """The rows held until the end of the stream, completed."""
        parts: list[pd.DataFrame] = []
        if self.held and self.plan.running is not None:
            _, before, after = self.plan.running
            rows = pd.concat(self.held, ignore_index=True)
            rows[after] = self.samples - rows[before]
            parts.append(rows)
            self.held = []
        if self.tails:
            # Only the last of them closes the signal (earlier ones: a signal
            # that paused longer than max_carry, already warned about).
            parts.extend(rows for rows, _, _ in self.tails[:-1])
            rows, last, used = self.tails[-1]
            parts.append(self._correct_last_window(rows, last, used))
            self.tails = []
        if not parts:
            return self._empty_result(pd.DataFrame())
        return parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)

    # -- running counts ------------------------------------------------------

    def _count(self, frame: pd.DataFrame | None) -> int:
        """Rows of the counted signal in ``frame``."""
        if self.plan.running is None or frame is None or frame.empty:
            return 0
        if "uuid" not in frame.columns:
            return len(frame)
        return int((frame["uuid"] == self.plan.running[0]).sum())

    def _hold_counted(self, emit: pd.DataFrame, offset: int) -> pd.DataFrame:
        if emit.empty or self.plan.running is None:
            return emit
        before = self.plan.running[1]
        rows = emit.copy()
        if before in rows.columns:
            rows[before] = rows[before] + offset
        self.held.append(rows)
        return emit.iloc[:0]

    # -- zero fill -----------------------------------------------------------

    def _fill_zeros(self, emit: pd.DataFrame) -> pd.DataFrame:
        if emit.empty or "start" not in emit.columns:
            return emit
        starts = _column_times(emit, "start")
        order = np.argsort(to_ns(starts), kind="stable")
        emit, starts = emit.iloc[order], starts.iloc[order]
        first, last = starts.iloc[0], starts.iloc[-1]
        previous = self.last_window
        self.last_window = last if previous is None else max(previous, last)
        if previous is None or first <= previous:
            return emit
        zeros = self._zero_windows(previous, first, emit.iloc[[0]])
        if zeros.empty:
            return emit
        return pd.concat([zeros, emit], ignore_index=True)

    def _zero_windows(
        self, after: pd.Timestamp, before: pd.Timestamp, like: pd.DataFrame
    ) -> pd.DataFrame:
        """Zero rows for the method's windows strictly between two starts."""
        from ts_shape.events._windows import window_bounds

        window = self.plan.window_arg
        # The method's own grid: resample from the earlier window start.
        probe = pd.Series([0, 0], index=pd.DatetimeIndex([after, before]))
        labels = probe.resample(window).size().index
        labels = labels[(labels > after) & (labels < before)]
        if not len(labels):
            return like.iloc[:0]
        start, end = window_bounds(pd.Series(labels), window)
        rows = like.iloc[np.zeros(len(labels), dtype=int)].reset_index(drop=True)
        rows["start"] = start.reset_index(drop=True)
        if "end" in rows.columns:
            rows["end"] = end.reset_index(drop=True)
        if "duration_seconds" in rows.columns:
            rows["duration_seconds"] = (end - start).dt.total_seconds().to_numpy()
        for col in self.plan.zero_fill:
            if col in rows.columns:
                rows[col] = np.zeros(len(rows), dtype=rows[col].dtype)
        return rows

    # -- whole-signal sampling step -----------------------------------------

    def _signal_times(self, frame: pd.DataFrame) -> pd.Series:
        rows = frame
        if "uuid" in frame.columns:
            rows = frame[(frame["uuid"] == self.plan.last_step).to_numpy()]
        if rows.empty:
            return pd.Series([], dtype="datetime64[ns]")
        times = _times(rows, self.plan.time_column)
        return times.sort_values(kind="stable").reset_index(drop=True)

    def _hold_last_window(
        self,
        emit: pd.DataFrame,
        context: pd.DataFrame,
        ahead: pd.DataFrame,
        sl: _Slice,
    ) -> pd.DataFrame:
        """Count the slice's sampling steps; hold the signal's last window.

        Each step is counted once, by the slice holding its later sample.
        When no reading of the signal follows the slice, the windows ending
        after its last sample are held: their missing time after the last
        sample depends on the whole signal's median step, known at the end.
        """
        times = self._signal_times(context)
        if len(times) >= 2:
            ns = to_ns(times)
            later = ns[1:]
            diffs = later - ns[:-1]
            lo, hi = sl.start.value, sl.end.value
            mine = (later >= lo) & (later < hi) & (diffs > 0)
            values, counts = np.unique(diffs[mine], return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True):
                self.steps[value] = self.steps.get(value, 0) + count
        if not len(times) or len(self._signal_times(ahead)) or emit.empty:
            return emit
        last = times.iloc[-1]
        steps = times.diff()
        steps = steps[steps > pd.Timedelta(0)]
        used = pd.Timedelta(steps.median()) if len(steps) else pd.Timedelta(0)
        hold = (_column_times(emit, "end") > last).to_numpy()
        if hold.any():
            self.tails.append((emit[hold].copy(), last, used))
            emit = emit[~hold]
        return emit

    def _median_step(self) -> pd.Timedelta:
        """The median positive sampling step over the whole stream."""
        if not self.steps:
            return pd.Timedelta(0)
        values = np.array(sorted(self.steps), dtype=np.int64)
        cum = np.cumsum([self.steps[v] for v in values.tolist()])
        n = int(cum[-1])
        lo = values[np.searchsorted(cum, (n - 1) // 2, side="right")]
        hi = values[np.searchsorted(cum, n // 2, side="right")]
        # The same float arithmetic as Series.median() of all the steps.
        pair = pd.Series(pd.to_timedelta(np.array([lo, hi]), unit="ns"))
        return pd.Timedelta(pair.median())

    def _correct_last_window(
        self, rows: pd.DataFrame, last: pd.Timestamp, used: pd.Timedelta
    ) -> pd.DataFrame:
        """Recount the missing time after ``last`` with the whole-signal step.

        ``coverage_by_period`` counts ``end - max(start, last + step)`` as
        missing in a window ending after ``last + step``; the slice's run used
        ``used`` as step.
        """
        step = self._median_step()
        if step == used or not {"gap_seconds", "coverage_pct"} <= set(rows.columns):
            return rows
        start = _column_times(rows, "start")
        end = _column_times(rows, "end")

        def missing(after: pd.Timedelta) -> pd.Series:
            data_end = last + after
            lower = start.where(start > data_end, data_end)
            seconds = (end - lower).dt.total_seconds()
            return seconds.where(end > data_end, 0.0)

        out = rows.copy()
        gap = out["gap_seconds"] - missing(used) + missing(step)
        out["gap_seconds"] = gap
        span = (end - start).dt.total_seconds()
        out["coverage_pct"] = ((1.0 - gap / span) * 100).clip(0.0, 100.0)
        return out

    def _call(
        self, executor: Callable[..., Any], context: pd.DataFrame
    ) -> pd.DataFrame:
        result = executor(context, context)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"{self.plan.key} returned {type(result).__name__}, "
                f"expected a DataFrame"
            )
        return result

    def _empty_result(self, core: pd.DataFrame) -> pd.DataFrame:
        if self.empty is not None:
            return self.empty
        return core.iloc[:0]

    def _start(self, result: pd.DataFrame) -> pd.Series:
        column = _TIME_COLUMNS[self.plan.shape]
        if column not in result.columns:
            raise ValueError(
                f"{self.plan.key} returned no {column!r} column; cannot place "
                f"its rows in slices"
            )
        return _column_times(result, column)

    def _lower(self, start: pd.Series) -> pd.Series:
        if self.frontier is None:
            return start.notna()
        return start.notna() & (start >= self.frontier)

    def _is_open(
        self, result: pd.DataFrame, start: pd.Series, sl: _Slice, last: pd.Timestamp
    ) -> pd.Series:
        """Rows that more data after the slice could still change."""
        plan = self.plan
        if plan.shape == "point":
            if plan.horizon > pd.Timedelta(0):
                return start >= last - plan.horizon
            return start >= sl.end
        end = _column_times(result, "end") if "end" in result.columns else start
        if plan.shape == "interval":
            return end.isna() | (end >= last - plan.horizon)
        return end.isna() | (end > sl.end - plan.horizon)

    def _open_since(
        self, result: pd.DataFrame, sl: _Slice, last: pd.Timestamp
    ) -> pd.Timestamp:
        """The start of the earliest open row (the slice end when none)."""
        if result.empty:
            return sl.end
        start = self._start(result)
        candidates = start[self._is_open(result, start, sl, last) & self._lower(start)]
        return min(sl.end, candidates.min()) if len(candidates) else sl.end

    def _emit(
        self, result: pd.DataFrame, frontier: pd.Timestamp, sl: _Slice
    ) -> pd.DataFrame:
        if result.empty:
            return result
        start = self._start(result)
        keep = self._lower(start)
        if not sl.final:
            keep &= start < frontier
        out = result[keep.to_numpy()]
        column = self.plan.ordinal
        if column is not None and column in out.columns and len(out):
            start = _column_times(out, _TIME_COLUMNS[self.plan.shape])
            order = np.argsort(to_ns(start), kind="stable")
            out = out.iloc[order].copy()
            out[column] = np.arange(self.counted, self.counted + len(out))
            self.counted += len(out)
        return out

    def _retain(
        self,
        context: pd.DataFrame,
        times: pd.Series,
        frontier: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        cut = frontier - self.plan.overlap
        if self.plan.since_change is not None:
            cut = min(cut, self._last_change(context, times, frontier))
            if end - cut > self.max_carry:
                raise ValueError(
                    f"{self.plan.key}: the state of {self.plan.since_change!r} "
                    f"has not changed since {cut}, longer than max_carry "
                    f"({self.max_carry}); pass a larger max_carry= or run on a "
                    f"shorter time range in memory"
                )
        inside = ((times >= cut) & (times < end)).to_numpy()
        before = (times < cut).to_numpy()
        pieces = []
        if before.any() and not self.plan.fresh_start:
            earlier = context[before]
            order = np.argsort(to_ns(times[before]), kind="stable")
            earlier = earlier.iloc[order]
            if "uuid" in earlier.columns:
                pieces.append(earlier.drop_duplicates("uuid", keep="last"))
            else:
                pieces.append(earlier.iloc[-1:])
        if inside.any():
            pieces.append(context[inside])
        if not pieces:
            return context.iloc[:0]
        return pd.concat(pieces, ignore_index=True)

    def _last_change(
        self, context: pd.DataFrame, times: pd.Series, before: pd.Timestamp
    ) -> pd.Timestamp:
        """When the state signal last changed value before ``before``.

        That is the start of the run in progress just before ``before``; the
        earliest context time when the signal did not change in the context.
        """
        uuid = self.plan.since_change
        mask = (times < before).to_numpy().copy()
        if "uuid" in context.columns:
            mask &= (context["uuid"] == uuid).to_numpy()
        rows = context[mask]
        if rows.empty:
            return before
        stamps = times[mask].reset_index(drop=True)
        order = np.argsort(to_ns(stamps), kind="stable")
        rows, stamps = rows.iloc[order], stamps.iloc[order].reset_index(drop=True)
        differs = np.zeros(max(len(rows) - 1, 0), dtype=bool)
        for col in _EMPTY_COLUMNS[1:]:
            if col not in rows.columns:
                continue
            values = rows[col].reset_index(drop=True)
            now, prev = values.iloc[1:], values.iloc[:-1]
            missing = now.isna().to_numpy() & prev.isna().to_numpy()
            same = (now.to_numpy() == prev.to_numpy()) | missing
            differs |= ~np.asarray(same, dtype=bool)
        changed = np.flatnonzero(differs) + 1
        if not len(changed):
            return stamps.min() if len(stamps) else before
        return min(before, stamps.iloc[int(changed[-1])])


def _empty_input(template: pd.DataFrame | None, time_column: str) -> pd.DataFrame:
    if template is not None:
        return template.iloc[:0].copy()
    cols: dict[str, pd.Series] = {time_column: pd.Series(dtype="datetime64[ns]")}
    for col in _EMPTY_COLUMNS:
        dtype = {"value_double": "float64", "value_integer": "Int64"}.get(col, object)
        cols[col] = pd.Series(dtype=dtype)
    return pd.DataFrame(cols)


def _finish(parts: list[pd.DataFrame], empty: pd.DataFrame) -> pd.DataFrame:
    filled = [p for p in parts if len(p)]
    if not filled:
        return empty.iloc[:0].reset_index(drop=True)
    if len(filled) == 1:
        return filled[0].reset_index(drop=True)
    out = pd.concat(filled, ignore_index=True)
    # A slice whose values are all missing may give an object column (None)
    # where the others give numbers; keep the typed column.
    for col in out.columns:
        if not pd.api.types.is_object_dtype(out[col]):
            continue
        if any(
            col in p.columns and not pd.api.types.is_object_dtype(p[col])
            for p in filled
        ):
            out[col] = out[col].infer_objects()
    return out


def _max_carry(value: Any) -> pd.Timedelta:
    if value is None:
        return DEFAULT_MAX_CARRY
    delta = _as_delta(value)
    if delta is None or delta <= pd.Timedelta(0):
        raise ValueError(f"max_carry must be a positive Timedelta, got {value!r}")
    return delta


def iter_chunked(
    source: Any,
    target: Any,
    method: str | None = None,
    *,
    chunk: Any = "1D",
    overlap: Any = None,
    time_column: str = "systime",
    max_carry: Any = None,
    **kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Run a detector slice by slice and yield the finished rows of each slice.

    Same arguments as :func:`run_chunked`. Each result row is yielded exactly
    once, in slice order; slices that finish no row yield nothing. Use it to
    write results out as they come (e.g. one parquet file per slice).

    Rows that need the whole run are yielded together at the end, after the
    last slice: every row of ``DataGapAnalysisEvents.detect_gaps``
    (``samples_after_gap`` counts to the end of the signal) and the last
    window of ``DataGapAnalysisEvents.coverage_by_period`` (its missing time
    after the last sample uses the whole signal's median sampling step).

    Raises:
        NotChunkableError: If the method is not in :data:`SUPPORT`, or is in
            :data:`CONDITIONAL` and the call lacks the argument named there.
        ValueError: For a chunk that does not tile the method's windows, an
            out-of-order source, or an event open longer than ``max_carry``.
    """
    step = _chunk_grid(chunk)
    plan = _plan(
        target, method, kwargs, chunk=step, overlap=overlap, time_column=time_column
    )
    carry = _max_carry(max_carry)

    def _gen() -> Iterator[pd.DataFrame]:
        engine = _Engine(plan, carry)
        for sl in _Slicer(source, step, time_column, carry, plan.uuids):
            out = engine.run(sl)
            if len(out):
                yield out.reset_index(drop=True)
        tail = engine.close()
        if len(tail):
            yield tail.reset_index(drop=True)

    return _gen()


def run_chunked(
    source: Any,
    target: Any,
    method: str | None = None,
    *,
    chunk: Any = "1D",
    overlap: Any = None,
    time_column: str = "systime",
    max_carry: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run a detector slice by slice on data larger than memory.

    The result equals ``target(full_df, **init_kwargs).method(**method_kwargs)``
    on the concatenated data (up to float rounding and row order) while
    memory holds about one slice.

    Args:
        source: An iterable of long-format DataFrames or of ``(name,
            DataFrame)`` pairs (a loader's ``stream_by_time_range``), in time
            order at slice granularity, or a :func:`time_range_source`.
        target: The detector class, e.g. ``MachineStateEvents``.
        method: The method name, e.g. ``"detect_run_idle"``. Deprecated
            names resolve to their current names.
        chunk: Slice length (``"1D"`` default, ``"12h"``, a Timedelta), or a
            calendar period: ``"W"`` (Monday 00:00 to Monday 00:00; also
            ``"W-<DAY>"``, weeks ending on ``<DAY>``) or ``"MS"`` (calendar
            months). Slices are aligned to the wall clock of the data's
            timezone, so a daily slice on a DST change lasts 23 or 25 hours.
            ``weekly_summary`` / ``monthly_summary`` need ``"W"`` / ``"MS"``.
        overlap: History before each slice for rolling methods. Default:
            the method's window-like argument (``window``,
            ``baseline_window``, ...), or none.
        time_column: Time column of the source frames (also passed to the
            detector when it is not ``"systime"``).
        max_carry: Longest an event may stay open across slices, and longest
            the run waits for a signal's next reading (default 7 days).
        **kwargs: Constructor and method arguments, routed by name as in
            :meth:`ts_shape.Pipeline.detect`.

    Returns:
        The detector's result, one row per event or window, in slice order
        with a fresh index. An empty source gives the method's empty result.

    Raises:
        NotChunkableError: If the method is not in :data:`SUPPORT`, or is in
            :data:`CONDITIONAL` and the call lacks the argument named there.
        ValueError: For a chunk that does not tile the method's windows, an
            out-of-order source, or an event open longer than ``max_carry``.

    Example:
        >>> import pandas as pd
        >>> from ts_shape import MachineStateEvents, run_chunked
        >>> t = pd.date_range("2024-01-01", periods=6, freq="12h")
        >>> df = pd.DataFrame({"systime": t, "uuid": "run",
        ...                    "value_bool": [1, 1, 0, 1, 1, 1]})
        >>> days = (g for _, g in df.groupby(df["systime"].dt.date))
        >>> out = run_chunked(days, MachineStateEvents, "detect_run_idle",
        ...                   run_state_uuid="run")
        >>> out[["start", "end", "state"]]
                        start                 end state
        0 2024-01-01 00:00:00 2024-01-02 00:00:00   run
        1 2024-01-02 00:00:00 2024-01-02 12:00:00  idle
        2 2024-01-02 12:00:00 2024-01-03 12:00:00   run
    """
    return _run_chunked(
        source,
        target,
        method,
        chunk=chunk,
        overlap=overlap,
        time_column=time_column,
        max_carry=max_carry,
        kwargs=kwargs,
    )


def _run_chunked(
    source: Any,
    target: Any,
    method: str | None,
    *,
    chunk: Any,
    overlap: Any,
    time_column: str,
    max_carry: Any,
    kwargs: dict[str, Any],
    check: bool = True,
) -> pd.DataFrame:
    step = _chunk_grid(chunk)
    plan = _plan(
        target,
        method,
        kwargs,
        chunk=step,
        overlap=overlap,
        time_column=time_column,
        check=check,
    )
    carry = _max_carry(max_carry)
    engine = _Engine(plan, carry)
    slicer = _Slicer(source, step, time_column, carry, plan.uuids)
    parts = [engine.run(sl) for sl in slicer]
    parts.append(engine.close())
    if engine.empty is None:
        engine.empty = plan.executor(
            *(2 * [_empty_input(slicer.template, time_column)])
        )
    return _finish(parts, engine.empty)


def _run_pipeline(
    pipe: Pipeline,
    source: Any,
    *,
    chunk: Any,
    overlap: Any,
    time_column: str,
    max_carry: Any,
    keep_data: bool,
) -> PipelineResult:
    """The engine behind :meth:`ts_shape.Pipeline.run_stream`."""
    from ts_shape.pipeline import PipelineResult

    steps = list(pipe._steps)
    if steps and steps[0].kind == "source":
        raise TypeError(
            f"pipeline {pipe.name!r} defines a source step; run_stream() takes "
            f"its data from the source argument -- build the pipeline without "
            f".source()"
        )
    step_len = _chunk_grid(chunk)
    carry = _max_carry(max_carry)
    engines: dict[str, _Engine] = {}
    detector_ids: dict[str, str | None] = {}
    for step in steps:
        if step.kind != "detect":
            continue
        plan = _plan(
            step.target,
            step.method,
            step.kwargs,
            chunk=step_len,
            overlap=overlap,
            time_column=time_column,
        )
        engines[step.name] = _Engine(plan, carry)
        detector_ids[step.name] = step.detector_id
    watch: frozenset[str] | None = frozenset()
    for engine in engines.values():
        if engine.plan.uuids is None or watch is None:
            watch = None
        else:
            watch = watch | engine.plan.uuids
    transforms = [s for s in steps if s.kind == "transform"]
    if transforms:
        watch = None  # a transform may create or rename signals
    outputs: dict[str, list[pd.DataFrame]] = {name: [] for name in engines}
    data_parts: list[pd.DataFrame] = []
    data_empty: pd.DataFrame | None = None
    slicer = _Slicer(source, step_len, time_column, carry, watch)

    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        raw = frame
        working = frame
        for index, step in enumerate(transforms):
            try:
                result = step.executor(working.copy(), raw)
            except Exception as exc:
                raise RuntimeError(
                    f"pipeline {pipe.name!r}: transform {step.name!r} "
                    f"(#{index}) failed: {exc}"
                ) from exc
            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"pipeline {pipe.name!r}: transform {step.name!r} returned "
                    f"{type(result).__name__}, expected a DataFrame"
                )
            working = result
        return working

    for sl in slicer:
        core = _transform(sl.core)
        ahead = _transform(sl.ahead) if len(sl.ahead) else sl.ahead
        if data_empty is None:
            data_empty = core.iloc[:0]
        if keep_data:
            data_parts.append(core)
        for name, engine in engines.items():
            outputs[name].append(engine.step(sl, core, ahead))
    for name, engine in engines.items():
        outputs[name].append(engine.close())

    empty_in: pd.DataFrame | None = None
    events: dict[str, pd.DataFrame] = {}
    for name, engine in engines.items():
        if engine.empty is None:
            if empty_in is None:
                empty_in = _transform(_empty_input(slicer.template, time_column))
            engine.empty = engine.plan.executor(empty_in, empty_in)
        events[name] = _finish(outputs[name], engine.empty)
    if data_empty is None:
        data_empty = (
            empty_in
            if empty_in is not None
            else _transform(_empty_input(slicer.template, time_column))
        )
    data = _finish(data_parts, data_empty) if keep_data else data_empty.iloc[:0]
    return PipelineResult(
        name=pipe.name, data=data, events=events, _detector_ids=detector_ids
    )
