"""Vectorized detectors give the same frames as their former loop versions.

The ``_ref_*`` functions below are the implementations these methods had
before they were vectorized (per-run ``groupby`` loops and per-row loops),
kept verbatim minus docstrings and comments. Each test runs the current
method and its reference on seeded random frames -- duplicates, NaN values,
tz-aware and nanosecond timestamps, tiny and constant signals -- and asserts
the frames are identical: values (bit for bit, signed zeros included),
dtypes, column order, row order and index.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import (
    add_twin_columns,
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
    finalize_summary_df,
)
from ts_shape.events._windows import add_window_bounds, finish_events
from ts_shape.events.engineering.rate_of_change import RateOfChangeEvents
from ts_shape.events.engineering.startup_events import (
    StartupDetectionEvents,
    _between,
)
from ts_shape.events.engineering.threshold_monitoring import (
    ThresholdMonitoringEvents,
)
from ts_shape.events.engineering.warmup_analysis import (
    WarmUpCoolDownEvents,
    _bridge_pauses,
)
from ts_shape.events.maintenance._rolling import time_window_bounds, window_moments
from ts_shape.events.maintenance.degradation_detection import (
    DegradationDetectionEvents,
)
from ts_shape.events.maintenance.vibration_analysis import VibrationAnalysisEvents
from ts_shape.events.production.alarm_management import AlarmManagementEvents
from ts_shape.events.production.batch_tracking import BatchTrackingEvents
from ts_shape.events.production.changeover import ChangeoverEvents
from ts_shape.events.production.flow_constraints import (
    FlowConstraintEvents,
    _check_interval_end,
)
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.multi_process_traceability import (
    MultiProcessTraceabilityEvents,
    _parallel_flags,
)
from ts_shape.events.production.operating_state import OperatingStateEvents
from ts_shape.events.production.order_traceability import ValueTraceabilityEvents
from ts_shape.events.production.routing_traceability import (
    RoutingTraceabilityEvents,
)
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents
from ts_shape.events.quality.signal_quality import SignalQualityEvents
from ts_shape.events.supplychain.inventory_monitoring import (
    InventoryMonitoringEvents,
)
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import REMOVAL, caller_stacklevel
from ts_shape.utils.intervals import INTERVAL_END_MODES, interval_end_times

# ---------------------------------------------------------------------------
# Reference implementations (the former loop versions)
# ---------------------------------------------------------------------------


def _ref_rateofchange_detect_rapid_change(
    self, threshold: float, window: str | None = None
) -> pd.DataFrame:
    if window is not None:
        warnings.warn(
            f"RateOfChangeEvents.detect_rapid_change: window is not used (the rate is computed between consecutive samples) and will be removed in {REMOVAL}; filter the result on duration_seconds instead",
            DeprecationWarning,
            stacklevel=caller_stacklevel(),
        )
    cols = ["start", "end", "max_rate", "direction"]
    empty = empty_event_df("interval", ["max_rate", "direction"])
    rate_df = self._compute_rate()
    if rate_df.empty:
        return empty
    rapid = rate_df["rate"].abs() >= threshold
    if not rapid.any():
        return empty
    groups = (rapid != rapid.shift()).cumsum()
    events: list[dict[str, Any]] = []
    for _, seg in rate_df.groupby(groups):
        seg_rapid = rapid.loc[seg.index]
        if not seg_rapid.iloc[0]:
            continue
        # The rate at row i describes the step from row i-1 (rate_df keeps
        # the positional labels of the signal it was computed from).
        start = self.signal[self.time_column].iloc[seg.index[0] - 1]
        end = seg[self.time_column].iloc[-1]
        max_rate_idx = seg["rate"].abs().idxmax()
        max_rate = float(seg.loc[max_rate_idx, "rate"])
        events.append(
            {
                "start": start,
                "end": end,
                "max_rate": max_rate,
                "direction": "increasing" if max_rate > 0 else "decreasing",
            }
        )
    if not events:
        return empty
    return finalize_interval_df(
        pd.DataFrame(events, columns=cols),
        uuid=self.event_uuid,
        source_uuid=self.signal_uuid,
    )


def _ref_rateofchange_detect_step_changes(
    self, min_delta: float, max_duration: str = "5s"
) -> pd.DataFrame:
    cols = ["time", "value_before", "value_after", "delta", "duration"]
    empty = empty_event_df("point", cols)
    if self.signal.empty or len(self.signal) < 2:
        return empty
    sig = (
        self.signal[[self.time_column, self.value_column]].copy().reset_index(drop=True)
    )
    max_td = pd.to_timedelta(max_duration)
    events: list[dict[str, Any]] = []
    values = sig[self.value_column].values
    times = sig[self.time_column].array
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        duration = pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])
        if abs(delta) >= min_delta and duration <= max_td:
            events.append(
                {
                    "time": pd.Timestamp(times[i]),
                    "value_before": float(values[i - 1]),
                    "value_after": float(values[i]),
                    "delta": float(delta),
                    "duration": duration,
                }
            )
    if not events:
        return empty
    out = pd.DataFrame(events, columns=cols)
    out["systime"] = out["time"]
    return finalize_point_df(out, uuid=self.event_uuid, source_uuid=self.signal_uuid)


def _with_step_bounds(out: pd.DataFrame) -> pd.DataFrame:
    """The reference step changes plus the canonical bounds (added in round 11).

    ``start`` / ``end`` follow ``time`` and ``duration_seconds`` follows
    ``duration``; the step values are unchanged.
    """
    if out.empty:
        cols = ["time", "start", "end", "value_before", "value_after", "delta"]
        return empty_event_df("point", [*cols, "duration", "duration_seconds"])
    out = out.copy()
    duration = pd.to_timedelta(out["duration"])
    at = list(out.columns).index("time") + 1
    out.insert(at, "start", out["time"] - duration)
    out.insert(at + 1, "end", out["time"])
    out.insert(
        list(out.columns).index("duration") + 1,
        "duration_seconds",
        duration.dt.total_seconds().astype(float),
    )
    return out


def _ref_degradationdetection_detect_trend_degradation(
    self,
    window: str = "1h",
    min_slope: float = 0.0,
    direction: str = "decreasing",
    *,
    interval_end: str = "next_change",
) -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "uuid",
        "source_uuid",
        "is_delta",
        "avg_slope",
        "total_change",
        "duration_seconds",
    ]
    if self.signal.empty:
        return pd.DataFrame(columns=cols)
    sig = (
        self.signal[[self.time_column, self.value_column]].copy().reset_index(drop=True)
    )
    sig["t_seconds"] = (
        sig[self.time_column] - sig[self.time_column].iloc[0]
    ).dt.total_seconds()
    window_td = pd.to_timedelta(window)
    lo, hi = time_window_bounds(sig[self.time_column], window_td)
    mom = window_moments(
        sig["t_seconds"].to_numpy(), sig[self.value_column].to_numpy(), lo, hi
    )
    sig["slope"] = np.where(mom["n"] < mom["n_total"], np.nan, mom["slope"])
    if direction == "decreasing":
        degrading = sig["slope"] <= -abs(min_slope)
    else:
        degrading = sig["slope"] >= abs(min_slope)
    degrading = degrading.fillna(False)
    if not degrading.any():
        return pd.DataFrame(columns=cols)
    group_id = (degrading != degrading.shift()).cumsum()
    ends = interval_end_times(sig[self.time_column], interval_end)
    events: list[dict[str, Any]] = []
    for _, seg in sig.groupby(group_id):
        seg_deg = degrading.loc[seg.index]
        if not seg_deg.iloc[0]:
            continue
        start_time = seg[self.time_column].iloc[0]
        end_time = ends.loc[seg.index[-1]]
        avg_slope = float(seg["slope"].mean())
        total_change = float(
            seg[self.value_column].iloc[-1] - seg[self.value_column].iloc[0]
        )
        duration = (end_time - start_time).total_seconds()
        events.append(
            {
                "start": start_time,
                "end": end_time,
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "avg_slope": avg_slope,
                "total_change": total_change,
                "duration_seconds": duration,
            }
        )
    return pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)


def _ref_degradationdetection_detect_variance_increase(
    self,
    window: str = "1h",
    threshold_factor: float = 2.0,
    *,
    interval_end: str = "next_change",
) -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "uuid",
        "source_uuid",
        "is_delta",
        "baseline_variance",
        "current_variance",
        "ratio",
        "duration_seconds",
    ]
    if self.signal.empty:
        return pd.DataFrame(columns=cols)
    sig = (
        self.signal[[self.time_column, self.value_column]].copy().reset_index(drop=True)
    )
    window_td = pd.to_timedelta(window)
    first_time = sig[self.time_column].iloc[0]
    baseline_mask = sig[self.time_column] <= first_time + window_td
    baseline_data = sig.loc[baseline_mask, self.value_column]
    if len(baseline_data) < 2:
        return pd.DataFrame(columns=cols)
    baseline_var = float(baseline_data.var())
    if baseline_var == 0:
        baseline_var = np.finfo(float).eps
    lo, hi = time_window_bounds(sig[self.time_column], window_td)
    sig["current_variance"] = window_moments(
        np.zeros(len(sig)), sig[self.value_column].to_numpy(), lo, hi
    )["var"]
    sig["ratio"] = sig["current_variance"] / baseline_var
    exceeded = (sig["ratio"] >= threshold_factor).fillna(False)
    if not exceeded.any():
        return pd.DataFrame(columns=cols)
    group_id = (exceeded != exceeded.shift()).cumsum()
    ends = interval_end_times(sig[self.time_column], interval_end)
    events: list[dict[str, Any]] = []
    for _, seg in sig.groupby(group_id):
        seg_exc = exceeded.loc[seg.index]
        if not seg_exc.iloc[0]:
            continue
        events.append(
            {
                "start": seg[self.time_column].iloc[0],
                "end": ends.loc[seg.index[-1]],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "baseline_variance": baseline_var,
                "current_variance": float(seg["current_variance"].mean()),
                "ratio": float(seg["ratio"].mean()),
                "duration_seconds": (
                    ends.loc[seg.index[-1]] - seg[self.time_column].iloc[0]
                ).total_seconds(),
            }
        )
    return pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)


def _ref_vibrationanalysis_detect_rms_exceedance(
    self,
    baseline_rms: float,
    threshold_factor: float = 1.5,
    window: str = "1min",
    *,
    interval_end: str = "next_change",
) -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "uuid",
        "source_uuid",
        "is_delta",
        "rms_value",
        "baseline_rms",
        "ratio",
        "duration_seconds",
    ]
    if self.signal.empty:
        return pd.DataFrame(columns=cols)
    sig = (
        self.signal[[self.time_column, self.value_column]].copy().reset_index(drop=True)
    )
    window_td = pd.to_timedelta(window)
    threshold = baseline_rms * threshold_factor
    lo, hi = time_window_bounds(sig[self.time_column], window_td)
    values = sig[self.value_column].to_numpy(dtype=float)
    mom = window_moments(np.zeros(len(sig)), values**2, lo, hi)
    sig["rms"] = np.sqrt(np.where(mom["n"] < mom["n_total"], np.nan, mom["mean"]))
    exceeded = (sig["rms"] >= threshold).fillna(False)
    if not exceeded.any():
        return pd.DataFrame(columns=cols)
    group_id = (exceeded != exceeded.shift()).cumsum()
    ends = interval_end_times(sig[self.time_column], interval_end)
    events: list[dict[str, Any]] = []
    for _, seg in sig.groupby(group_id):
        seg_exc = exceeded.loc[seg.index]
        if not seg_exc.iloc[0]:
            continue
        start_time = seg[self.time_column].iloc[0]
        end_time = ends.loc[seg.index[-1]]
        avg_rms = float(seg["rms"].mean())
        events.append(
            {
                "start": start_time,
                "end": end_time,
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": True,
                "rms_value": avg_rms,
                "baseline_rms": baseline_rms,
                "ratio": avg_rms / baseline_rms if baseline_rms > 0 else None,
                "duration_seconds": (end_time - start_time).total_seconds(),
            }
        )
    return pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)


def _ref_inventorymonitoring_detect_low_stock(
    self, min_level: float, hold: str = "0s", *, interval_end: str = "next_change"
) -> pd.DataFrame:
    empty_cols = [
        "start",
        "end",
        "uuid",
        "source_uuid",
        "is_delta",
        "min_value",
        "avg_value",
        "duration_seconds",
    ]
    if self.levels.empty:
        return pd.DataFrame(columns=empty_cols)
    hold_td = pd.to_timedelta(hold)
    lv = (
        self.levels[[self.time_column, self.value_column]].copy().reset_index(drop=True)
    )
    lv["below"] = lv[self.value_column] < min_level
    ends = interval_end_times(lv[self.time_column], interval_end)
    lv["group"] = (lv["below"] != lv["below"].shift()).cumsum()
    rows: list[dict[str, Any]] = []
    for _, seg in lv[lv["below"]].groupby("group"):
        start = seg[self.time_column].iloc[0]
        end = ends.loc[seg.index[-1]]
        duration = (end - start).total_seconds()
        if end - start < hold_td:
            continue
        rows.append(
            {
                "start": start,
                "end": end,
                "uuid": self.event_uuid,
                "source_uuid": self.level_uuid,
                "is_delta": True,
                "min_value": float(seg[self.value_column].min()),
                "avg_value": float(seg[self.value_column].mean()),
                "duration_seconds": duration,
            }
        )
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows)


def _ref_signalquality_sampling_regularity(self, window: str = "1h") -> pd.DataFrame:
    cols = [
        "start",
        "mean_interval",
        "std_interval",
        "min_interval",
        "max_interval",
        "regularity_score",
    ]
    if self.signal.empty or len(self.signal) < 2:
        return finish_events(
            [],
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )
    sig = self.signal[[self.time_column]].copy().set_index(self.time_column)
    sig["interval"] = sig.index.to_series().diff().dt.total_seconds()
    sig = sig.dropna(subset=["interval"])
    if sig.empty:
        return finish_events(
            [],
            cols,
            "summary",
            uuid=self.event_uuid,
            source_uuid=self.signal_uuid,
            window=window,
        )
    resampled = (
        sig["interval"].resample(window).agg(["mean", "std", "min", "max", "count"])
    )
    events: list[dict[str, Any]] = []
    for ts, row in resampled.iterrows():
        if row["count"] < 2 or pd.isna(row["mean"]):
            continue
        mean_val = row["mean"]
        std_val = row["std"] if pd.notna(row["std"]) else 0.0
        regularity = max(0.0, 1.0 - std_val / (mean_val + 1e-10))
        events.append(
            {
                "start": ts,
                "mean_interval": float(mean_val),
                "std_interval": float(std_val),
                "min_interval": float(row["min"]),
                "max_interval": float(row["max"]),
                "regularity_score": max(0.0, min(1.0, regularity)),
            }
        )
    return finish_events(
        events,
        cols,
        "summary",
        uuid=self.event_uuid,
        source_uuid=self.signal_uuid,
        window=window,
    )


def _ref_signalquality_detect_out_of_range(
    self, min_value: float, max_value: float, *, interval_end: str = "next_change"
) -> pd.DataFrame:
    cols = ["start", "end", "duration", "min_observed", "max_observed", "direction"]
    if self.signal.empty:
        return finish_events(
            [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
    sig = (
        self.signal[[self.time_column, self.value_column]].copy().reset_index(drop=True)
    )
    out_of_range = (sig[self.value_column] < min_value) | (
        sig[self.value_column] > max_value
    )
    if not out_of_range.any():
        return finish_events(
            [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
    groups = (out_of_range != out_of_range.shift()).cumsum()
    ends = interval_end_times(sig[self.time_column], interval_end)
    events: list[dict[str, Any]] = []
    for _, seg in sig.groupby(groups):
        seg_oor = out_of_range.loc[seg.index]
        if not seg_oor.iloc[0]:
            continue
        start = seg[self.time_column].iloc[0]
        end = ends.loc[seg.index[-1]]
        min_obs = float(seg[self.value_column].min())
        max_obs = float(seg[self.value_column].max())
        if max_obs > max_value:
            direction = "above"
        elif min_obs < min_value:
            direction = "below"
        else:
            direction = "both"
        events.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "min_observed": min_obs,
                "max_observed": max_obs,
                "direction": direction,
            }
        )
    return finish_events(
        events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
    )


def _ref_alarmmanagement_intervalize(self) -> pd.DataFrame:
    if self.series.empty:
        return pd.DataFrame(columns=["start", "end", "duration_seconds"])
    s = self.series[[self.time_column, self.value_column]].copy().reset_index(drop=True)
    s["state"] = bool_or_false(s[self.value_column])
    s["group"] = (s["state"] != s["state"].shift()).cumsum()
    rows: list[dict[str, Any]] = []
    groups = list(s.groupby("group"))
    for idx, (_, seg) in enumerate(groups):
        if not seg["state"].iloc[0]:
            continue
        start = seg[self.time_column].iloc[0]
        if idx + 1 < len(groups):
            _, next_seg = groups[idx + 1]
            end = next_seg[self.time_column].iloc[0]
        else:
            end = seg[self.time_column].iloc[-1]
        rows.append(
            {
                "start": start,
                "end": end,
                "duration_seconds": (end - start).total_seconds(),
            }
        )
    return pd.DataFrame(rows)


def _ref_alarmmanagement_chattering_detection(
    self, min_transitions: int = 5, window: str = "10min"
) -> pd.DataFrame:
    if self.series.empty or len(self.series) < 2:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration_seconds",
                "transition_count",
                "uuid",
                "source_uuid",
            ]
        )
    s = self.series[[self.time_column, self.value_column]].copy()
    s["state"] = bool_or_false(s[self.value_column])
    s["transition"] = (s["state"] != s["state"].shift()).astype(int)
    s.iloc[0, s.columns.get_loc("transition")] = 0
    s = s.set_index(self.time_column)
    window_td = pd.to_timedelta(window)
    rolling_counts = s["transition"].rolling(window_td, min_periods=1).sum()
    flagged = rolling_counts[rolling_counts >= min_transitions]
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration_seconds",
                "transition_count",
                "uuid",
                "source_uuid",
            ]
        )
    flagged_times = flagged.index.to_series().reset_index(drop=True)
    transition_counts = flagged.values
    rows: list[dict[str, Any]] = []
    start = flagged_times.iloc[0]
    max_count = int(transition_counts[0])
    for i in range(1, len(flagged_times)):
        gap = flagged_times.iloc[i] - flagged_times.iloc[i - 1]
        if gap > window_td:
            rows.append(
                {
                    "start": start,
                    "end": flagged_times.iloc[i - 1],
                    "transition_count": max_count,
                    "uuid": self.event_uuid,
                    "source_uuid": self.alarm_uuid,
                }
            )
            start = flagged_times.iloc[i]
            max_count = int(transition_counts[i])
        else:
            max_count = max(max_count, int(transition_counts[i]))
    rows.append(
        {
            "start": start,
            "end": flagged_times.iloc[-1],
            "transition_count": max_count,
            "uuid": self.event_uuid,
            "source_uuid": self.alarm_uuid,
        }
    )
    out = finalize_interval_df(
        pd.DataFrame(rows), uuid=self.event_uuid, source_uuid=self.alarm_uuid
    )
    cols = ["start", "end", "duration_seconds", "transition_count"]
    return out[cols + ["uuid", "source_uuid"]]


def _ref_batchtracking_detect_batches(self) -> pd.DataFrame:
    if self.series.empty:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "start",
                "end",
                "duration_seconds",
                "sample_count",
                "uuid",
                "source_uuid",
            ]
        )
    s = self._segments()
    rows: list[dict[str, Any]] = []
    for _, seg in s.groupby("group"):
        batch_id = seg["batch_id"].iloc[0]
        if batch_id == "":
            continue
        start = seg[self.time_column].iloc[0]
        end = seg[self.time_column].iloc[-1]
        rows.append(
            {
                "batch_id": batch_id,
                "start": start,
                "end": end,
                "duration_seconds": (end - start).total_seconds(),
                "sample_count": len(seg),
                "uuid": self.event_uuid,
                "source_uuid": self.batch_uuid,
            }
        )
    return pd.DataFrame(rows)


def _ref_valuetraceability_detect_intervals(self, uuid: str) -> pd.DataFrame:
    sdf = self._station_data.get(uuid, pd.DataFrame())
    if sdf.empty:
        return pd.DataFrame(
            columns=[
                "identifier",
                "station_uuid",
                "station_name",
                "start",
                "end",
                "duration_seconds",
                "sample_count",
            ]
        )
    s = sdf[[self.time_column, self.value_column]].copy()
    s["identifier"] = s[self.value_column].fillna("")
    s["group"] = (s["identifier"] != s["identifier"].shift()).cumsum()
    station_name = self.station_uuids[uuid]
    rows: list[dict[str, Any]] = []
    for _, seg in s.groupby("group"):
        identifier = seg["identifier"].iloc[0]
        if identifier == "":
            continue
        start = seg[self.time_column].iloc[0]
        end = seg[self.time_column].iloc[-1]
        rows.append(
            {
                "identifier": identifier,
                "station_uuid": uuid,
                "station_name": station_name,
                "start": start,
                "end": end,
                "duration_seconds": (end - start).total_seconds(),
                "sample_count": len(seg),
            }
        )
    return pd.DataFrame(rows)


def _ref_multiprocesstraceability_detect_intervals(
    self, id_uuid: str, station_name: str
) -> pd.DataFrame:
    sdf = self._process_data.get(id_uuid, pd.DataFrame())
    if sdf.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "station",
                "id_uuid",
                "start",
                "end",
                "duration_seconds",
                "sample_count",
            ]
        )
    s = sdf[[self.time_column, self.value_column_id]].copy()
    s["item_id"] = s[self.value_column_id].fillna("")
    value_change = s["item_id"] != s["item_id"].shift()
    dt = s[self.time_column].diff()
    median_dt = dt.median()
    if pd.notna(median_dt) and median_dt > pd.Timedelta(0):
        gap = dt > median_dt * 3
    else:
        gap = pd.Series(False, index=s.index)
    s["group"] = (value_change | gap).cumsum()
    rows: list[dict[str, Any]] = []
    for _, seg in s.groupby("group"):
        item_id = seg["item_id"].iloc[0]
        if item_id == "":
            continue
        start = seg[self.time_column].iloc[0]
        end = seg[self.time_column].iloc[-1]
        rows.append(
            {
                "item_id": item_id,
                "station": station_name,
                "id_uuid": id_uuid,
                "start": start,
                "end": end,
                "duration_seconds": (end - start).total_seconds(),
                "sample_count": len(seg),
            }
        )
    return pd.DataFrame(rows)


def _ref_multiprocesstraceability_build_timeline(
    self, *, infer_dwell_from_next: bool = False
) -> pd.DataFrame:
    all_intervals: list[pd.DataFrame] = []
    for proc in self.processes:
        intervals = _ref_multiprocesstraceability_detect_intervals(
            self, proc["id_uuid"], proc["station"]
        )
        if not intervals.empty:
            all_intervals.append(intervals)
    if not all_intervals:
        return pd.DataFrame(
            columns=[
                "item_id",
                "station",
                "id_uuid",
                "start",
                "end",
                "duration_seconds",
                "sample_count",
                "step_sequence",
                "is_parallel",
                "uuid",
            ]
        )
    timeline = pd.concat(all_intervals, ignore_index=True)
    timeline = timeline.sort_values(["item_id", "start"]).reset_index(drop=True)
    timeline["step_sequence"] = timeline.groupby("item_id").cumcount() + 1
    timeline["is_parallel"] = False
    for _, grp in timeline.groupby("item_id"):
        if len(grp) < 2:
            continue
        idxs = grp.index.tolist()
        for i, idx_i in enumerate(idxs):
            for idx_j in idxs[i + 1 :]:
                row_i = timeline.loc[idx_i]
                row_j = timeline.loc[idx_j]
                if row_i["start"] < row_j["end"] and row_j["start"] < row_i["end"]:
                    timeline.loc[idx_i, "is_parallel"] = True
                    timeline.loc[idx_j, "is_parallel"] = True
    timeline["uuid"] = self.event_uuid
    if infer_dwell_from_next:
        next_start = timeline.groupby("item_id")["start"].shift(-1)
        infer = ~timeline["is_parallel"] & next_start.notna()
        timeline.loc[infer, "end"] = next_start[infer]
        timeline["duration_seconds"] = (
            timeline["end"] - timeline["start"]
        ).dt.total_seconds()
    return timeline


def _ref_multiprocesstraceability_parallel_activity(self) -> pd.DataFrame:
    timeline = _ref_multiprocesstraceability_build_timeline(
        self,
    )
    if timeline.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "station_a",
                "station_b",
                "overlap_start",
                "overlap_end",
                "overlap_seconds",
                "uuid",
            ]
        )
    rows: list[dict[str, Any]] = []
    for item_id, grp in timeline.groupby("item_id"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("start")
        records = grp.to_dict("records")
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ri, rj = (records[i], records[j])
                overlap_start = max(ri["start"], rj["start"])
                overlap_end = min(ri["end"], rj["end"])
                if overlap_start < overlap_end:
                    rows.append(
                        {
                            "item_id": item_id,
                            "station_a": ri["station"],
                            "station_b": rj["station"],
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_seconds": (
                                overlap_end - overlap_start
                            ).total_seconds(),
                            "uuid": self.event_uuid,
                        }
                    )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[
                "item_id",
                "station_a",
                "station_b",
                "overlap_start",
                "overlap_end",
                "overlap_seconds",
                "uuid",
            ]
        )
    )


def _ref_flowconstraint_blocked_events(
    self,
    *,
    roles: dict[str, str],
    tolerance: str = "200ms",
    tolerance_before: str | None = None,
    tolerance_after: str | None = None,
    min_duration: str = "0s",
    interval_end: str = "next_change",
) -> pd.DataFrame:
    _check_interval_end(interval_end)
    up = self._align_bool(roles["upstream_run"])
    dn = self._align_bool(roles["downstream_run"])
    if up.empty or dn.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "uuid",
                "source_uuid",
                "is_delta",
                "type",
                "time_alignment_quality",
                "duration",
                "severity",
            ]
        )
    tol_before = (
        pd.to_timedelta(tolerance_before)
        if tolerance_before
        else pd.to_timedelta(tolerance)
    )
    tol_after = (
        pd.to_timedelta(tolerance_after)
        if tolerance_after
        else pd.to_timedelta(tolerance)
    )
    max_tol = max(tol_before, tol_after)
    dn = dn.assign(**{self.time_column + "_dn": dn[self.time_column]})
    merged = pd.merge_asof(
        up,
        dn,
        on=self.time_column,
        suffixes=("_up", "_dn"),
        tolerance=max_tol,
        direction="nearest",
    )
    merged["time_up"] = up[self.time_column].array
    if tolerance_before or tolerance_after:
        time_diff = merged[self.time_column + "_dn"] - merged[self.time_column]
        valid_mask = (time_diff <= pd.Timedelta(0)) & (
            time_diff.abs() <= tol_before
        ) | (time_diff >= pd.Timedelta(0)) & (time_diff <= tol_after)
        merged["state_dn"] = merged["state_dn"].astype("boolean").where(valid_mask)
    alignment_quality = (
        merged["state_dn"].notna().sum() / len(merged) if len(merged) > 0 else 0.0
    )
    cond = merged["state_up"] & ~merged["state_dn"].fillna(False)
    gid = cond.ne(cond.shift()).cumsum()
    min_td = pd.to_timedelta(min_duration)
    rows: list[dict[str, Any]] = []
    ends = interval_end_times(merged[self.time_column], interval_end)
    for _, seg in merged.groupby(gid):
        m = cond.loc[seg.index]
        if not m.any():
            continue
        start = seg[self.time_column].iloc[0]
        end = ends.loc[seg.index[-1]]
        duration = end - start
        if duration < min_td:
            continue
        severity = self._calculate_severity(duration)
        rows.append(
            {
                "start": start,
                "end": end,
                "uuid": self.event_uuid,
                "source_uuid": roles["upstream_run"],
                "is_delta": True,
                "type": "blocked",
                "time_alignment_quality": alignment_quality,
                "duration": duration,
                "severity": severity,
            }
        )
    return pd.DataFrame(rows)


def _ref_flowconstraint_starved_events(
    self,
    *,
    roles: dict[str, str],
    tolerance: str = "200ms",
    tolerance_before: str | None = None,
    tolerance_after: str | None = None,
    min_duration: str = "0s",
    interval_end: str = "next_change",
) -> pd.DataFrame:
    _check_interval_end(interval_end)
    up = self._align_bool(roles["upstream_run"])
    dn = self._align_bool(roles["downstream_run"])
    if up.empty or dn.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "uuid",
                "source_uuid",
                "is_delta",
                "type",
                "time_alignment_quality",
                "duration",
                "severity",
            ]
        )
    tol_before = (
        pd.to_timedelta(tolerance_before)
        if tolerance_before
        else pd.to_timedelta(tolerance)
    )
    tol_after = (
        pd.to_timedelta(tolerance_after)
        if tolerance_after
        else pd.to_timedelta(tolerance)
    )
    max_tol = max(tol_before, tol_after)
    up = up.assign(**{self.time_column + "_up": up[self.time_column]})
    merged = pd.merge_asof(
        dn,
        up,
        on=self.time_column,
        suffixes=("_dn", "_up"),
        tolerance=max_tol,
        direction="nearest",
    )
    merged["time_dn"] = dn[self.time_column].array
    if tolerance_before or tolerance_after:
        time_diff = merged[self.time_column + "_up"] - merged[self.time_column]
        valid_mask = (time_diff <= pd.Timedelta(0)) & (
            time_diff.abs() <= tol_before
        ) | (time_diff >= pd.Timedelta(0)) & (time_diff <= tol_after)
        merged["state_up"] = merged["state_up"].astype("boolean").where(valid_mask)
    alignment_quality = (
        merged["state_up"].notna().sum() / len(merged) if len(merged) > 0 else 0.0
    )
    cond = merged["state_dn"] & ~merged["state_up"].fillna(False)
    gid = cond.ne(cond.shift()).cumsum()
    min_td = pd.to_timedelta(min_duration)
    rows: list[dict[str, Any]] = []
    ends = interval_end_times(merged[self.time_column], interval_end)
    for _, seg in merged.groupby(gid):
        m = cond.loc[seg.index]
        if not m.any():
            continue
        start = seg[self.time_column].iloc[0]
        end = ends.loc[seg.index[-1]]
        duration = end - start
        if duration < min_td:
            continue
        severity = self._calculate_severity(duration)
        rows.append(
            {
                "start": start,
                "end": end,
                "uuid": self.event_uuid,
                "source_uuid": roles["downstream_run"],
                "is_delta": True,
                "type": "starved",
                "time_alignment_quality": alignment_quality,
                "duration": duration,
                "severity": severity,
            }
        )
    return pd.DataFrame(rows)


def _ref_changeover_compute_stable_band_end(
    self, t0: pd.Timestamp, config: dict[str, Any]
) -> pd.Timestamp | None:
    metric_defs = config.get("metrics", [])
    reference_method = config.get("reference_method", "expanding_median")
    metric_ends: list[pd.Timestamp] = []
    for mdef in metric_defs:
        uid = mdef["uuid"]
        vcol = mdef.get("value_column", "value_double")
        band = float(mdef.get("band", 0.0))
        hold_td = pd.to_timedelta(mdef.get("hold", "0s"))
        s = (
            self.dataframe[self.dataframe["uuid"] == uid]
            .copy()
            .sort_values(self.time_column)
        )
        s[self.time_column] = pd.to_datetime(s[self.time_column])
        s = s[s[self.time_column] >= t0]
        if s.empty:
            continue
        ref = self._calculate_reference(s[vcol], reference_method, config, mdef)
        inside = (s[vcol] - ref).abs() <= band
        if not inside.any():
            continue
        gid = inside.ne(inside.shift()).cumsum()
        end_found: pd.Timestamp | None = None
        for _, seg in s.groupby(gid):
            seg_inside = inside.loc[seg.index]
            if not seg_inside.iloc[0]:
                continue
            start_seg = seg[self.time_column].iloc[0]
            end_seg = seg[self.time_column].iloc[-1]
            if end_seg - start_seg >= hold_td:
                end_found = start_seg
                break
        if end_found is not None:
            metric_ends.append(end_found)
    if metric_defs and len(metric_ends) == len(metric_defs):
        return max(metric_ends)
    return None


def _ref_startupdetection_detect_startup_by_threshold(
    self,
    *,
    threshold: float,
    hysteresis: tuple[float, float] | None = None,
    min_above: str = "0s",
) -> pd.DataFrame:
    cols = ["start", "end", "uuid", "is_delta", "method", "threshold"]
    if self.series.empty:
        return self._intervals([], cols)
    enter_thr = threshold if hysteresis is None else hysteresis[0]
    exit_thr = threshold if hysteresis is None else hysteresis[1]
    min_above_td = pd.to_timedelta(min_above)
    s = self.series[[self.time_column, self.value_column]].copy()
    above_enter = s[self.value_column] >= enter_thr
    rising = ~above_enter.shift(fill_value=False) & above_enter
    rise_times = s.loc[rising, self.time_column]
    events: list[dict[str, Any]] = []
    for t0 in rise_times:
        win = _between(s, self.time_column, t0, t0 + min_above_td)
        if win.empty:
            continue
        if (win[self.value_column] >= exit_thr).all():
            events.append(
                {
                    "start": t0,
                    "end": t0 + min_above_td,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "method": "threshold",
                    "threshold": float(threshold),
                }
            )
    return self._intervals(events, cols)


def _ref_startupdetection_detect_startup_by_slope(
    self, *, min_slope: float, slope_window: str = "0s", min_duration: str = "0s"
) -> pd.DataFrame:
    cols = ["start", "end", "uuid", "is_delta", "method", "min_slope", "avg_slope"]
    if self.series.empty:
        return self._intervals([], cols)
    s = self.series[[self.time_column, self.value_column]].copy()
    win = pd.to_timedelta(slope_window)
    if win > pd.Timedelta(0):
        s = s.reset_index(drop=True)
        probe = pd.DataFrame(
            {"_t": s[self.time_column] - win, "_row": np.arange(len(s))}
        )
        ref = s[[self.time_column, self.value_column]].rename(
            columns={self.time_column: "_t", self.value_column: "_v0"}
        )
        ref["_t0"] = ref["_t"]
        back = pd.merge_asof(probe, ref, on="_t", direction="backward").sort_values(
            "_row"
        )
        s["_t_ref"] = back["_t0"].array
        s["dt_s"] = (s[self.time_column] - s["_t_ref"]).dt.total_seconds()
        s["dv"] = s[self.value_column] - back["_v0"].to_numpy()
    else:
        s["_t_ref"] = s[self.time_column].shift()
        s["dt_s"] = s[self.time_column].diff().dt.total_seconds()
        s["dv"] = s[self.value_column].diff()
    dup = s["dt_s"] <= 0
    if dup.any():
        warnings.warn(
            f"detect_startup_by_slope(): {int(dup.sum())} sample(s) share a timestamp with the previous sample; their slope is undefined and they are not counted as ramping",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    s["slope"] = s["dv"] / s["dt_s"].where(~dup)
    mask = s["slope"] >= float(min_slope)
    gid = (mask != mask.shift()).cumsum()
    min_d = pd.to_timedelta(min_duration)
    events: list[dict[str, Any]] = []
    for _, seg in s.groupby(gid):
        seg_mask = mask.loc[seg.index]
        if not seg_mask.any():
            continue
        # A run starts at the reference sample of its first ramping sample.
        start_t = seg.loc[seg_mask, "_t_ref"].iloc[0]
        end_t = seg.loc[seg_mask, self.time_column].iloc[-1]
        if end_t - start_t < min_d:
            continue
        avg_slope = seg.loc[seg_mask, "slope"].mean()
        events.append(
            {
                "start": start_t,
                "end": end_t,
                "uuid": self.event_uuid,
                "is_delta": True,
                "method": "slope",
                "min_slope": float(min_slope),
                "avg_slope": float(avg_slope) if pd.notna(avg_slope) else None,
            }
        )
    return self._intervals(events, cols)


def _ref_datagapanalysis_detect_gaps(self, min_gap: str = "5s") -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "gap_duration_seconds",
        "samples_before_gap",
        "samples_after_gap",
    ]
    if self.signal.empty or len(self.signal) < 2:
        return finish_events(
            [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
    threshold = pd.to_timedelta(min_gap)
    times = self.signal[self.time_column].array
    diffs = times[1:] - times[:-1]
    events: list[dict[str, Any]] = []
    for i, d in enumerate(diffs):
        gap = pd.Timedelta(d)
        if gap > threshold and gap > pd.Timedelta(0):  # strictly longer
            events.append(
                {
                    "start": pd.Timestamp(times[i]),
                    "end": pd.Timestamp(times[i + 1]),
                    "gap_duration_seconds": float(gap.total_seconds()),
                    "samples_before_gap": i + 1,
                    "samples_after_gap": len(times) - (i + 1),
                }
            )
    return finish_events(
        events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
    )


def _ref_datagapanalysis_interpolation_candidates(
    self, max_gap: str = "10s", min_gap: str = "0s"
) -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "gap_duration_seconds",
        "value_before",
        "value_after",
        "value_jump",
        "safe_to_interpolate",
    ]
    if self.signal.empty or len(self.signal) < 2:
        return finish_events(
            [], cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
        )
    max_td = pd.to_timedelta(max_gap)
    min_td = pd.to_timedelta(min_gap)
    times = self.signal[self.time_column].array
    values = self.signal[self.value_column].values
    diffs = times[1:] - times[:-1]
    signal_std = float(np.nanstd(values))
    safe_threshold = 2.0 * signal_std if signal_std > 0 else float("inf")
    events: list[dict[str, Any]] = []
    for i, d in enumerate(diffs):
        gap = pd.Timedelta(d)
        if gap <= pd.Timedelta(0) or gap < min_td or gap > max_td:
            continue
        val_before = float(values[i]) if not np.isnan(values[i]) else None
        val_after = float(values[i + 1]) if not np.isnan(values[i + 1]) else None
        if val_before is not None and val_after is not None:
            jump = abs(val_after - val_before)
        else:
            jump = float("nan")
        safe = not np.isnan(jump) and jump <= safe_threshold
        events.append(
            {
                "start": pd.Timestamp(times[i]),
                "end": pd.Timestamp(times[i + 1]),
                "gap_duration_seconds": float(gap.total_seconds()),
                "value_before": val_before,
                "value_after": val_after,
                "value_jump": float(jump) if not np.isnan(jump) else None,
                "safe_to_interpolate": safe,
            }
        )
    return finish_events(
        events, cols, "interval", uuid=self.event_uuid, source_uuid=self.signal_uuid
    )


def _ref_thresholdmonitoring_time_above_threshold(
    self, threshold: float, window: str = "1h", *, interval_end: str = "next_change"
) -> pd.DataFrame:
    cols = ["start", "time_above", "pct_above", "exceedance_count"]
    if self.signal.empty:
        return empty_event_df("summary", cols[1:])
    sig = self.signal[[self.time_column, self.value_column]].copy()
    sig = sig.set_index(self.time_column)
    sig["above"] = (sig[self.value_column] >= threshold).astype(float)
    sig["enter_above"] = (sig["above"].diff().fillna(sig["above"]) == 1).astype(int)
    window_td = pd.to_timedelta(window)
    resampled = sig.resample(window).agg(
        above_frac=("above", "mean"),
        exceedance_count=("enter_above", "sum"),
        sample_count=("above", "count"),
    )
    if interval_end not in INTERVAL_END_MODES:
        raise ValueError(
            f"interval_end must be one of {INTERVAL_END_MODES}, got {interval_end!r}"
        )
    if interval_end == "next_change":
        resampled["above_frac"] = _ref_thresholdmonitoring_time_weighted_above_fraction(
            sig["above"], resampled.index, window_td
        )
    events: list[dict[str, Any]] = []
    for ts, row in resampled.iterrows():
        if row["sample_count"] == 0 and interval_end != "next_change":
            continue
        pct = float(row["above_frac"] * 100)
        time_above_seconds = row["above_frac"] * window_td.total_seconds()
        events.append(
            {
                "start": ts,
                "time_above": float(time_above_seconds),
                "pct_above": pct,
                "exceedance_count": int(row["exceedance_count"]),
            }
        )
    if not events:
        return empty_event_df("summary", cols[1:])
    return finalize_summary_df(
        add_window_bounds(pd.DataFrame(events, columns=cols), window)
    )


def _ref_thresholdmonitoring_time_weighted_above_fraction(
    above: pd.Series, window_starts: pd.Index, window_td: pd.Timedelta
) -> list[float]:
    times = above.index.to_series().reset_index(drop=True)
    seg_end = times.shift(-1).fillna(times)
    flags = above.reset_index(drop=True).astype(bool)
    starts, stops = (times[flags], seg_end[flags])
    fractions: list[float] = []
    for w_start in window_starts:
        w_end = w_start + window_td
        overlap = (
            stops.clip(upper=w_end) - starts.clip(lower=w_start)
        ).dt.total_seconds()
        fractions.append(float(overlap.clip(lower=0).sum()) / window_td.total_seconds())
    return fractions


def _ref_machinestate_detect_rapid_transitions(
    self, threshold: str = "5s", min_count: int = 3
) -> pd.DataFrame:
    cols = [
        "systime",
        "uuid",
        "source_uuid",
        "start",
        "end",
        "transition_count",
        "duration_seconds",
    ]
    transitions = self.transition_events()
    if transitions.empty or len(transitions) < min_count:
        return pd.DataFrame(columns=cols)
    threshold_td = pd.to_timedelta(threshold)
    rapid_events: list[dict[str, Any]] = []
    for i in range(len(transitions) - min_count + 1):
        start = transitions.iloc[i]["systime"]
        for j in range(i + min_count - 1, len(transitions)):
            end = transitions.iloc[j]["systime"]
            duration = end - start
            if duration <= threshold_td:
                transition_count = j - i + 1
                rapid_events.append(
                    {
                        "start": start,
                        "end": end,
                        "transition_count": transition_count,
                        "duration_seconds": duration.total_seconds(),
                    }
                )
            else:
                break
    if not rapid_events:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rapid_events)
    out.insert(0, "systime", out["start"])
    out.insert(1, "uuid", self.event_uuid)
    out.insert(2, "source_uuid", self.run_state_uuid)
    return out


def _ref_warmupcooldown_detect_ramps(
    self,
    direction: str,
    min_change: float,
    min_duration: str = "1min",
    max_pause: int = 0,
) -> list[dict[str, Any]]:
    if self.signal.empty or len(self.signal) < 3:
        return []
    min_td = pd.Timedelta(min_duration)
    sig = (
        self.signal[[self.time_column, self.value_column]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(sig) < 3:
        return []
    vals = sig[self.value_column].values
    times = sig[self.time_column].array
    diff = np.diff(vals)
    if direction == "rising":
        moving = diff > 0
    else:
        moving = diff < 0
    if max_pause > 0:
        moving = _bridge_pauses(moving, max_pause)
    intervals: list[dict[str, Any]] = []
    in_ramp = False
    start_idx = 0
    for i in range(len(moving)):
        if moving[i] and (not in_ramp):
            start_idx = i
            in_ramp = True
        elif not moving[i] and in_ramp:
            end_idx = i
            in_ramp = False
            s_time = pd.Timestamp(times[start_idx])
            e_time = pd.Timestamp(times[end_idx])
            s_val = float(vals[start_idx])
            e_val = float(vals[end_idx])
            change = abs(e_val - s_val)
            dur = e_time - s_time
            if change >= min_change and dur >= min_td:
                dur_s = dur.total_seconds()
                intervals.append(
                    {
                        "start": s_time,
                        "end": e_time,
                        "start_value": s_val,
                        "end_value": e_val,
                        "change": change,
                        "duration_seconds": dur_s,
                        "avg_rate": change / dur_s if dur_s > 0 else 0.0,
                    }
                )
    if in_ramp:
        end_idx = len(vals) - 1
        s_time = pd.Timestamp(times[start_idx])
        e_time = pd.Timestamp(times[end_idx])
        s_val = float(vals[start_idx])
        e_val = float(vals[end_idx])
        change = abs(e_val - s_val)
        dur = e_time - s_time
        if change >= min_change and dur >= min_td:
            dur_s = dur.total_seconds()
            intervals.append(
                {
                    "start": s_time,
                    "end": e_time,
                    "start_value": s_val,
                    "end_value": e_val,
                    "change": change,
                    "duration_seconds": dur_s,
                    "avg_rate": change / dur_s if dur_s > 0 else 0.0,
                }
            )
    return intervals


def _ref_warmupcooldown_detect_warmup(
    self, min_rise: float, min_duration: str = "1min", *, max_pause: int = 0
) -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "uuid",
        "source_uuid",
        "is_delta",
        "start_value",
        "end_value",
        "rise",
        "duration_seconds",
        "avg_rate",
    ]
    ramps = _ref_warmupcooldown_detect_ramps(
        self, "rising", min_rise, min_duration, max_pause
    )
    if not ramps:
        return pd.DataFrame(columns=cols)
    events: list[dict[str, Any]] = []
    for r in ramps:
        events.append(
            {
                "start": r["start"],
                "end": r["end"],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": False,
                "start_value": r["start_value"],
                "end_value": r["end_value"],
                "rise": r["change"],
                "duration_seconds": r["duration_seconds"],
                "avg_rate": r["avg_rate"],
            }
        )
    return pd.DataFrame(events, columns=cols)


def _ref_warmupcooldown_detect_cooldown(
    self, min_fall: float, min_duration: str = "1min", *, max_pause: int = 0
) -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "uuid",
        "source_uuid",
        "is_delta",
        "start_value",
        "end_value",
        "fall",
        "duration_seconds",
        "avg_rate",
    ]
    ramps = _ref_warmupcooldown_detect_ramps(
        self, "falling", min_fall, min_duration, max_pause
    )
    if not ramps:
        return pd.DataFrame(columns=cols)
    events: list[dict[str, Any]] = []
    for r in ramps:
        events.append(
            {
                "start": r["start"],
                "end": r["end"],
                "uuid": self.event_uuid,
                "source_uuid": self.signal_uuid,
                "is_delta": False,
                "start_value": r["start_value"],
                "end_value": r["end_value"],
                "fall": r["change"],
                "duration_seconds": r["duration_seconds"],
                "avg_rate": r["avg_rate"],
            }
        )
    return pd.DataFrame(events, columns=cols)


def _ref_operatingstate_unusual_state_sequence(
    self, n_states: int = 2, expected_order: list[int] | None = None
) -> pd.DataFrame:
    cols = [
        "transition_time",
        "from_state",
        "to_state",
        "from_label",
        "to_label",
        "is_unusual",
    ]
    intervals = self.infer_states(n_states=n_states)
    if len(intervals) < 2:
        return pd.DataFrame(columns=cols)
    labels = self._state_labels(n_states)
    states = intervals["state"].tolist()
    if expected_order is not None:
        normal_transitions = set()
        for i in range(len(expected_order) - 1):
            normal_transitions.add((expected_order[i], expected_order[i + 1]))
    else:
        transition_counts: dict[tuple, int] = {}
        for i in range(len(states) - 1):
            pair = (states[i], states[i + 1])
            transition_counts[pair] = transition_counts.get(pair, 0) + 1
        total = sum(transition_counts.values())
        threshold = max(1, total * 0.1)
        normal_transitions = {
            pair for pair, count in transition_counts.items() if count >= threshold
        }

    def _label(state_id: int) -> str:
        return labels[state_id] if state_id < len(labels) else f"state_{state_id}"

    from_states = [int(x) for x in states[:-1]]
    to_states = [int(x) for x in states[1:]]
    return pd.DataFrame(
        {
            "transition_time": intervals["start"].iloc[1:].reset_index(drop=True),
            "from_state": from_states,
            "to_state": to_states,
            "from_label": [_label(x) for x in from_states],
            "to_label": [_label(x) for x in to_states],
            "is_unusual": [
                (a, b) not in normal_transitions
                for a, b in zip(from_states, to_states, strict=True)
            ],
        },
        columns=cols,
    )


def _ref_operatingstate_dwell_time_violation(
    self, state: int, max_duration: str, n_states: int = 2
) -> pd.DataFrame:
    cols = [
        "start",
        "end",
        "state_label",
        "duration_seconds",
        "excess_seconds",
        "uuid",
        "source_uuid",
    ]
    intervals = self.infer_states(n_states=n_states)
    if intervals.empty:
        return pd.DataFrame(columns=cols)
    max_td = pd.to_timedelta(max_duration).total_seconds()
    state_intervals = intervals[intervals["state"] == state]
    events: list[dict[str, Any]] = []
    for _, row in state_intervals.iterrows():
        dur = row["duration_seconds"]
        if dur > max_td:
            events.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "state_label": row["state_label"],
                    "duration_seconds": dur,
                    "excess_seconds": dur - max_td,
                    "uuid": self.event_uuid,
                    "source_uuid": self.signal_uuid,
                }
            )
    return pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)


def _ref_routingtraceability_build_routing_timeline(
    self, *, infer_dwell_from_next: bool = False
) -> pd.DataFrame:
    if self.routing_data.empty or self.id_data.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "routing_value",
                "station_name",
                "start",
                "end",
                "duration_seconds",
                "sample_count",
                "station_sequence",
                "uuid",
                "source_uuid",
            ]
        )
    routing_subset = self.routing_data[
        [self.time_column, self.value_column_routing]
    ].copy()
    id_subset = self.id_data[[self.time_column, self.value_column_id]].copy()
    merged = pd.merge_asof(
        routing_subset, id_subset, on=self.time_column, direction="backward"
    )
    merged = merged.rename(
        columns={
            self.value_column_id: "item_id",
            self.value_column_routing: "routing_value",
        }
    )
    merged = merged.dropna(subset=["item_id"])
    merged["item_id"] = merged["item_id"].astype(str)
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "routing_value",
                "station_name",
                "start",
                "end",
                "duration_seconds",
                "sample_count",
                "station_sequence",
                "uuid",
                "source_uuid",
            ]
        )
    merged["combo"] = merged["item_id"] + "|" + merged["routing_value"].astype(str)
    merged["group"] = (merged["combo"] != merged["combo"].shift()).cumsum()
    n = len(merged)
    first_pos = np.flatnonzero(merged["group"].diff().fillna(1).to_numpy() != 0)
    last_pos = np.r_[first_pos[1:] - 1, n - 1]
    times = merged[self.time_column]
    start = times.iloc[first_pos].reset_index(drop=True)
    end = times.iloc[last_pos].reset_index(drop=True)
    routing_vals = merged["routing_value"].iloc[first_pos].tolist()
    timeline = pd.DataFrame(
        {
            "item_id": merged["item_id"].iloc[first_pos].tolist(),
            "routing_value": routing_vals,
            "station_name": self._resolve_states(routing_vals),
            "start": start,
            "end": end,
            "duration_seconds": (end - start).dt.total_seconds(),
            "sample_count": np.diff(np.r_[first_pos, n]).tolist(),
        }
    )
    timeline = timeline.sort_values(["item_id", "start"]).reset_index(drop=True)
    timeline["station_sequence"] = timeline.groupby("item_id").cumcount() + 1
    timeline["uuid"] = self.event_uuid
    timeline["source_uuid"] = self._source_uuid
    if infer_dwell_from_next:
        next_start = timeline.groupby("item_id")["start"].shift(-1)
        timeline["end"] = next_start.fillna(timeline["end"])
        timeline["duration_seconds"] = (
            timeline["end"] - timeline["start"]
        ).dt.total_seconds()
    return timeline


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _frame(seed, n, tz=None, nan_frac=0.0, dup_frac=0.0, ns=False, const=False):
    rng = np.random.default_rng(seed)
    parts = []

    def times(m):
        steps = rng.choice(
            [0, 1, 1, 1, 2, 5, 30, 400],
            size=m,
            p=[dup_frac, 0.4 - dup_frac, 0.2, 0.1, 0.1, 0.1, 0.07, 0.03],
        )
        t = pd.Timestamp("2024-03-30 22:00") + pd.to_timedelta(np.cumsum(steps), "s")
        if ns:
            t = t + pd.to_timedelta(rng.integers(0, 999, size=m), unit="ns")
        t = pd.DatetimeIndex(t)
        return t.tz_localize("UTC").tz_convert(tz) if tz else t

    def runs(m, choices, maxlen):
        out = []
        while len(out) < m:
            out += [choices[rng.integers(len(choices))]] * int(rng.integers(1, maxlen))
        return out[:m]

    def add(uuid, col, vals):
        d = pd.DataFrame({"systime": times(n), "uuid": uuid})
        for c in ["value_double", "value_bool", "value_string", "value_integer"]:
            d[c] = None
        d[col] = vals
        parts.append(d)

    walk = np.cumsum(rng.normal(size=n) * rng.choice([0.0, 1.0, 5.0], size=n))
    walk = walk + np.repeat(rng.normal(size=n // 7 + 1) * 3, 7)[:n]
    if const:
        walk = np.full(n, 3.0)
    if nan_frac:
        walk[rng.random(n) < nan_frac] = np.nan
    add("sig", "value_double", walk)
    scale = np.repeat(rng.choice([1, 4], size=n // 20 + 1), 20)[:n]
    add("sig2", "value_double", np.abs(rng.normal(size=n)) * scale)
    b = runs(n, [True, False], 12)
    if nan_frac:
        b = [None if rng.random() < nan_frac else x for x in b]
    add("b1", "value_bool", b)
    add("b2", "value_bool", runs(n, [True, False], 9))
    for k, u in enumerate(["id1", "id2", "id3"]):
        ids = ["A1", "A2", "B7", "", None, "C3"] + (["A1|x"] if k == 2 else [])
        add(u, "value_string", runs(n, ids, 15))
    add("route", "value_integer", runs(n, [1, 2, 3, 4], 10))
    add("ho", "value_integer", runs(n, [0, 0, 1], 3))
    df = pd.concat(parts, ignore_index=True)
    df["is_delta"] = True
    return df


FRAMES = {
    "naive": (0, 300, {}),
    "tz_nan_dup": (1, 400, {"tz": "UTC", "nan_frac": 0.08, "dup_frac": 0.1}),
    "berlin_dst": (2, 1500, {"tz": "Europe/Berlin", "dup_frac": 0.02}),
    "ns_times": (3, 300, {"ns": True, "dup_frac": 0.05}),
    "tiny": (5, 2, {}),
    "single": (4, 1, {}),
    "const": (7, 150, {"const": True}),
}


@pytest.fixture(scope="module")
def frames():
    return {k: _frame(seed, n, **kw) for k, (seed, n, kw) in FRAMES.items()}


def _same(new, ref):
    if isinstance(ref, pd.DataFrame):
        pd.testing.assert_frame_equal(new, ref, check_dtype=True, check_exact=True)
        for col in ref.columns:
            if ref[col].dtype.kind == "f":
                np.testing.assert_array_equal(
                    np.signbit(new[col].to_numpy()), np.signbit(ref[col].to_numpy())
                )
    else:
        assert type(new) is type(ref)
        assert repr(new) == repr(ref)


def _check(frames, make, new, ref):
    for name, df in frames.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                expected = ref(make(df))
            except Exception as exc:  # the reference raised: so must the new code
                with pytest.raises(type(exc)):
                    new(make(df))
                continue
            got = new(make(df))
        try:
            _same(got, expected)
        except AssertionError as exc:
            raise AssertionError(f"frame {name!r}: {exc}") from exc


def _changeover_with_ref(obj, config):
    """``changeover_window`` with the reference stable-band search."""
    original = type(obj)._compute_stable_band_end
    try:
        type(obj)._compute_stable_band_end = _ref_changeover_compute_stable_band_end
        return obj.changeover_window("id1", until="stable_band", config=config)
    finally:
        type(obj)._compute_stable_band_end = original


ROLES = {"upstream_run": "b1", "downstream_run": "b2"}
BAND = {
    "metrics": [
        {"uuid": "sig", "value_column": "value_double", "band": 0.5, "hold": "0s"},
        {"uuid": "sig2", "value_column": "value_double", "band": 2.0, "hold": "10s"},
    ]
}
PROCESSES = [
    {"id_uuid": "id1", "station": "W1"},
    {"id_uuid": "id2", "station": "W2"},
    {"id_uuid": "id3", "station": "P"},
]


def _routing(uuid, routing_uuid, column, state_map=None):
    return lambda df: RoutingTraceabilityEvents(
        df, uuid, routing_uuid, state_map=state_map, value_column_routing=column
    )


CASES = [
    # (id, constructor, new call, reference call)
    *[
        (
            f"rapid_change_{t}",
            lambda df: RateOfChangeEvents(df, "sig"),
            lambda o, t=t: o.detect_rapid_change(t),
            lambda o, t=t: _ref_rateofchange_detect_rapid_change(o, t),
        )
        for t in (0.0, 0.5, 2.0)
    ],
    *[
        (
            f"step_changes_{d}_{w}",
            lambda df: RateOfChangeEvents(df, "sig"),
            lambda o, d=d, w=w: o.detect_step_changes(d, max_duration=w),
            lambda o, d=d, w=w: _with_step_bounds(
                _ref_rateofchange_detect_step_changes(o, d, w)
            ),
        )
        for d, w in ((0.5, "5s"), (0.0, "0s"), (3.0, "1s"))
    ],
    *[
        (
            f"trend_{w}_{s}_{d}_{ie}",
            lambda df: DegradationDetectionEvents(df, "sig"),
            lambda o, w=w, s=s, d=d, ie=ie: o.detect_trend_degradation(
                window=w, min_slope=s, direction=d, interval_end=ie
            ),
            lambda o, w=w, s=s, d=d, ie=ie: (
                _ref_degradationdetection_detect_trend_degradation(
                    o, w, s, d, interval_end=ie
                )
            ),
        )
        for w, s, d, ie in (
            ("1min", 0.0, "decreasing", "next_change"),
            ("30s", 0.01, "increasing", "last_sample"),
        )
    ],
    *[
        (
            f"variance_{w}_{f}_{ie}",
            lambda df: DegradationDetectionEvents(df, "sig"),
            lambda o, w=w, f=f, ie=ie: o.detect_variance_increase(
                window=w, threshold_factor=f, interval_end=ie
            ),
            lambda o, w=w, f=f, ie=ie: (
                _ref_degradationdetection_detect_variance_increase(
                    o, w, f, interval_end=ie
                )
            ),
        )
        for w, f, ie in (("1min", 1.5, "next_change"), ("30s", 1.0, "last_sample"))
    ],
    *[
        (
            f"rms_{b}_{ie}",
            lambda df: VibrationAnalysisEvents(df, "sig2"),
            lambda o, b=b, ie=ie: o.detect_rms_exceedance(
                b, threshold_factor=1.0, window="20s", interval_end=ie
            ),
            lambda o, b=b, ie=ie: _ref_vibrationanalysis_detect_rms_exceedance(
                o, b, 1.0, "20s", interval_end=ie
            ),
        )
        for b, ie in ((1.0, "next_change"), (0.5, "last_sample"), (0.0, "next_change"))
    ],
    *[
        (
            f"low_stock_{lvl}_{h}",
            lambda df: InventoryMonitoringEvents(df, "sig"),
            lambda o, lvl=lvl, h=h: o.detect_low_stock(lvl, hold=h),
            lambda o, lvl=lvl, h=h: _ref_inventorymonitoring_detect_low_stock(
                o, lvl, h
            ),
        )
        for lvl, h in ((0.0, "0s"), (2.0, "5s"))
    ],
    *[
        (
            f"regularity_{w}",
            lambda df: SignalQualityEvents(df, "sig"),
            lambda o, w=w: o.sampling_regularity(window=w),
            lambda o, w=w: add_twin_columns(
                _ref_signalquality_sampling_regularity(o, w),
                {
                    "mean_interval": "mean_interval_seconds",
                    "std_interval": "std_interval_seconds",
                    "min_interval": "min_interval_seconds",
                    "max_interval": "max_interval_seconds",
                },
            ),
        )
        for w in ("1min", "1h")
    ],
    *[
        (
            f"out_of_range_{lo}_{ie}",
            lambda df: SignalQualityEvents(df, "sig"),
            lambda o, lo=lo, ie=ie: o.detect_out_of_range(lo, -lo, interval_end=ie),
            lambda o, lo=lo, ie=ie: _ref_signalquality_detect_out_of_range(
                o, lo, -lo, interval_end=ie
            ),
        )
        for lo, ie in (
            (-2.0, "next_change"),
            (-5.0, "last_sample"),
            (0.0, "next_change"),
        )
    ],
    *[
        (
            f"alarm_intervals_{u}",
            lambda df, u=u: AlarmManagementEvents(df, u),
            lambda o: o._intervalize(),
            _ref_alarmmanagement_intervalize,
        )
        for u in ("b1", "b2")
    ],
    *[
        (
            f"chattering_{m}_{w}",
            lambda df: AlarmManagementEvents(df, "b1"),
            lambda o, m=m, w=w: o.chattering_detection(min_transitions=m, window=w),
            lambda o, m=m, w=w: _ref_alarmmanagement_chattering_detection(o, m, w),
        )
        for m, w in ((2, "1min"), (5, "10min"), (1, "1s"))
    ],
    *[
        (
            f"batches_{u}",
            lambda df, u=u: BatchTrackingEvents(df, u),
            lambda o: o.detect_batches(),
            _ref_batchtracking_detect_batches,
        )
        for u in ("id1", "id3")
    ],
    (
        "batches_integer",
        lambda df: BatchTrackingEvents(df, "route", value_column="value_integer"),
        lambda o: o.detect_batches(),
        _ref_batchtracking_detect_batches,
    ),
    *[
        (
            f"value_trace_intervals_{u}",
            lambda df: ValueTraceabilityEvents(df, {"id1": "S1", "id3": "S3"}),
            lambda o, u=u: o._detect_intervals(u),
            lambda o, u=u: _ref_valuetraceability_detect_intervals(o, u),
        )
        for u in ("id1", "id3")
    ],
    *[
        (
            f"multi_timeline_{inf}",
            lambda df: MultiProcessTraceabilityEvents(df, PROCESSES),
            lambda o, inf=inf: o.build_timeline(infer_dwell_from_next=inf),
            lambda o, inf=inf: _ref_multiprocesstraceability_build_timeline(
                o, infer_dwell_from_next=inf
            ),
        )
        for inf in (False, True)
    ],
    (
        "multi_parallel",
        lambda df: MultiProcessTraceabilityEvents(df, PROCESSES),
        lambda o: o.parallel_activity(),
        _ref_multiprocesstraceability_parallel_activity,
    ),
    *[
        (
            f"flow_{kind}_{i}",
            lambda df: FlowConstraintEvents(df),
            # The public method adds severity_level (outside the loop).
            lambda o, kind=kind, kw=kw: getattr(o, f"{kind}_events")(
                roles=ROLES, **kw
            ).drop(columns=["severity_level"], errors="ignore"),
            lambda o, kind=kind, kw=kw: globals()[f"_ref_flowconstraint_{kind}_events"](
                o, roles=ROLES, **kw
            ),
        )
        for kind in ("blocked", "starved")
        for i, kw in enumerate(
            (
                {},
                {"tolerance_before": "1s", "tolerance_after": "2s"},
                {"min_duration": "5s", "interval_end": "last_sample"},
            )
        )
    ],
    *[
        (
            f"stable_band_{m}",
            lambda df: ChangeoverEvents(df),
            lambda o, m=m: o.changeover_window(
                "id1", until="stable_band", config={**BAND, "reference_method": m}
            ),
            lambda o, m=m: _changeover_with_ref(o, {**BAND, "reference_method": m}),
        )
        for m in ("expanding_median", "rolling_mean")
    ],
    *[
        (
            f"startup_slope_{s}_{w}_{d}",
            lambda df: StartupDetectionEvents(df, "sig"),
            lambda o, s=s, w=w, d=d: o.detect_startup_by_slope(
                min_slope=s, slope_window=w, min_duration=d
            ),
            lambda o, s=s, w=w, d=d: _ref_startupdetection_detect_startup_by_slope(
                o, min_slope=s, slope_window=w, min_duration=d
            ),
        )
        for s, w, d in ((0.0, "0s", "0s"), (0.1, "0s", "2s"), (0.05, "10s", "0s"))
    ],
    *[
        (
            f"startup_threshold_{t}_{h}_{m}",
            lambda df: StartupDetectionEvents(df, "sig"),
            lambda o, t=t, h=h, m=m: o.detect_startup_by_threshold(
                threshold=t, hysteresis=h, min_above=m
            ),
            lambda o, t=t, h=h, m=m: (
                _ref_startupdetection_detect_startup_by_threshold(
                    o, threshold=t, hysteresis=h, min_above=m
                )
            ),
        )
        for t, h, m in ((0.0, None, "0s"), (2.0, (2.0, 1.0), "10s"))
    ],
    *[
        (
            f"gaps_{g}",
            lambda df: DataGapAnalysisEvents(df, "sig"),
            lambda o, g=g: o.detect_gaps(min_gap=g),
            lambda o, g=g: _ref_datagapanalysis_detect_gaps(o, g),
        )
        for g in ("0s", "5s")
    ],
    *[
        (
            f"interpolation_{mx}_{mn}",
            lambda df: DataGapAnalysisEvents(df, "sig"),
            lambda o, mx=mx, mn=mn: o.interpolation_candidates(max_gap=mx, min_gap=mn),
            lambda o, mx=mx, mn=mn: _ref_datagapanalysis_interpolation_candidates(
                o, mx, mn
            ),
        )
        for mx, mn in (("10s", "0s"), ("1h", "2s"))
    ],
    *[
        (
            f"time_above_{t}_{w}_{ie}",
            lambda df: ThresholdMonitoringEvents(df, "sig"),
            lambda o, t=t, w=w, ie=ie: o.time_above_threshold(
                t, window=w, interval_end=ie
            ),
            lambda o, t=t, w=w, ie=ie: add_twin_columns(
                _ref_thresholdmonitoring_time_above_threshold(o, t, w, interval_end=ie),
                {"time_above": "time_above_seconds", "pct_above": "time_above_pct"},
            ),
        )
        for t, w, ie in (
            (0.0, "1h", "next_change"),
            (1.0, "10min", "last_sample"),
            (-2.0, "1min", "next_change"),
            (0.5, "1D", "next_change"),
        )
    ],
    *[
        (
            f"rapid_transitions_{t}_{m}",
            lambda df: MachineStateEvents(df, "b1"),
            lambda o, t=t, m=m: o.detect_rapid_transitions(threshold=t, min_count=m),
            lambda o, t=t, m=m: _ref_machinestate_detect_rapid_transitions(o, t, m),
        )
        for t, m in (("5s", 3), ("30s", 2), ("0s", 1))
    ],
    *[
        (
            f"{kind}_{r}_{d}_{p}",
            lambda df: WarmUpCoolDownEvents(df, "sig"),
            lambda o, kind=kind, r=r, d=d, p=p: getattr(o, f"detect_{kind}")(
                r, min_duration=d, max_pause=p
            ),
            lambda o, kind=kind, r=r, d=d, p=p: globals()[
                f"_ref_warmupcooldown_detect_{kind}"
            ](o, r, d, max_pause=p),
        )
        for kind in ("warmup", "cooldown")
        for r, d, p in ((0.5, "0s", 0), (2.0, "10s", 0), (1.0, "5s", 1))
    ],
    *[
        (
            f"state_sequence_{n}_{order}",
            lambda df: OperatingStateEvents(df, "sig"),
            lambda o, n=n, order=order: o.unusual_state_sequence(
                n_states=n, expected_order=order
            ),
            lambda o, n=n, order=order: _ref_operatingstate_unusual_state_sequence(
                o, n, order
            ),
        )
        for n, order in ((2, None), (3, None), (3, [0, 1, 2, 1, 0]))
    ],
    *[
        (
            f"dwell_{n}_{s}_{m}",
            lambda df: OperatingStateEvents(df, "sig"),
            lambda o, n=n, s=s, m=m: o.dwell_time_violation(s, m, n_states=n),
            lambda o, n=n, s=s, m=m: _ref_operatingstate_dwell_time_violation(
                o, s, m, n
            ),
        )
        for n, s, m in ((2, 0, "0s"), (3, 2, "10s"))
    ],
    *[
        (
            f"routing_{name}_{inf}",
            make,
            lambda o, inf=inf: o.build_routing_timeline(infer_dwell_from_next=inf),
            lambda o, inf=inf: _ref_routingtraceability_build_routing_timeline(
                o, infer_dwell_from_next=inf
            ),
        )
        for name, make in (
            ("integer", _routing("id1", "route", "value_integer", {1: "W", 2: "P"})),
            ("double", _routing("id3", "sig2", "value_double")),
            ("string", _routing("id3", "id2", "value_string")),
        )
        for inf in (False, True)
    ],
]


# The pairwise / per-change references are quadratic: keep their frames small.
_SLOW = ("multi_", "stable_band_")


@pytest.mark.parametrize("cid,make,new,ref", CASES, ids=[c[0] for c in CASES])
def test_matches_reference(frames, cid, make, new, ref):
    if cid.startswith(_SLOW):
        frames = {k: v for k, v in frames.items() if len(v) < 5000}
    _check(frames, make, new, ref)


def test_parallel_flags_with_ties_and_zero_length():
    # Many tied starts, zero-length visits and nested intervals per item.
    rng = np.random.default_rng(11)
    t0 = pd.Timestamp("2024-01-01")
    rows = []
    for item in range(60):
        for _ in range(int(rng.integers(1, 7))):
            start = t0 + pd.Timedelta(int(rng.integers(0, 8)), "s")
            rows.append((f"I{item:02d}", start, int(rng.integers(0, 4))))
    timeline = pd.DataFrame(rows, columns=["item_id", "start", "len"])
    timeline["end"] = timeline["start"] + pd.to_timedelta(timeline["len"], "s")
    timeline = timeline.sort_values(["item_id", "start"]).reset_index(drop=True)
    expected = timeline.copy()
    MultiProcessTraceabilityEvents._parallel_flags_loop(expected)
    np.testing.assert_array_equal(_parallel_flags(timeline), expected["is_parallel"])
