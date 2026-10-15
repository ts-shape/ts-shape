"""Runtime / operating-hours accounting for equipment.

Computes operating-time metrics from a single boolean run signal:

- total run time, idle time and utilization,
- equipment start count and longest continuous run,
- run time per calendar window,
- a cumulative operating-hours meter (like a physical hour meter).

Per-sample duration is the gap to the next sample, summed over ``True``
samples -- the same approach as ``OEEEvents.availability``. Per window, each
sample's interval is split at the window boundaries, so a run that crosses
midnight counts in both days (the ISO 22400-2 time model).
This is distinct from ``DutyCycleEvents`` (duty *percentage* and cycle
counts): here the focus is *absolute* run time and an hour-meter reading.
"""

import logging
import warnings
from typing import Any

import pandas as pd

from ts_shape.events._output import empty_event_df, finalize_summary_df
from ts_shape.events._windows import floor_local
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    REMOVAL,
    caller_stacklevel,
    deprecated_method,
    keyword_only,
    renamed_kwargs,
)
from ts_shape.utils.time_accounting import (
    floor_periods,
    reading_intervals,
    time_by_period,
)

logger = logging.getLogger(__name__)


class RuntimeAccountingEvents(Base):
    """Operating-hours accounting from one boolean run signal.

    Example usage::

        rt = RuntimeAccountingEvents(df, run_state_uuid="machine:running")
        rt.runtime_summary()
        rt.runtime_by_window(window="1D")
        rt.operating_hours_meter(window="1h")
    """

    @renamed_kwargs(run_uuid="run_state_uuid")
    def __init__(
        self,
        dataframe: pd.DataFrame,
        run_state_uuid: str,
        *,
        event_uuid: str = "prod:runtime",
        value_column: str = "value_bool",
        time_column: str = "systime",
    ) -> None:
        """Initialize the runtime-accounting analyzer.

        Args:
            dataframe: Input DataFrame with timeseries data.
            run_state_uuid: UUID of the boolean run-state signal (True = running).
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the boolean run state.
            time_column: Name of the timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.run_state_uuid = run_state_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, run_state_uuid)

    @property
    def run_uuid(self) -> str:
        """Deprecated read-only alias of :attr:`run_state_uuid`."""
        warnings.warn(
            "RuntimeAccountingEvents.run_uuid is deprecated, use .run_state_uuid; "
            f"the old name will be removed in {REMOVAL}",
            DeprecationWarning,
            stacklevel=caller_stacklevel(),
        )
        return self.run_state_uuid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _samples(self, max_gap: str | None = None) -> pd.DataFrame:
        """Return the run signal with per-sample duration and start flag.

        Columns: ``<time_column>``, ``state`` (bool), ``dur`` (seconds to the
        next sample), ``is_start`` (rising edge).

        When ``max_gap`` is set, a gap to the next sample longer than it is
        treated as a logging outage and contributes no time, so it inflates
        neither run nor idle totals.
        """
        sig = (
            self.dataframe[self.dataframe["uuid"] == self.run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        cols = [self.time_column, "state", "dur", "is_start"]
        if sig.empty:
            return pd.DataFrame(columns=cols)
        sig[self.time_column] = pd.to_datetime(sig[self.time_column])
        sig["state"] = sig[self.value_column].fillna(False).astype(bool)
        sig["dur"] = (
            sig[self.time_column].shift(-1) - sig[self.time_column]
        ).dt.total_seconds()
        # A gap longer than max_gap is "no data" -> contributes no time.
        if max_gap is not None:
            cap_seconds = pd.to_timedelta(max_gap).total_seconds()
            sig.loc[sig["dur"] > cap_seconds, "dur"] = 0.0
        # The final sample has no successor; assume it contributes no time.
        sig["dur"] = sig["dur"].fillna(0.0)
        sig["is_start"] = sig["state"] & (~sig["state"].shift(fill_value=False))
        return sig[cols].reset_index(drop=True)

    def _run_intervals(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Contiguous run segments. Columns: ``start``, ``duration_seconds``."""
        change = (samples["state"] != samples["state"].shift()).cumsum()
        rows: list[dict[str, Any]] = []
        for _, grp in samples.groupby(change):
            if not bool(grp["state"].iloc[0]):
                continue
            rows.append(
                {
                    "start": grp[self.time_column].iloc[0],
                    "duration_seconds": float(grp["dur"].sum()),
                }
            )
        return pd.DataFrame(rows, columns=["start", "duration_seconds"])

    def _time_by_window(
        self, samples: pd.DataFrame, window: str, max_gap: str | None
    ) -> pd.DataFrame:
        """Run / idle seconds and start count per ``floor``-aligned window.

        Each sample's state holds until the next sample; the intervals are
        split at the window boundaries, so a run that crosses midnight counts
        in both days (ISO 22400-2 time model). A rising edge counts in the
        window of the sample that starts the run. Windows that hold a sample
        or any run / idle time are reported, in time order.

        Columns: ``start``, ``end``, ``run``, ``idle``, ``start_count``.
        """
        times = samples[self.time_column]
        intervals = reading_intervals(
            times, samples["state"].map({True: "run", False: "idle"}), max_gap=max_gap
        )
        periods = floor_periods(times.iloc[0], times.iloc[-1], window)
        seconds = time_by_period(
            intervals, periods, keys=["period"], states=["run", "idle"]
        ).set_index("period")
        labels = floor_local(times, window)
        starts = samples["is_start"].groupby(labels.to_numpy()).sum()
        keep = periods["period"].isin(seconds.index) | periods["period"].isin(labels)
        out = periods.loc[keep, ["period", "start", "end"]].reset_index(drop=True)
        out["run"] = (
            out["period"].map(seconds["run"]).fillna(0.0).to_numpy(dtype="float64")
        )
        out["idle"] = (
            out["period"].map(seconds["idle"]).fillna(0.0).to_numpy(dtype="float64")
        )
        out["start_count"] = out["period"].map(starts).fillna(0).to_numpy(dtype="int64")
        return out[["start", "end", "run", "idle", "start_count"]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def runtime_summary(self, *, max_gap: str | None = None) -> pd.DataFrame:
        """Overall operating-time summary across the whole dataset.

        Args:
            max_gap: Optional cap (e.g. "15min"). A gap to the next sample
                longer than this is treated as a logging outage and excluded, so
                it does not inflate run/idle time or utilization. Without it, a
                gap while the last state was ``running`` is booked as run time.
                ``None`` (default) applies no cap: every interval between
                readings, however long, counts toward the preceding state.

        Returns:
            Summary-shape DataFrame (one row) with columns: start, end,
            duration_seconds, run_seconds, run_hours, idle_seconds,
            start_count, longest_run_seconds, mean_run_seconds,
            utilization_pct.
        """
        extra = [
            "run_seconds",
            "run_hours",
            "idle_seconds",
            "start_count",
            "longest_run_seconds",
            "mean_run_seconds",
            "utilization_pct",
        ]
        samples = self._samples(max_gap)
        if samples.empty:
            return empty_event_df("summary", extra_cols=extra)

        run_seconds = float(samples.loc[samples["state"], "dur"].sum())
        idle_seconds = float(samples.loc[~samples["state"], "dur"].sum())
        runs = self._run_intervals(samples)
        total = run_seconds + idle_seconds
        row = {
            "start": samples[self.time_column].iloc[0],
            "end": samples[self.time_column].iloc[-1],
            "run_seconds": run_seconds,
            "run_hours": run_seconds / 3600.0,
            "idle_seconds": idle_seconds,
            "start_count": int(len(runs)),
            "longest_run_seconds": (
                float(runs["duration_seconds"].max()) if not runs.empty else 0.0
            ),
            "mean_run_seconds": (
                float(runs["duration_seconds"].mean()) if not runs.empty else 0.0
            ),
            "utilization_pct": (run_seconds / total * 100.0) if total > 0 else 0.0,
        }
        return finalize_summary_df(pd.DataFrame([row]))

    runtime_per_window = deprecated_method("runtime_by_window")

    @keyword_only(1)
    def runtime_by_window(
        self, window: str = "1D", *, max_gap: str | None = None
    ) -> pd.DataFrame:
        """Run time per calendar window.

        Each sample's state holds until the next sample; the interval is
        split at the window boundaries, so each window gets the run and idle
        time that fell inside it. A start counts in the window of the sample
        that begins the run. Windows holding a sample or any observed time
        are reported.

        Args:
            window: Resample window (e.g. ``"1D"``, ``"8h"``).
            max_gap: Optional cap (e.g. "15min") excluding a logging gap from
                run time and utilization (see :meth:`runtime_summary`).
                ``None`` (default) applies no cap.

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            run_seconds, run_hours, start_count, utilization_pct.
        """
        extra = ["run_seconds", "run_hours", "start_count", "utilization_pct"]
        samples = self._samples(max_gap)
        if samples.empty:
            return empty_event_df("summary", extra_cols=extra)

        by_window = self._time_by_window(samples, window, max_gap)
        rows: list[dict[str, Any]] = []
        for start, end, run_s, idle_s, n_starts in by_window.itertuples(
            index=False, name=None
        ):
            covered = run_s + idle_s
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "run_seconds": run_s,
                    "run_hours": run_s / 3600.0,
                    "start_count": int(n_starts),
                    "utilization_pct": (
                        (run_s / covered * 100.0) if covered > 0 else 0.0
                    ),
                }
            )
        return finalize_summary_df(pd.DataFrame(rows))

    @keyword_only(1)
    def operating_hours_meter(
        self, window: str = "1h", *, max_gap: str | None = None
    ) -> pd.DataFrame:
        """Cumulative operating-hours meter, sampled per window.

        Mirrors a physical equipment hour meter: a monotonically increasing
        total of run hours. Run time is split at the window boundaries (see
        :meth:`runtime_by_window`); every window holding a sample or any
        observed time is reported, so the meter has no holes.

        Args:
            window: Resample window for the meter readings.
            max_gap: Optional cap (e.g. "15min") excluding a logging gap from
                run time so the meter does not advance across an outage.
                ``None`` (default) applies no cap.

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            run_seconds, cumulative_run_hours.
        """
        extra = ["run_seconds", "cumulative_run_hours"]
        samples = self._samples(max_gap)
        if samples.empty:
            return empty_event_df("summary", extra_cols=extra)

        by_window = self._time_by_window(samples, window, max_gap)
        out = by_window[["start", "end", "run"]].rename(columns={"run": "run_seconds"})
        out["cumulative_run_hours"] = out["run_seconds"].cumsum() / 3600.0
        return finalize_summary_df(out)
