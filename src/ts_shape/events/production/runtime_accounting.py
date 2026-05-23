"""Runtime / operating-hours accounting for equipment.

Computes operating-time metrics from a single boolean run signal:

- total run time, idle time and utilization,
- equipment start count and longest continuous run,
- run time per calendar window,
- a cumulative operating-hours meter (like a physical hour meter).

Per-sample duration is the gap to the next sample, summed over ``True``
samples -- the same approach as ``OEECalculator.calculate_availability``.
This is distinct from ``DutyCycleEvents`` (duty *percentage* and cycle
counts): here the focus is *absolute* run time and an hour-meter reading.
"""

import logging
from typing import Any

import pandas as pd  # type: ignore

from ts_shape.events._output import empty_event_df, finalize_summary_df
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class RuntimeAccountingEvents(Base):
    """Operating-hours accounting from one boolean run signal.

    Example usage::

        rt = RuntimeAccountingEvents(df, run_uuid="machine:running")
        rt.runtime_summary()
        rt.runtime_per_window(window="1D")
        rt.operating_hours_meter(window="1h")
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        run_uuid: str,
        *,
        event_uuid: str = "prod:runtime",
        value_column: str = "value_bool",
        time_column: str = "systime",
    ) -> None:
        """Initialize the runtime-accounting analyser.

        Args:
            dataframe: Input DataFrame with timeseries data.
            run_uuid: UUID of the boolean run-state signal (True = running).
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the boolean run state.
            time_column: Name of the timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.run_uuid = run_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, run_uuid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _samples(self) -> pd.DataFrame:
        """Return the run signal with per-sample duration and start flag.

        Columns: ``<time_column>``, ``state`` (bool), ``dur`` (seconds to the
        next sample), ``is_start`` (rising edge).
        """
        sig = (
            self.dataframe[self.dataframe["uuid"] == self.run_uuid]
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def runtime_summary(self) -> pd.DataFrame:
        """Overall operating-time summary across the whole dataset.

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
        samples = self._samples()
        if samples.empty:
            return empty_event_df("summary", extra_cols=extra)

        run_seconds = float(samples.loc[samples["state"], "dur"].sum())
        idle_seconds = float(samples.loc[~samples["state"], "dur"].sum())
        runs = self._run_intervals(samples)
        total = run_seconds + idle_seconds
        row = {
            "start": samples[self.time_column].iloc[0],
            "end": samples[self.time_column].iloc[-1],
            "run_seconds": round(run_seconds, 2),
            "run_hours": round(run_seconds / 3600.0, 3),
            "idle_seconds": round(idle_seconds, 2),
            "start_count": int(len(runs)),
            "longest_run_seconds": (
                round(float(runs["duration_seconds"].max()), 2)
                if not runs.empty
                else 0.0
            ),
            "mean_run_seconds": (
                round(float(runs["duration_seconds"].mean()), 2)
                if not runs.empty
                else 0.0
            ),
            "utilization_pct": (
                round(run_seconds / total * 100.0, 2) if total > 0 else 0.0
            ),
        }
        return finalize_summary_df(pd.DataFrame([row]))

    def runtime_per_window(self, window: str = "1D") -> pd.DataFrame:
        """Run time per calendar window.

        Args:
            window: Resample window (e.g. ``"1D"``, ``"8h"``).

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            run_seconds, run_hours, start_count, utilization_pct.
        """
        extra = ["run_seconds", "run_hours", "start_count", "utilization_pct"]
        samples = self._samples()
        if samples.empty:
            return empty_event_df("summary", extra_cols=extra)

        samples = samples.copy()
        samples["win"] = samples[self.time_column].dt.floor(window)
        rows: list[dict[str, Any]] = []
        for win, grp in samples.groupby("win", sort=True):
            run_s = float(grp.loc[grp["state"], "dur"].sum())
            covered = float(grp["dur"].sum())
            rows.append(
                {
                    "start": win,
                    "end": win + pd.Timedelta(window),
                    "run_seconds": round(run_s, 2),
                    "run_hours": round(run_s / 3600.0, 3),
                    "start_count": int(grp["is_start"].sum()),
                    "utilization_pct": (
                        round(run_s / covered * 100.0, 2) if covered > 0 else 0.0
                    ),
                }
            )
        return finalize_summary_df(pd.DataFrame(rows))

    def operating_hours_meter(self, window: str = "1h") -> pd.DataFrame:
        """Cumulative operating-hours meter, sampled per window.

        Mirrors a physical equipment hour meter: a monotonically increasing
        total of run hours.

        Args:
            window: Resample window for the meter readings.

        Returns:
            Summary-shape DataFrame with columns: start, end, duration_seconds,
            run_seconds, cumulative_run_hours.
        """
        extra = ["run_seconds", "cumulative_run_hours"]
        samples = self._samples()
        if samples.empty:
            return empty_event_df("summary", extra_cols=extra)

        samples = samples.copy()
        samples["win"] = samples[self.time_column].dt.floor(window)
        all_windows = samples.groupby("win", sort=True).size().index
        run_per = (
            samples[samples["state"]]
            .groupby("win")["dur"]
            .sum()
            .reindex(all_windows, fill_value=0.0)
            .sort_index()
        )
        out = run_per.reset_index()
        out.columns = ["start", "run_seconds"]
        out["cumulative_run_hours"] = (out["run_seconds"].cumsum() / 3600.0).round(3)
        out["run_seconds"] = out["run_seconds"].round(2)
        out["end"] = out["start"] + pd.Timedelta(window)
        return finalize_summary_df(out)
