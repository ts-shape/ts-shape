import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_summary_df,
    twin_columns,
)
from ts_shape.events._windows import add_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import caller_stacklevel, keyword_only
from ts_shape.utils.time_weighted import time_weighted_resample

logger = logging.getLogger(__name__)


class MaterialBalanceEvents(Base):
    """Engineering: Material / Energy Balance

    Check whether inputs and outputs balance (mass, energy, flow) per time
    window. The most fundamental daily engineering check for any process.

    Methods:
    - balance_check: Per-window sum(inputs) vs sum(outputs).
    - imbalance_trend: Track whether imbalance is growing or shrinking.
    - detect_balance_exceedance: Sustained imbalance events.
    - contribution_breakdown: Each signal's share of total input/output.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_uuids: list[str],
        output_uuids: list[str],
        *,
        event_uuid: str = "eng:material_balance",
        value_column: str = "value_double",
        time_column: str = "systime",
        time_weighted: bool = True,
        missing_as_zero: bool = False,
    ) -> None:
        """
        Args:
            dataframe: Long-format input data.
            input_uuids: Signals flowing into the balance boundary.
            output_uuids: Signals flowing out of it.
            event_uuid: UUID assigned to generated events.
            value_column: Column holding the flow values.
            time_column: Timestamp column.
            time_weighted: ``True`` (default) averages each flow by time
                (value held until the next sample). ``False`` uses
                ``resample().mean()`` over the samples, which is biased on
                irregular or change-encoded flow signals.
            missing_as_zero: ``False`` (default) leaves a window's totals NaN
                when a signal has no data in it; the window is marked not
                balanced but is not reported by
                :meth:`detect_balance_exceedance`, so a sensor outage is not
                read as a real imbalance. ``True`` treats a window without data
                for a signal as zero flow.
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_weighted = time_weighted
        self.missing_as_zero = missing_as_zero
        self.input_uuids = input_uuids
        self.output_uuids = output_uuids
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.dataframe[self.time_column] = pd.to_datetime(
            self.dataframe[self.time_column]
        )

    def _resample_signals(self, uuids: list[str], window: str) -> pd.DataFrame:
        """Resample each UUID to window and return sum per window."""
        frames: list[pd.Series] = []
        missing: list[str] = []
        # Hold every flow until the last sample of any balance signal, so a
        # signal that stops reporting earlier keeps its last value.
        balance = self.dataframe["uuid"].isin([*self.input_uuids, *self.output_uuids])
        data_end = self.dataframe.loc[balance, self.time_column].max()
        for uid in uuids:
            sig = self.dataframe[self.dataframe["uuid"] == uid]
            if sig.empty:
                missing.append(uid)
                continue
            values = sig.set_index(self.time_column)[self.value_column]
            if self.time_weighted:
                s = time_weighted_resample(values, window, data_end=data_end)
            else:
                s = values.resample(window).mean()
            if self.missing_as_zero:
                s = s.fillna(0.0)
            frames.append(s.rename(uid))

        if missing:
            treated = "zero flow" if self.missing_as_zero else "missing (NaN totals)"
            warnings.warn(
                f"Material balance signal(s) {missing!r} have no rows; "
                f"treated as {treated}.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        if self.missing_as_zero:
            return combined.fillna(0.0)
        for uid in missing:
            combined[uid] = np.nan
        return combined

    @keyword_only(1)
    def balance_check(
        self,
        window: str = "1h",
        tolerance_pct: float = 5.0,
    ) -> pd.DataFrame:
        """Per-window balance check: sum(inputs) vs sum(outputs).

        Args:
            window: Resample window.
            tolerance_pct: Maximum acceptable imbalance percentage.

        Returns:
            DataFrame with columns: start, end, duration_seconds,
            total_input, total_output, imbalance, imbalance_pct, balanced.
            ``end`` is ``start`` + ``window``.
        """
        cols = [
            "start",
            "total_input",
            "total_output",
            "imbalance",
            "imbalance_pct",
            "balanced",
        ]
        inputs = self._resample_signals(self.input_uuids, window)
        outputs = self._resample_signals(self.output_uuids, window)

        if inputs.empty and outputs.empty:
            return empty_event_df("summary", cols[1:])

        # Align indices
        all_idx = (
            inputs.index.union(outputs.index)
            if not inputs.empty and not outputs.empty
            else (inputs.index if not inputs.empty else outputs.index)
        )

        fill = 0.0 if self.missing_as_zero else np.nan
        if inputs.empty:
            total_in = pd.Series(fill, index=all_idx)
        else:
            total_in = inputs.reindex(all_idx, fill_value=fill).sum(
                axis=1, skipna=self.missing_as_zero
            )

        if outputs.empty:
            total_out = pd.Series(fill, index=all_idx)
        else:
            total_out = outputs.reindex(all_idx, fill_value=fill).sum(
                axis=1, skipna=self.missing_as_zero
            )

        imbalance = total_in - total_out
        denom = total_in.replace(0, np.nan)
        imbalance_pct = imbalance.abs() / denom * 100
        # Zero input: balanced only if nothing flowed out either. (Previously
        # any zero-input window read 0 % imbalance, i.e. balanced, even with
        # output flowing.)
        zero_in = total_in == 0
        imbalance_pct = imbalance_pct.mask(
            zero_in, np.where(total_out.abs() > 0, np.inf, 0.0)
        )

        result = pd.DataFrame(
            {
                "start": all_idx,
                "total_input": total_in.values,
                "total_output": total_out.values,
                "imbalance": imbalance.values,
                "imbalance_pct": imbalance_pct.values,
                "balanced": (imbalance_pct <= tolerance_pct).values,
            }
        )

        return finalize_summary_df(
            add_window_bounds(result[cols].reset_index(drop=True), window)
        )

    @keyword_only(1)
    def imbalance_trend(self, window: str = "1h") -> pd.DataFrame:
        """Track whether imbalance is growing, shrinking, or stable.

        Args:
            window: Resample window passed to :meth:`balance_check`. The rolling
                average spans the last three windows.

        Returns:
            DataFrame with columns: start, end, duration_seconds,
            imbalance_pct, rolling_avg_imbalance, trend_direction. ``end`` is
            ``start`` + ``window``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "imbalance_pct",
            "rolling_avg_imbalance",
            "trend_direction",
        ]
        bc = self.balance_check(window=window)
        if bc.empty or len(bc) < 2:
            return empty_event_df("summary", cols)

        result = bc[["start", "end", "duration_seconds", "imbalance_pct"]].copy()
        result["rolling_avg_imbalance"] = (
            result["imbalance_pct"].rolling(3, min_periods=1).mean()
        )

        diff = result["rolling_avg_imbalance"].diff()
        result["trend_direction"] = np.where(
            diff > 0.5, "growing", np.where(diff < -0.5, "shrinking", "stable")
        )

        return result[cols].reset_index(drop=True)

    @keyword_only(1)
    def detect_balance_exceedance(
        self,
        window: str = "1h",
        tolerance_pct: float = 5.0,
        min_duration: str = "2h",
    ) -> pd.DataFrame:
        """Detect sustained imbalance events.

        Args:
            window: Resample window passed to :meth:`balance_check`.
            tolerance_pct: Maximum acceptable imbalance percentage per window.
            min_duration: Minimum length of a run of consecutive unbalanced
                windows to report it as an event (pandas offset string).
                Windows whose totals are NaN (missing data) end a run.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid (input and output uuids, comma-joined), is_delta,
            avg_imbalance_pct, max_imbalance_pct, likely_cause.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "avg_imbalance_pct",
            "max_imbalance_pct",
            "duration_seconds",
            "likely_cause",
        ]
        bc = self.balance_check(window=window, tolerance_pct=tolerance_pct)
        if bc.empty:
            return empty_event_df("interval", cols)

        # Windows with unknown totals (a signal without data) are not
        # exceedances; ``imbalance_pct`` is NaN there.
        unbalanced = bc["imbalance_pct"] > tolerance_pct
        if not unbalanced.any():
            return empty_event_df("interval", cols)

        window_td = pd.Timedelta(window)
        min_td = pd.Timedelta(min_duration)

        groups = (unbalanced != unbalanced.shift()).cumsum()
        events: list[dict[str, Any]] = []

        for _, seg_idx in unbalanced.groupby(groups):
            if not seg_idx.iloc[0]:
                continue
            seg = bc.loc[seg_idx.index]
            start = seg["start"].iloc[0]
            end = seg["start"].iloc[-1] + window_td
            dur = end - start
            if dur < min_td:
                continue

            imb = seg["imbalance"]
            avg_pct = float(seg["imbalance_pct"].mean())
            max_pct = float(seg["imbalance_pct"].max())

            # Classify likely cause
            if (imb > 0).all():
                cause = "accumulation"
            elif (imb < 0).all():
                cause = "leak_or_unmetered"
            else:
                cause = "measurement_error"

            events.append(
                {
                    "start": start,
                    "end": end,
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "avg_imbalance_pct": avg_pct,
                    "max_imbalance_pct": max_pct,
                    "duration_seconds": dur.total_seconds(),
                    "likely_cause": cause,
                }
            )

        if not events:
            return empty_event_df("interval", cols)
        return finalize_interval_df(
            pd.DataFrame(events, columns=cols),
            uuid=self.event_uuid,
            source_uuid=",".join([*self.input_uuids, *self.output_uuids]),
        )

    @keyword_only(1)
    @twin_columns({"pct_of_total": "share_of_total_pct"})
    def contribution_breakdown(self, window: str = "1h") -> pd.DataFrame:
        """Each signal's contribution to total input/output per window.

        Args:
            window: Resample window (pandas offset string).

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, role, value, pct_of_total. ``uuid`` and
            ``source_uuid`` both hold the contributing signal's uuid; ``end``
            is ``start`` + ``window``.

            Canonical twin columns: ``share_of_total_pct`` (after ``pct_of_total``); each
            holds the same values as the deprecated column right before it, which is
            removed in the next major release.
        """
        cols = ["start", "uuid", "role", "value", "pct_of_total"]
        inputs = self._resample_signals(self.input_uuids, window)
        outputs = self._resample_signals(self.output_uuids, window)

        if inputs.empty and outputs.empty:
            return empty_event_df("summary", ["uuid", "source_uuid", *cols[2:]])

        events: list[dict[str, Any]] = []

        if not inputs.empty:
            for ws in inputs.index:
                total = float(inputs.loc[ws].sum())
                for uid in inputs.columns:
                    val = float(inputs.loc[ws, uid])
                    events.append(
                        {
                            "start": ws,
                            "uuid": uid,
                            "role": "input",
                            "value": val,
                            "pct_of_total": (val / total * 100) if total > 0 else 0.0,
                        }
                    )

        if not outputs.empty:
            for ws in outputs.index:
                total = float(outputs.loc[ws].sum())
                for uid in outputs.columns:
                    val = float(outputs.loc[ws, uid])
                    events.append(
                        {
                            "start": ws,
                            "uuid": uid,
                            "role": "output",
                            "value": val,
                            "pct_of_total": (val / total * 100) if total > 0 else 0.0,
                        }
                    )

        if not events:
            return empty_event_df("summary", ["uuid", "source_uuid", *cols[2:]])
        result = pd.DataFrame(events, columns=cols)
        result["source_uuid"] = result["uuid"]
        return finalize_summary_df(add_window_bounds(result, window))
