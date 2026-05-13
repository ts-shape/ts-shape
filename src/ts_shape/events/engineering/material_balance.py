import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

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
        input_uuids: List[str],
        output_uuids: List[str],
        *,
        event_uuid: str = "eng:material_balance",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.input_uuids = input_uuids
        self.output_uuids = output_uuids
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.dataframe[self.time_column] = pd.to_datetime(
            self.dataframe[self.time_column]
        )

    def _resample_signals(self, uuids: List[str], window: str) -> pd.DataFrame:
        """Resample each UUID to window and return sum per window."""
        frames: List[pd.Series] = []
        for uid in uuids:
            sig = self.dataframe[self.dataframe["uuid"] == uid]
            if sig.empty:
                continue
            s = (
                sig.set_index(self.time_column)[self.value_column]
                .resample(window)
                .mean()
                .fillna(0.0)
            )
            frames.append(s.rename(uid))

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).fillna(0.0)
        return combined

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
            DataFrame with columns: window_start, total_input,
            total_output, imbalance, imbalance_pct, balanced.
        """
        cols = [
            "window_start",
            "total_input",
            "total_output",
            "imbalance",
            "imbalance_pct",
            "balanced",
        ]
        inputs = self._resample_signals(self.input_uuids, window)
        outputs = self._resample_signals(self.output_uuids, window)

        if inputs.empty and outputs.empty:
            return pd.DataFrame(columns=cols)

        # Align indices
        all_idx = (
            inputs.index.union(outputs.index)
            if not inputs.empty and not outputs.empty
            else (inputs.index if not inputs.empty else outputs.index)
        )

        if inputs.empty:
            total_in = pd.Series(0.0, index=all_idx)
        else:
            total_in = inputs.reindex(all_idx, fill_value=0.0).sum(axis=1)

        if outputs.empty:
            total_out = pd.Series(0.0, index=all_idx)
        else:
            total_out = outputs.reindex(all_idx, fill_value=0.0).sum(axis=1)

        imbalance = total_in - total_out
        denom = total_in.replace(0, np.nan)
        imbalance_pct = (imbalance.abs() / denom * 100).fillna(0.0)

        result = pd.DataFrame(
            {
                "window_start": all_idx,
                "total_input": total_in.values,
                "total_output": total_out.values,
                "imbalance": imbalance.values,
                "imbalance_pct": imbalance_pct.values,
                "balanced": (imbalance_pct <= tolerance_pct).values,
            }
        )

        return result[cols].reset_index(drop=True)

    def imbalance_trend(self, window: str = "1h") -> pd.DataFrame:
        """Track whether imbalance is growing, shrinking, or stable.

        Returns:
            DataFrame with columns: window_start, imbalance_pct,
            rolling_avg_imbalance, trend_direction.
        """
        cols = [
            "window_start",
            "imbalance_pct",
            "rolling_avg_imbalance",
            "trend_direction",
        ]
        bc = self.balance_check(window)
        if bc.empty or len(bc) < 2:
            return pd.DataFrame(columns=cols)

        result = bc[["window_start", "imbalance_pct"]].copy()
        result["rolling_avg_imbalance"] = (
            result["imbalance_pct"].rolling(3, min_periods=1).mean()
        )

        diff = result["rolling_avg_imbalance"].diff()
        result["trend_direction"] = np.where(
            diff > 0.5, "growing", np.where(diff < -0.5, "shrinking", "stable")
        )

        return result[cols].reset_index(drop=True)

    def detect_balance_exceedance(
        self,
        window: str = "1h",
        tolerance_pct: float = 5.0,
        min_duration: str = "2h",
    ) -> pd.DataFrame:
        """Detect sustained imbalance events.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            avg_imbalance_pct, max_imbalance_pct, duration_seconds,
            likely_cause.
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
        bc = self.balance_check(window, tolerance_pct)
        if bc.empty:
            return pd.DataFrame(columns=cols)

        unbalanced = ~bc["balanced"]
        if not unbalanced.any():
            return pd.DataFrame(columns=cols)

        window_td = pd.Timedelta(window)
        min_td = pd.Timedelta(min_duration)

        groups = (unbalanced != unbalanced.shift()).cumsum()
        events: List[Dict[str, Any]] = []

        for _, seg_idx in unbalanced.groupby(groups):
            if not seg_idx.iloc[0]:
                continue
            seg = bc.loc[seg_idx.index]
            start = seg["window_start"].iloc[0]
            end = seg["window_start"].iloc[-1] + window_td
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

        return pd.DataFrame(events, columns=cols)

    def contribution_breakdown(self, window: str = "1h") -> pd.DataFrame:
        """Each signal's contribution to total input/output per window.

        Returns:
            DataFrame with columns: window_start, uuid, role, value,
            pct_of_total.
        """
        cols = ["window_start", "uuid", "role", "value", "pct_of_total"]
        inputs = self._resample_signals(self.input_uuids, window)
        outputs = self._resample_signals(self.output_uuids, window)

        if inputs.empty and outputs.empty:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []

        if not inputs.empty:
            for ws in inputs.index:
                total = float(inputs.loc[ws].sum())
                for uid in inputs.columns:
                    val = float(inputs.loc[ws, uid])
                    events.append(
                        {
                            "window_start": ws,
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
                            "window_start": ws,
                            "uuid": uid,
                            "role": "output",
                            "value": val,
                            "pct_of_total": (val / total * 100) if total > 0 else 0.0,
                        }
                    )

        return pd.DataFrame(events, columns=cols)
