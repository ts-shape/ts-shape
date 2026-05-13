import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class SignalCorrelationEvents(Base):
    """Correlation: Signal Correlation Analysis

    Analyze time-windowed correlations between pairs of numeric signals.
    Useful for detecting when normally correlated process variables diverge.

    Methods:
    - rolling_correlation: Pearson correlation over rolling windows.
    - correlation_breakdown: Detect periods where correlation drops below threshold.
    - lag_correlation: Cross-correlation with time lag to find delayed relationships.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "corr:signal",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

    def _align_signals(
        self, uuid_a: str, uuid_b: str, resample: str = "1min"
    ) -> pd.DataFrame:
        """Align two signals on a common time index via resampling."""
        a = (
            self.dataframe[self.dataframe["uuid"] == uuid_a]
            .copy()
            .sort_values(self.time_column)
        )
        b = (
            self.dataframe[self.dataframe["uuid"] == uuid_b]
            .copy()
            .sort_values(self.time_column)
        )

        if a.empty or b.empty:
            return pd.DataFrame(columns=["signal_a", "signal_b"])

        a[self.time_column] = pd.to_datetime(a[self.time_column])
        b[self.time_column] = pd.to_datetime(b[self.time_column])

        a = a.set_index(self.time_column)[self.value_column].resample(resample).mean()
        b = b.set_index(self.time_column)[self.value_column].resample(resample).mean()

        aligned = pd.DataFrame({"signal_a": a, "signal_b": b}).dropna()
        return aligned

    def rolling_correlation(
        self,
        uuid_a: str,
        uuid_b: str,
        *,
        resample: str = "1min",
        window: int = 60,
    ) -> pd.DataFrame:
        """Compute rolling Pearson correlation between two signals.

        Args:
            uuid_a: UUID of first signal.
            uuid_b: UUID of second signal.
            resample: Resample interval for alignment.
            window: Rolling window size (in resampled periods).

        Returns:
            DataFrame: systime, uuid, source_uuid_a, source_uuid_b,
                       is_delta, correlation
        """
        aligned = self._align_signals(uuid_a, uuid_b, resample=resample)
        if aligned.empty or len(aligned) < window:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid_a",
                    "source_uuid_b",
                    "is_delta",
                    "correlation",
                ]
            )

        corr = (
            aligned["signal_a"]
            .rolling(window=window, min_periods=max(2, window // 2))
            .corr(aligned["signal_b"])
        )
        out = pd.DataFrame(
            {
                "systime": aligned.index,
                "uuid": self.event_uuid,
                "source_uuid_a": uuid_a,
                "source_uuid_b": uuid_b,
                "is_delta": True,
                "correlation": corr.values,
            }
        ).dropna(subset=["correlation"])

        return out.reset_index(drop=True)

    def correlation_breakdown(
        self,
        uuid_a: str,
        uuid_b: str,
        *,
        resample: str = "1min",
        window: int = 60,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Detect periods where correlation drops below a threshold.

        Returns intervals where previously correlated signals diverge,
        which may indicate process issues.

        Args:
            uuid_a: UUID of first signal.
            uuid_b: UUID of second signal.
            resample: Resample interval for alignment.
            window: Rolling window size.
            threshold: Correlation threshold below which to flag.

        Returns:
            DataFrame: start, end, uuid, source_uuid_a, source_uuid_b,
                       is_delta, min_correlation, duration_seconds
        """
        corr_df = self.rolling_correlation(
            uuid_a, uuid_b, resample=resample, window=window
        )
        if corr_df.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "source_uuid_a",
                    "source_uuid_b",
                    "is_delta",
                    "min_correlation",
                    "duration_seconds",
                ]
            )

        corr_df["below"] = corr_df["correlation"] < threshold
        corr_df["group"] = (corr_df["below"] != corr_df["below"].shift()).cumsum()

        breakdowns = corr_df[corr_df["below"]].groupby("group")
        rows: List[Dict[str, Any]] = []
        for _, grp in breakdowns:
            rows.append(
                {
                    "start": grp["systime"].iloc[0],
                    "end": grp["systime"].iloc[-1],
                    "uuid": self.event_uuid,
                    "source_uuid_a": uuid_a,
                    "source_uuid_b": uuid_b,
                    "is_delta": True,
                    "min_correlation": grp["correlation"].min(),
                    "duration_seconds": (
                        grp["systime"].iloc[-1] - grp["systime"].iloc[0]
                    ).total_seconds(),
                }
            )

        return pd.DataFrame(rows)

    def lag_correlation(
        self,
        uuid_a: str,
        uuid_b: str,
        *,
        resample: str = "1min",
        max_lag: int = 30,
    ) -> pd.DataFrame:
        """Cross-correlation with time lag analysis.

        Finds the time lag at which two signals are most correlated.

        Args:
            uuid_a: UUID of first signal (reference).
            uuid_b: UUID of second signal (lagged).
            resample: Resample interval for alignment.
            max_lag: Maximum lag periods to test (in both directions).

        Returns:
            DataFrame: lag_periods, correlation, is_best_lag
        """
        aligned = self._align_signals(uuid_a, uuid_b, resample=resample)
        if aligned.empty or len(aligned) < max_lag * 2:
            return pd.DataFrame(columns=["lag_periods", "correlation", "is_best_lag"])

        a = aligned["signal_a"].values
        b = aligned["signal_b"].values
        n = len(a)

        rows: List[Dict[str, Any]] = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(a[:lag], b[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(a[lag:], b[: n - lag])[0, 1]
            else:
                corr = np.corrcoef(a, b)[0, 1]

            if not np.isnan(corr):
                rows.append({"lag_periods": lag, "correlation": corr})

        result = pd.DataFrame(rows)
        if result.empty:
            return pd.DataFrame(columns=["lag_periods", "correlation", "is_best_lag"])

        best_idx = result["correlation"].abs().idxmax()
        result["is_best_lag"] = False
        result.loc[best_idx, "is_best_lag"] = True

        return result
