import functools
import logging
import numbers
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs
from ts_shape.utils.intervals import interval_end_times

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_OLD_WINDOW_NAMES = {"window": "window_size", "resample": "window"}


def _legacy_window_kwargs(func: F) -> F:
    """Accept the pre-1.0 ``resample`` (grid) / ``window`` (row count) names.

    ``window`` now means the resample grid (an offset string) and the rolling
    row count is ``window_size``. A call is treated as old-style -- and warns
    -- when it passes ``resample`` or an integer ``window``.
    """
    renamed = renamed_kwargs(**_OLD_WINDOW_NAMES)(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        window = kwargs.get("window")
        old_window = isinstance(window, numbers.Integral) and not isinstance(
            window, bool
        )
        if "resample" in kwargs or old_window:
            return renamed(*args, **kwargs)
        return func(*args, **kwargs)

    wrapper.__deprecated_kwargs__ = dict(_OLD_WINDOW_NAMES)  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


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
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

    def _align_signals(
        self, uuid_a: str, uuid_b: str, window: str = "1min", keep_gaps: bool = False
    ) -> pd.DataFrame:
        """Align two signals on a common time index via resampling.

        With ``keep_gaps`` the full regular grid is returned (NaN where either
        signal has no data), so a shift of k rows is a lag of k periods and a
        rolling window of k rows spans k periods of real time.
        """
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

        a = a.set_index(self.time_column)[self.value_column].resample(window).mean()
        b = b.set_index(self.time_column)[self.value_column].resample(window).mean()

        aligned = pd.DataFrame({"signal_a": a, "signal_b": b})
        if keep_gaps:
            return aligned.loc[aligned.notna().any(axis=1).cummax()]
        return aligned.dropna()

    @_legacy_window_kwargs
    def rolling_correlation(
        self,
        uuid_a: str,
        uuid_b: str,
        *,
        window: str = "1min",
        window_size: int = 60,
    ) -> pd.DataFrame:
        """Compute rolling Pearson correlation between two signals.

        Args:
            uuid_a: UUID of first signal.
            uuid_b: UUID of second signal.
            window: Resample interval for alignment (offset string).
            window_size: Rolling window size, in resampled periods (rows).

        Returns:
            DataFrame: systime, uuid, source_uuid, source_uuid_a,
                       source_uuid_b, is_delta, correlation. ``source_uuid``
                       is ``"<uuid_a>,<uuid_b>"``.
        """
        # Keep the regular grid: rolling over dropna()'d rows let a window of
        # `window_size` rows span a multi-day gap.
        aligned = self._align_signals(uuid_a, uuid_b, window=window, keep_gaps=True)
        if aligned.dropna().empty or len(aligned) < window_size:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "source_uuid_a",
                    "source_uuid_b",
                    "is_delta",
                    "correlation",
                ]
            )

        corr = (
            aligned["signal_a"]
            .rolling(window=window_size, min_periods=max(2, window_size // 2))
            .corr(aligned["signal_b"])
        )
        out = pd.DataFrame(
            {
                "systime": aligned.index,
                "uuid": self.event_uuid,
                "source_uuid": f"{uuid_a},{uuid_b}",
                "source_uuid_a": uuid_a,
                "source_uuid_b": uuid_b,
                "is_delta": True,
                "correlation": corr.values,
            }
        ).dropna(subset=["correlation"])

        return out.reset_index(drop=True)

    @_legacy_window_kwargs
    def correlation_breakdown(
        self,
        uuid_a: str,
        uuid_b: str,
        *,
        window: str = "1min",
        window_size: int = 60,
        threshold: float = 0.5,
        interval_end: str = "last_sample",
    ) -> pd.DataFrame:
        """Detect periods where correlation drops below a threshold.

        Returns intervals where previously correlated signals diverge,
        which may indicate process issues.

        Args:
            uuid_a: UUID of first signal.
            uuid_b: UUID of second signal.
            window: Resample interval for alignment (offset string).
            window_size: Rolling window size, in resampled periods (rows).
            threshold: Correlation threshold below which to flag.
            interval_end: ``"last_sample"`` (default) ends an interval at its
                last flagged sample. ``"next_change"`` ends it at the next sample
                (the transition), so change-encoded signals -- one row per change --
                no longer collapse to zero-length intervals. See
                :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame: start, end, uuid, source_uuid_a, source_uuid_b,
                       is_delta, min_correlation, duration_seconds. Empty
                       (same columns) when no breakdown is found.
        """
        corr_df = self.rolling_correlation(
            uuid_a, uuid_b, window=window, window_size=window_size
        )
        cols = [
            "start",
            "end",
            "uuid",
            "source_uuid_a",
            "source_uuid_b",
            "is_delta",
            "min_correlation",
            "duration_seconds",
        ]
        if corr_df.empty:
            return pd.DataFrame(columns=cols)

        corr_df["below"] = corr_df["correlation"] < threshold
        corr_df["group"] = (corr_df["below"] != corr_df["below"].shift()).cumsum()

        corr_df = corr_df.reset_index(drop=True)
        ends = interval_end_times(corr_df["systime"], interval_end)
        breakdowns = corr_df[corr_df["below"]].groupby("group")
        rows: list[dict[str, Any]] = []
        for _, grp in breakdowns:
            end = ends.loc[grp.index[-1]]
            rows.append(
                {
                    "start": grp["systime"].iloc[0],
                    "end": end,
                    "uuid": self.event_uuid,
                    "source_uuid_a": uuid_a,
                    "source_uuid_b": uuid_b,
                    "is_delta": True,
                    "min_correlation": grp["correlation"].min(),
                    "duration_seconds": (end - grp["systime"].iloc[0]).total_seconds(),
                }
            )

        return pd.DataFrame(rows, columns=cols)

    @renamed_kwargs(resample="window")
    def lag_correlation(
        self,
        uuid_a: str,
        uuid_b: str,
        *,
        window: str = "1min",
        max_lag: int = 30,
    ) -> pd.DataFrame:
        """Cross-correlation with time lag analysis.

        Finds the time lag at which two signals are most correlated.

        Args:
            uuid_a: UUID of first signal (reference).
            uuid_b: UUID of second signal (lagged).
            window: Resample interval for alignment (offset string).
            max_lag: Maximum lag periods to test (in both directions).

        Returns:
            DataFrame: lag_periods, correlation, is_best_lag
        """
        # Shift on the regular grid (gaps kept as NaN) so a lag of k periods
        # is k * window of real time; shifting dropna()'d rows mixed lags
        # across gaps.
        aligned = self._align_signals(uuid_a, uuid_b, window=window, keep_gaps=True)
        if aligned.dropna().shape[0] < max_lag * 2:
            return pd.DataFrame(columns=["lag_periods", "correlation", "is_best_lag"])

        a = aligned["signal_a"]
        b = aligned["signal_b"]

        rows: list[dict[str, Any]] = []
        for lag in range(-max_lag, max_lag + 1):
            # lag > 0 pairs a[t] with b[t - lag]; pairs with a NaN are skipped.
            corr = a.corr(b.shift(lag))

            if not np.isnan(corr):
                rows.append({"lag_periods": lag, "correlation": corr})

        result = pd.DataFrame(rows)
        if result.empty:
            return pd.DataFrame(columns=["lag_periods", "correlation", "is_best_lag"])

        best_idx = result["correlation"].abs().idxmax()
        result["is_best_lag"] = False
        result.loc[best_idx, "is_best_lag"] = True

        return result
