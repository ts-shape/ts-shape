import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.runs import true_runs

logger = logging.getLogger(__name__)


class ChangeoverEvents(Base):
    """Production: Changeover

    Detect product/recipe changes and compute changeover windows without
    requiring a dedicated 'first good' signal.

    Methods:
    - detect_changeover: point events when product/recipe changes.
    - changeover_window: derive an end time via fixed window or 'stable_band' metrics.
    - changeover_quality_metrics: per-product changeover counts and time between changeovers.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "prod:changeover",
        time_column: str = "systime",
    ) -> None:
        """Initialize the changeover detector.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            event_uuid: UUID stamped on emitted events.
            time_column: Timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column

    def detect_changeover(
        self,
        product_uuid: str,
        *,
        value_column: str = "value_string",
        min_hold: str = "0s",
        include_initial: bool = True,
    ) -> pd.DataFrame:
        """Emit point events when the product/recipe changes value.

        Uses a hold check: the new product must persist for at least min_hold
        until the next change.

        Args:
            product_uuid: UUID of the product/recipe signal.
            value_column: Column containing the product value.
            min_hold: Minimum time the new product must persist to count.
            include_initial: When True (default) the first reading is emitted
                as a changeover event, marking the product active at the start
                of the data. A changeover strictly requires a *prior, different*
                value, so pass ``include_initial=False`` to suppress this
                synthetic leading event and report only real value changes.
        """
        p = (
            self.dataframe[self.dataframe["uuid"] == product_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if p.empty:
            return pd.DataFrame(
                columns=["systime", "uuid", "source_uuid", "is_delta", "new_value"]
            )
        p[self.time_column] = pd.to_datetime(p[self.time_column])
        series = p[value_column]
        changed = series.ne(series.shift())
        if not include_initial and len(changed) > 0:
            # series.shift() yields NaN for the first row, so ne(...) always
            # flags it. That is the product being active at the window start,
            # not a changeover.
            changed.iloc[0] = False
        change_times = p.loc[changed, self.time_column]
        min_td = pd.to_timedelta(min_hold)
        next_change = change_times.shift(-1)
        ok = (next_change - change_times >= min_td) | next_change.isna()
        change_times = change_times[ok]
        out = p[p[self.time_column].isin(change_times)][
            [self.time_column, value_column]
        ].rename(columns={self.time_column: "systime", value_column: "new_value"})
        out["uuid"] = self.event_uuid
        out["source_uuid"] = product_uuid
        out["is_delta"] = True
        return out[["systime", "uuid", "source_uuid", "is_delta", "new_value"]]

    def changeover_window(
        self,
        product_uuid: str,
        *,
        value_column: str = "value_string",
        start_time: pd.Timestamp | None = None,
        until: str = "fixed_window",
        config: dict[str, Any] | None = None,
        fallback: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compute changeover windows per product change with enhanced configurability.

        Args:
            product_uuid: UUID of the product/recipe signal.
            value_column: Column containing the product value.
            start_time: Only changes at or after this timestamp are used. None
                uses all changes.
            until: How each window's end is found:

                - ``"fixed_window"``: end = start + ``config['duration']``
                  (default ``'10min'``).
                - ``"stable_band"``: end when every metric in
                  ``config['metrics']`` stays within its band of a reference
                  for its ``hold``; the window ends at the latest metric's
                  stable start.

                Any other value, or a stable band that is never reached, uses
                ``fallback``.
            config: Method settings. Keys: ``'min_hold'`` (minimum hold of a
                new product, passed to :meth:`detect_changeover`, default
                ``'0s'``), ``'duration'`` (for ``fixed_window``), and for
                ``stable_band``::

                    {
                      'metrics': [
                        {'uuid': 'm1', 'value_column': 'value_double',
                         'band': 0.2, 'hold': '2min'},
                        ...
                      ],
                      'reference_method': 'expanding_median' | 'rolling_mean'
                                          | 'ewma' | 'target_value',
                      'rolling_window': 5,  # for rolling_mean (number of points)
                      'ewma_span': 10,  # for ewma
                      'target_values': {'m1': 100.0, ...}  # for target_value
                    }

                None means an empty dict.
            fallback: Used when the window end cannot be determined:
                ``{'default_duration': '10min', 'completed': False}`` (also the
                default when None). ``completed`` is copied to the output.

        Returns:
            DataFrame with columns: start, end, duration_seconds (float
            seconds, end - start), uuid, source_uuid, is_delta, method,
            completed.
        """
        config = config or {}
        fallback = fallback or {"default_duration": "10min", "completed": False}

        changes = self.detect_changeover(
            product_uuid,
            value_column=value_column,
            min_hold=config.get("min_hold", "0s"),
        )
        if start_time is not None:
            changes = changes[changes["systime"] >= pd.to_datetime(start_time)]
        if changes.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "method",
                    "completed",
                ]
            )

        rows: list[dict[str, Any]] = []
        for _, r in changes.iterrows():
            t0 = pd.to_datetime(r["systime"])
            if until == "fixed_window":
                duration = pd.to_timedelta(config.get("duration", "10min"))
                end = t0 + duration
                rows.append(
                    {
                        "start": t0,
                        "end": end,
                        "uuid": self.event_uuid,
                        "source_uuid": product_uuid,
                        "is_delta": True,
                        "method": "fixed_window",
                        "completed": True,
                    }
                )
                continue

            if until == "stable_band":
                result = self._compute_stable_band_end(t0, config)
                if result is not None:
                    rows.append(
                        {
                            "start": t0,
                            "end": result,
                            "uuid": self.event_uuid,
                            "source_uuid": product_uuid,
                            "is_delta": True,
                            "method": "stable_band",
                            "completed": True,
                        }
                    )
                    continue

            # fallback
            end = t0 + pd.to_timedelta(fallback.get("default_duration", "10min"))
            rows.append(
                {
                    "start": t0,
                    "end": end,
                    "uuid": self.event_uuid,
                    "source_uuid": product_uuid,
                    "is_delta": True,
                    "method": until,
                    "completed": bool(fallback.get("completed", False)),
                }
            )

        out = pd.DataFrame(rows)
        out.insert(
            2, "duration_seconds", (out["end"] - out["start"]).dt.total_seconds()
        )
        return out

    def _compute_stable_band_end(
        self, t0: pd.Timestamp, config: dict[str, Any]
    ) -> pd.Timestamp | None:
        """Compute end time for stable_band method with configurable reference methods."""
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

            # Calculate reference based on method
            ref = self._calculate_reference(s[vcol], reference_method, config, mdef)

            # Check stability
            inside = (s[vcol] - ref).abs() <= band
            if not inside.any():
                continue

            # Find the first stable run (inside the band) lasting at least
            # the hold time.
            end_found: pd.Timestamp | None
            if not s.index.is_unique:
                end_found = self._stable_start_by_labels(s, inside, hold_td)
            else:
                runs = true_runs(inside.to_numpy(dtype=bool))
                times = s[self.time_column].array
                long_enough = np.flatnonzero(
                    (times[runs.last] - times[runs.starts]) >= hold_td
                )
                end_found = (
                    times[runs.starts[long_enough[0]]] if long_enough.size else None
                )

            if end_found is not None:
                metric_ends.append(end_found)

        if metric_defs and len(metric_ends) == len(metric_defs):
            return max(metric_ends)

        return None

    def _stable_start_by_labels(
        self, s: pd.DataFrame, inside: pd.Series, hold_td: pd.Timedelta
    ) -> pd.Timestamp | None:
        """Label-based run search, kept for frames with duplicate index labels
        (whose ``.loc`` lookups the vectorized search does not reproduce)."""
        gid = (inside.ne(inside.shift())).cumsum()
        for _, seg in s.groupby(gid):
            seg_inside = inside.loc[seg.index]
            if not seg_inside.iloc[0]:
                continue
            start_seg = seg[self.time_column].iloc[0]
            end_seg = seg[self.time_column].iloc[-1]
            if (end_seg - start_seg) >= hold_td:
                return start_seg
        return None

    def _calculate_reference(
        self,
        series: pd.Series,
        method: str,
        config: dict[str, Any],
        mdef: dict[str, Any],
    ) -> pd.Series:
        """Calculate reference values using various methods."""
        if method == "expanding_median":
            return series.expanding(min_periods=3).median()
        elif method == "rolling_mean":
            window_size = config.get("rolling_window", max(3, int(len(series) * 0.1)))
            return series.rolling(window=window_size, min_periods=3).mean()
        elif method == "ewma":
            span = config.get("ewma_span", 10)
            return series.ewm(span=span, min_periods=3).mean()
        elif method == "target_value":
            target_values = config.get("target_values", {})
            target = target_values.get(mdef["uuid"], series.median())
            return pd.Series([target] * len(series), index=series.index)
        else:
            # Default to expanding median
            return series.expanding(min_periods=3).median()

    def changeover_quality_metrics(
        self,
        product_uuid: str,
        *,
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Per-product changeover frequency: how often, and how far apart.

        For each product, counts the changeovers *to* that product and
        summarizes the time between consecutive ones. Changeover durations
        are not measured here (a product-signal change is a point, not an
        interval); use :meth:`changeover_window` for when the process is
        stable again after a changeover.

        Args:
            product_uuid: UUID of the product/recipe signal.
            value_column: Column containing the product value.

        Returns:
            DataFrame with one row per product: product, changeover_count,
            avg_time_between_seconds, min_time_between_seconds,
            max_time_between_seconds, std_time_between_seconds (time between
            consecutive changeovers to that product; None with fewer than
            two). Empty when there are fewer than two changeovers.
        """
        changes = self.detect_changeover(product_uuid, value_column=value_column)

        if changes.empty or len(changes) < 2:
            return pd.DataFrame(
                columns=[
                    "product",
                    "changeover_count",
                    "avg_time_between_seconds",
                    "min_time_between_seconds",
                    "max_time_between_seconds",
                    "std_time_between_seconds",
                ]
            )

        # Group by product (new_value)
        product_metrics = []
        for product in changes["new_value"].unique():
            product_changes = changes[changes["new_value"] == product]
            product_times = product_changes["systime"].sort_values()

            if len(product_times) < 2:
                metrics = {
                    "product": product,
                    "changeover_count": len(product_changes),
                    "avg_time_between_seconds": None,
                    "min_time_between_seconds": None,
                    "max_time_between_seconds": None,
                    "std_time_between_seconds": None,
                }
            else:
                product_diffs = product_times.diff().dt.total_seconds().dropna()

                metrics = {
                    "product": product,
                    "changeover_count": len(product_changes),
                    "avg_time_between_seconds": product_diffs.mean(),
                    "min_time_between_seconds": product_diffs.min(),
                    "max_time_between_seconds": product_diffs.max(),
                    "std_time_between_seconds": product_diffs.std(),
                }
            product_metrics.append(metrics)

        return pd.DataFrame(product_metrics)
