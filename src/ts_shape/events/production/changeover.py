import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any, Optional, Callable

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ChangeoverEvents(Base):
    """Production: Changeover

    Detect product/recipe changes and compute changeover windows without
    requiring a dedicated 'first good' signal.

    Methods:
    - detect_changeover: point events when product/recipe changes.
    - changeover_window: derive an end time via fixed window or 'stable_band' metrics.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "prod:changeover",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column

    def detect_changeover(
        self,
        product_uuid: str,
        *,
        value_column: str = "value_string",
        min_hold: str = "0s",
    ) -> pd.DataFrame:
        """Emit point events when the product/recipe changes value.

        Uses a hold check: the new product must persist for at least min_hold
        until the next change.
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
        start_time: Optional[pd.Timestamp] = None,
        until: str = "fixed_window",
        config: Optional[Dict[str, Any]] = None,
        fallback: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compute changeover windows per product change with enhanced configurability.

        until:
          - fixed_window: end = start + config['duration'] (e.g., '10m')
          - stable_band: end when all metrics stabilize within band for hold:
                config = {
                  'metrics': [
                    {'uuid': 'm1', 'value_column': 'value_double', 'band': 0.2, 'hold': '2m'},
                    ...
                  ],
                  'reference_method': 'expanding_median' | 'rolling_mean' | 'ewma' | 'target_value',
                  'rolling_window': 5,  # for rolling_mean (number of points)
                  'ewma_span': 10,  # for ewma
                  'target_values': {'m1': 100.0, ...}  # for target_value
                }
        fallback: {'default_duration': '10m', 'completed': False}
        """
        config = config or {}
        fallback = fallback or {"default_duration": "10m", "completed": False}

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
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "method",
                    "completed",
                ]
            )

        rows: List[Dict[str, Any]] = []
        for _, r in changes.iterrows():
            t0 = pd.to_datetime(r["systime"])
            if until == "fixed_window":
                duration = pd.to_timedelta(config.get("duration", "10m"))
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
            end = t0 + pd.to_timedelta(fallback.get("default_duration", "10m"))
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

        return pd.DataFrame(rows)

    def _compute_stable_band_end(
        self, t0: pd.Timestamp, config: Dict[str, Any]
    ) -> Optional[pd.Timestamp]:
        """Compute end time for stable_band method with configurable reference methods."""
        metric_defs = config.get("metrics", [])
        reference_method = config.get("reference_method", "expanding_median")

        metric_ends: List[pd.Timestamp] = []

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

            # Find first stable period
            gid = (inside.ne(inside.shift())).cumsum()
            end_found: Optional[pd.Timestamp] = None

            for _, seg in s.groupby(gid):
                seg_inside = inside.loc[seg.index]
                if not seg_inside.iloc[0]:
                    continue
                start_seg = seg[self.time_column].iloc[0]
                end_seg = seg[self.time_column].iloc[-1]
                if (end_seg - start_seg) >= hold_td:
                    end_found = start_seg
                    break

            if end_found is not None:
                metric_ends.append(end_found)

        if metric_defs and len(metric_ends) == len(metric_defs):
            return max(metric_ends)

        return None

    def _calculate_reference(
        self,
        series: pd.Series,
        method: str,
        config: Dict[str, Any],
        mdef: Dict[str, Any],
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
        """Compute quality metrics for changeovers.

        Returns metrics including:
        - changeover duration patterns
        - frequency statistics
        - time between changeovers
        - product-specific metrics
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
