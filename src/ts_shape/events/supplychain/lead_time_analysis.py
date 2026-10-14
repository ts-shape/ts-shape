"""Lead time analysis events for supply chain timeseries.

Matches order events to delivery events, computes lead time statistics,
and detects anomalous lead times.
"""

import logging

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_method, keyword_only

logger = logging.getLogger(__name__)


class LeadTimeAnalysisEvents(Base):
    """Analyze lead times between order placement and delivery in timeseries data.

    Order and delivery events are identified by separate UUIDs. When the
    delivery rows carry the order identifier, each order is matched to the
    first delivery of the same identifier at or after it; otherwise they are
    paired sequentially (first order to first delivery, etc.).

    Example usage:
        analyzer = LeadTimeAnalysisEvents(df)
        lead_times = analyzer.lead_times('order_signal', 'delivery_signal')
        stats = analyzer.lead_time_statistics('order_signal', 'delivery_signal')
        anomalies = analyzer.detect_lead_time_anomalies('order_signal', 'delivery_signal')
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "sc:lead_time",
        time_column: str = "systime",
    ) -> None:
        """Initialize lead time analyzer.

        Args:
            dataframe: Input DataFrame with timeseries data.
            event_uuid: UUID assigned to generated events.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column

    calculate_lead_times = deprecated_method("lead_times")

    @keyword_only(3)
    def lead_times(
        self,
        order_uuid: str,
        delivery_uuid: str,
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Match order events to delivery events.

        The *value_column* is used as an order identifier (e.g. PO number).
        When deliveries carry identifiers that occur among the orders, each
        order is matched to the first delivery with the same identifier at or
        after the order time (unmatched orders are left out). Otherwise the
        first order is paired with the first delivery, and so on.

        Args:
            order_uuid: UUID of the order placement signal.
            delivery_uuid: UUID of the delivery received signal.
            value_column: Column containing order identifiers.

        Returns:
            DataFrame with columns: start, end, duration_seconds, order_time,
            delivery_time, uuid, source_uuid, is_delta, lead_time_seconds,
            lead_time_hours, order_id. ``start``/``end`` are the order and
            delivery times; ``source_uuid`` is ``"<order_uuid>,<delivery_uuid>"``.
        """
        empty_cols = [
            "start",
            "end",
            "duration_seconds",
            "order_time",
            "delivery_time",
            "uuid",
            "source_uuid",
            "is_delta",
            "lead_time_seconds",
            "lead_time_hours",
            "order_id",
        ]

        orders = (
            self.dataframe[self.dataframe["uuid"] == order_uuid]
            .copy()
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )
        deliveries = (
            self.dataframe[self.dataframe["uuid"] == delivery_uuid]
            .copy()
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )

        if orders.empty or deliveries.empty:
            return pd.DataFrame(columns=empty_cols)

        orders[self.time_column] = pd.to_datetime(orders[self.time_column])
        deliveries[self.time_column] = pd.to_datetime(deliveries[self.time_column])

        has_ids = (
            value_column in orders.columns
            and value_column in deliveries.columns
            and deliveries[value_column].isin(orders[value_column].dropna()).any()
        )
        if has_ids:
            # Match by identifier: positional pairing mismatched every order
            # after an out-of-order or missing delivery.
            left = orders[[self.time_column, value_column]].rename(
                columns={self.time_column: "order_time", value_column: "order_id"}
            )
            right = deliveries[[self.time_column, value_column]].rename(
                columns={self.time_column: "delivery_time", value_column: "order_id"}
            )
            right = right.dropna(subset=["order_id"])
            right["_key"] = right["delivery_time"]
            left["_key"] = left["order_time"]
            matched = pd.merge_asof(
                left.sort_values("_key"),
                right.sort_values("_key"),
                on="_key",
                by="order_id",
                direction="forward",
            ).dropna(subset=["delivery_time"])
            order_times = matched["order_time"].reset_index(drop=True)
            delivery_times = matched["delivery_time"].reset_index(drop=True)
            order_ids = matched["order_id"].reset_index(drop=True)
        else:
            n_pairs = min(len(orders), len(deliveries))
            # Keep Series (not .values) so a tz-aware systime keeps its tz.
            order_times = orders[self.time_column].iloc[:n_pairs]
            delivery_times = deliveries[self.time_column].iloc[:n_pairs]
            order_ids = (
                orders[value_column].iloc[:n_pairs]
                if value_column in orders.columns
                else pd.Series([None] * n_pairs)
            )

        lead_seconds = (
            delivery_times.reset_index(drop=True) - order_times.reset_index(drop=True)
        ).dt.total_seconds()

        result = pd.DataFrame(
            {
                "start": order_times.reset_index(drop=True),
                "end": delivery_times.reset_index(drop=True),
                "duration_seconds": lead_seconds,
                "order_time": order_times.reset_index(drop=True),
                "delivery_time": delivery_times.reset_index(drop=True),
                "uuid": self.event_uuid,
                "source_uuid": f"{order_uuid},{delivery_uuid}",
                "is_delta": True,
                "lead_time_seconds": lead_seconds,
                "lead_time_hours": lead_seconds / 3600.0,
                "order_id": pd.Series(order_ids).reset_index(drop=True),
            }
        )

        return result

    @keyword_only(3)
    def lead_time_statistics(
        self,
        order_uuid: str,
        delivery_uuid: str,
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Compute summary statistics of lead times.

        Args:
            order_uuid: UUID of the order placement signal.
            delivery_uuid: UUID of the delivery received signal.
            value_column: Column containing order identifiers.

        Returns:
            Single-row DataFrame with columns: mean_hours, std_hours,
            min_hours, max_hours, p95_hours, count.
        """
        empty_cols = [
            "mean_hours",
            "std_hours",
            "min_hours",
            "max_hours",
            "p95_hours",
            "count",
        ]

        lt = self.lead_times(order_uuid, delivery_uuid, value_column=value_column)

        if lt.empty:
            return pd.DataFrame(columns=empty_cols)

        hours = lt["lead_time_hours"]
        result = pd.DataFrame(
            [
                {
                    "mean_hours": float(hours.mean()),
                    "std_hours": float(hours.std()) if len(hours) > 1 else 0.0,
                    "min_hours": float(hours.min()),
                    "max_hours": float(hours.max()),
                    "p95_hours": float(hours.quantile(0.95)),
                    "count": int(len(hours)),
                }
            ]
        )

        return result

    @keyword_only(3)
    def detect_lead_time_anomalies(
        self,
        order_uuid: str,
        delivery_uuid: str,
        threshold_factor: float = 2.0,
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Flag lead times exceeding mean + threshold_factor * std.

        Args:
            order_uuid: UUID of the order placement signal.
            delivery_uuid: UUID of the delivery received signal.
            threshold_factor: Number of standard deviations above mean
                to flag as anomalous.
            value_column: Column containing order identifiers.

        Returns:
            DataFrame with columns: systime, order_time, delivery_time, uuid,
            source_uuid, is_delta, lead_time_hours, z_score. ``systime`` is
            the delivery time (when the anomaly becomes known).
        """
        empty_cols = [
            "systime",
            "order_time",
            "delivery_time",
            "uuid",
            "source_uuid",
            "is_delta",
            "lead_time_hours",
            "z_score",
        ]

        lt = self.lead_times(order_uuid, delivery_uuid, value_column=value_column)

        if lt.empty or len(lt) < 2:
            return pd.DataFrame(columns=empty_cols)

        mean_h = lt["lead_time_hours"].mean()
        std_h = lt["lead_time_hours"].std()

        if std_h == 0 or np.isnan(std_h):
            return pd.DataFrame(columns=empty_cols)

        lt["z_score"] = (lt["lead_time_hours"] - mean_h) / std_h

        anomalies = lt[lt["z_score"] > threshold_factor].copy()

        if anomalies.empty:
            return pd.DataFrame(columns=empty_cols)

        anomalies["systime"] = anomalies["delivery_time"]
        return anomalies[empty_cols].reset_index(drop=True)
