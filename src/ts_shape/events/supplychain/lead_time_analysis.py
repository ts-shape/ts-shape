"""Lead time analysis events for supply chain timeseries.

Matches order events to delivery events, computes lead time statistics,
and detects anomalous lead times.
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class LeadTimeAnalysisEvents(Base):
    """Analyze lead times between order placement and delivery in timeseries data.

    Order and delivery events are identified by separate UUIDs. They are
    paired sequentially (first order to first delivery, etc.).

    Example usage:
        analyzer = LeadTimeAnalysisEvents(df)
        lead_times = analyzer.calculate_lead_times('order_signal', 'delivery_signal')
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
        super().__init__(dataframe, column_name=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column

    def calculate_lead_times(
        self,
        order_uuid: str,
        delivery_uuid: str,
        value_column: str = "value_string",
    ) -> pd.DataFrame:
        """Match order events to delivery events by sequential pairing.

        The first order is paired with the first delivery, and so on.
        The *value_column* is used as an order identifier (e.g. PO number).

        Args:
            order_uuid: UUID of the order placement signal.
            delivery_uuid: UUID of the delivery received signal.
            value_column: Column containing order identifiers.

        Returns:
            DataFrame with columns: order_time, delivery_time, uuid, is_delta,
            lead_time_seconds, lead_time_hours, order_id.
        """
        empty_cols = [
            "order_time",
            "delivery_time",
            "uuid",
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

        n_pairs = min(len(orders), len(deliveries))

        order_times = orders[self.time_column].iloc[:n_pairs].values
        delivery_times = deliveries[self.time_column].iloc[:n_pairs].values
        order_ids = (
            orders[value_column].iloc[:n_pairs].values
            if value_column in orders.columns
            else [None] * n_pairs
        )

        lead_seconds = pd.to_timedelta(delivery_times - order_times).total_seconds()

        result = pd.DataFrame(
            {
                "order_time": order_times,
                "delivery_time": delivery_times,
                "uuid": self.event_uuid,
                "is_delta": True,
                "lead_time_seconds": lead_seconds,
                "lead_time_hours": lead_seconds / 3600.0,
                "order_id": order_ids,
            }
        )

        return result

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

        lt = self.calculate_lead_times(order_uuid, delivery_uuid, value_column)

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
            DataFrame with columns: order_time, delivery_time, uuid, is_delta,
            lead_time_hours, z_score.
        """
        empty_cols = [
            "order_time",
            "delivery_time",
            "uuid",
            "is_delta",
            "lead_time_hours",
            "z_score",
        ]

        lt = self.calculate_lead_times(order_uuid, delivery_uuid, value_column)

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

        return anomalies[
            [
                "order_time",
                "delivery_time",
                "uuid",
                "is_delta",
                "lead_time_hours",
                "z_score",
            ]
        ].reset_index(drop=True)
