"""Lead time analysis events for supply chain timeseries.

Matches order events to delivery events, computes lead time statistics,
and detects anomalous lead times.
"""

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import twin_columns
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    caller_stacklevel,
    deprecated_method,
    keyword_only,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


def _pair_in_order(
    order_times: pd.Series, delivery_times: pd.Series
) -> tuple[list[int], list[int]]:
    """Positions of FIFO pairs: each order with the next unused delivery >= it.

    Both inputs are sorted. Pairs never cross and a delivery never precedes
    its order.
    """
    orders = order_times.to_numpy()
    deliveries = delivery_times.to_numpy()
    order_pos: list[int] = []
    delivery_pos: list[int] = []
    j = 0
    for i, t in enumerate(orders):
        if j >= len(deliveries):
            break
        if pd.isna(t):
            continue
        while j < len(deliveries) and (pd.isna(deliveries[j]) or deliveries[j] < t):
            j += 1
        if j < len(deliveries):
            order_pos.append(i)
            delivery_pos.append(j)
            j += 1
    return order_pos, delivery_pos


def _pair_by_id(
    order_times: pd.Series,
    order_ids: pd.Series,
    delivery_times: pd.Series,
    delivery_ids: pd.Series,
) -> tuple[list[int], list[int]]:
    """Positions of FIFO pairs within each identifier (:func:`_pair_in_order`).

    Inputs are sorted by time. Rows without an identifier stay unpaired.
    The pairs come back in order-time order.

    The greedy FIFO pairing of a group is its k-th order with its k-th
    delivery exactly when every such pair has the delivery at or after the
    order, so those groups are paired in one vectorized join; only groups
    with an early delivery (or a missing time) go through the per-group
    loop.
    """

    def ranked(ids: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame({"key": ids.to_numpy(), "pos": np.arange(len(ids))})
        frame = frame[ids.notna().to_numpy()]
        frame["rank"] = frame.groupby("key", sort=False).cumcount()
        return frame

    o_rank, d_rank = ranked(order_ids), ranked(delivery_ids)
    pairs = o_rank.merge(d_rank, on=["key", "rank"], suffixes=("_o", "_d"))
    o_t = order_times.array
    d_t = delivery_times.array
    po = pairs["pos_o"].to_numpy()
    dp = pairs["pos_d"].to_numpy()
    ok = d_t[dp] >= o_t[po]  # NaT compares False: such groups take the loop
    bad_keys = pd.unique(pairs["key"].to_numpy()[~ok])
    keep = ~pairs["key"].isin(bad_keys).to_numpy()
    order_pos: list[int] = po[keep].tolist()
    delivery_pos: list[int] = dp[keep].tolist()
    if len(bad_keys):
        o_groups = o_rank[o_rank["key"].isin(bad_keys)].groupby("key", sort=False)
        d_groups = d_rank[d_rank["key"].isin(bad_keys)].groupby("key", sort=False)
        d_idx_by_key = {k: g["pos"].to_numpy() for k, g in d_groups}
        for key, group in o_groups:
            o_idx = group["pos"].to_numpy()
            d_idx = d_idx_by_key[key]
            o_sub, d_sub = _pair_in_order(
                order_times.iloc[o_idx], delivery_times.iloc[d_idx]
            )
            order_pos.extend(o_idx[o_sub].tolist())
            delivery_pos.extend(d_idx[d_sub].tolist())
    if not order_pos:
        return [], []
    # Positions index time-sorted frames: sorting by order position orders
    # the pairs by order time (ties keep their row order).
    order_arr = np.asarray(order_pos)
    rank = np.argsort(order_arr, kind="stable")
    return order_arr[rank].tolist(), np.asarray(delivery_pos)[rank].tolist()


class LeadTimeAnalysisEvents(Base):
    """Analyze lead times between order placement and delivery in timeseries data.

    Order and delivery events are identified by separate UUIDs. When the
    delivery rows carry the order identifier, orders and deliveries are
    paired one-to-one within each identifier: each order, in time order,
    with the first not yet used delivery of the same identifier at or after
    it. Otherwise the same FIFO pairing runs over all orders and
    deliveries.

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

    if TYPE_CHECKING:

        @deprecated(
            "LeadTimeAnalysisEvents.calculate_lead_times() is deprecated, use .lead_times()"
        )
        def calculate_lead_times(
            self,
            order_uuid: str,
            delivery_uuid: str,
            value_column: str = "value_string",
        ) -> pd.DataFrame: ...

    else:
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
        Each order, in time order, is paired with the first delivery at or
        after it that is not paired yet. When deliveries carry identifiers
        that occur among the orders, this FIFO pairing runs within each
        identifier (one delivery per order, so two orders with the same
        identifier need two deliveries; a multi-line purchase order needs
        line-level identifiers such as ``"PO-1/10"``); otherwise it runs over
        all orders and deliveries. Orders and deliveries left without a
        partner (including rows without an identifier on the identifier
        path) are dropped with a
        :class:`~ts_shape.errors.DataQualityWarning`. A lead time is never
        negative. Rows come in order-time order.

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
            # Match by identifier, FIFO one-to-one within each identifier: a
            # forward as-of join reused one delivery for every order of the
            # same identifier and ignored the later deliveries.
            order_pos, delivery_pos = _pair_by_id(
                orders[self.time_column],
                orders[value_column],
                deliveries[self.time_column],
                deliveries[value_column],
            )
        else:
            order_pos, delivery_pos = _pair_in_order(
                orders[self.time_column], deliveries[self.time_column]
            )
        n_orphans = len(orders) + len(deliveries) - 2 * len(order_pos)
        if n_orphans:
            warnings.warn(
                f"lead_times: {len(orders) - len(order_pos)} order(s) without "
                f"a later delivery and {len(deliveries) - len(delivery_pos)} "
                "delivery(ies) without an earlier open order were dropped",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
        # Keep Series (not .values) so a tz-aware systime keeps its tz.
        order_times = orders[self.time_column].iloc[order_pos]
        delivery_times = deliveries[self.time_column].iloc[delivery_pos]
        order_ids = (
            orders[value_column].iloc[order_pos]
            if value_column in orders.columns
            else pd.Series([None] * len(order_pos))
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
    @twin_columns({"count": "order_count"})
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
            min_hours, max_hours, p95_hours, count, order_count (the number
            of matched orders).

            Canonical twin columns: ``order_count`` (after ``count``); it holds the same
            values as the deprecated column right before it, which is removed
            in the next major release.
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
                    "count": len(hours),
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
