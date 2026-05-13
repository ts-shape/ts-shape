"""Batch / recipe production tracking.

Detect batch boundaries from a string signal that carries the current batch ID,
compute duration statistics, per-batch yield, and batch transition matrices.
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class BatchTrackingEvents(Base):
    """Track batch/recipe production from a batch-ID string signal.

    A "batch" is defined as a contiguous period where the batch-ID value
    remains constant.  A new batch begins when the value changes.

    Example usage:
        batches = BatchTrackingEvents(df, batch_uuid='batch_id_signal')

        detected = batches.detect_batches()
        stats = batches.batch_duration_stats()
        yields = batches.batch_yield('part_counter')
        transitions = batches.batch_transition_matrix()
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        batch_uuid: str,
        *,
        event_uuid: str = "prod:batch",
        value_column: str = "value_string",
        time_column: str = "systime",
    ) -> None:
        """Initialize batch tracker.

        Args:
            dataframe: Input DataFrame with timeseries data.
            batch_uuid: UUID of the batch-ID signal.
            event_uuid: UUID to tag derived events with.
            value_column: Column holding the batch ID string.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, column_name=time_column)
        self.batch_uuid = batch_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.batch_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect_batches(self) -> pd.DataFrame:
        """Detect batch start/end from value changes in the batch-ID signal.

        Returns:
            DataFrame with columns:
            - batch_id: The batch identifier string.
            - start: Timestamp of the first sample in the batch.
            - end: Timestamp of the last sample in the batch.
            - duration_seconds: Duration of the batch.
            - sample_count: Number of data points in the batch.
            - uuid: Event UUID.
            - source_uuid: Source signal UUID.
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "batch_id",
                    "start",
                    "end",
                    "duration_seconds",
                    "sample_count",
                    "uuid",
                    "source_uuid",
                ]
            )

        s = self.series[[self.time_column, self.value_column]].copy()
        s["batch_id"] = s[self.value_column].fillna("")
        s["group"] = (s["batch_id"] != s["batch_id"].shift()).cumsum()

        rows: List[Dict[str, Any]] = []
        for _, seg in s.groupby("group"):
            batch_id = seg["batch_id"].iloc[0]
            if batch_id == "":
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            rows.append(
                {
                    "batch_id": batch_id,
                    "start": start,
                    "end": end,
                    "duration_seconds": (end - start).total_seconds(),
                    "sample_count": len(seg),
                    "uuid": self.event_uuid,
                    "source_uuid": self.batch_uuid,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Duration statistics
    # ------------------------------------------------------------------

    def batch_duration_stats(self) -> pd.DataFrame:
        """Compute duration statistics grouped by batch type (batch_id).

        Returns:
            DataFrame with columns:
            - batch_id
            - count: Number of occurrences of this batch type.
            - min_duration_seconds
            - avg_duration_seconds
            - max_duration_seconds
            - total_duration_seconds
        """
        batches = self.detect_batches()

        if batches.empty:
            return pd.DataFrame(
                columns=[
                    "batch_id",
                    "count",
                    "min_duration_seconds",
                    "avg_duration_seconds",
                    "max_duration_seconds",
                    "total_duration_seconds",
                ]
            )

        stats = (
            batches.groupby("batch_id")["duration_seconds"]
            .agg(
                count="count",
                min_duration_seconds="min",
                avg_duration_seconds="mean",
                max_duration_seconds="max",
                total_duration_seconds="sum",
            )
            .reset_index()
        )

        # Round for readability
        for col in [
            "min_duration_seconds",
            "avg_duration_seconds",
            "max_duration_seconds",
            "total_duration_seconds",
        ]:
            stats[col] = stats[col].round(2)

        return stats

    # ------------------------------------------------------------------
    # Batch yield
    # ------------------------------------------------------------------

    def batch_yield(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = "value_integer",
    ) -> pd.DataFrame:
        """Compute production quantity for each detected batch.

        Uses a monotonic counter signal.  The yield per batch is the
        counter increase during the batch interval.

        Args:
            counter_uuid: UUID of the monotonic part counter.
            value_column_counter: Column holding counter values.

        Returns:
            DataFrame with columns:
            - batch_id
            - start
            - end
            - duration_seconds
            - quantity: Parts produced during the batch.
            - uuid
            - source_uuid
        """
        batches = self.detect_batches()

        if batches.empty:
            return pd.DataFrame(
                columns=[
                    "batch_id",
                    "start",
                    "end",
                    "duration_seconds",
                    "quantity",
                    "uuid",
                    "source_uuid",
                ]
            )

        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if counter_data.empty:
            batches["quantity"] = 0
            return batches[
                [
                    "batch_id",
                    "start",
                    "end",
                    "duration_seconds",
                    "quantity",
                    "uuid",
                    "source_uuid",
                ]
            ]

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        quantities: List[int] = []
        for _, batch_row in batches.iterrows():
            mask = (counter_data[self.time_column] >= batch_row["start"]) & (
                counter_data[self.time_column] <= batch_row["end"]
            )
            batch_counter = counter_data.loc[mask, value_column_counter]
            if batch_counter.empty or len(batch_counter) < 2:
                quantities.append(0)
            else:
                qty = int(batch_counter.iloc[-1] - batch_counter.iloc[0])
                quantities.append(max(0, qty))

        batches = batches.copy()
        batches["quantity"] = quantities
        return batches[
            [
                "batch_id",
                "start",
                "end",
                "duration_seconds",
                "quantity",
                "uuid",
                "source_uuid",
            ]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def batch_transition_matrix(self) -> pd.DataFrame:
        """Build a transition frequency matrix: which batch follows which.

        Returns:
            DataFrame (pivot table) where index = from_batch, columns = to_batch,
            values = transition count.  An extra ``total`` column is appended.
        """
        batches = self.detect_batches()

        if batches.empty or len(batches) < 2:
            return pd.DataFrame()

        from_batch = batches["batch_id"].iloc[:-1].values
        to_batch = batches["batch_id"].iloc[1:].values

        transitions = pd.DataFrame({"from_batch": from_batch, "to_batch": to_batch})
        matrix = (
            transitions.groupby(["from_batch", "to_batch"]).size().unstack(fill_value=0)
        )
        matrix["total"] = matrix.sum(axis=1)
        return matrix
