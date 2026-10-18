"""Batch / recipe production tracking.

Detect batch boundaries from a string signal that carries the current batch ID,
compute duration statistics, per-batch yield, and batch transition matrices.
"""

import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events.production._counter_groups import counter_totals
from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_quantity
from ts_shape.utils.deprecation import deprecated_method
from ts_shape.utils.runs import run_bounds, total_seconds

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


class BatchTrackingEvents(Base):
    """Track batch/recipe production from a batch-ID string signal.

    A "batch" is defined as a contiguous period where the batch-ID value
    remains constant.  A new batch begins when the value changes.

    Example usage:
        batches = BatchTrackingEvents(df, batch_uuid='batch_id_signal')

        detected = batches.detect_batches()
        stats = batches.batch_duration_statistics()
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
        super().__init__(dataframe, time_column=time_column)
        self.batch_uuid = batch_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, batch_uuid)

        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.batch_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def _segments(self) -> pd.DataFrame:
        """Batch-ID samples with a ``group`` number per contiguous segment."""
        s = self.series[[self.time_column, self.value_column]].copy()
        s["batch_id"] = s[self.value_column].fillna("")
        s["group"] = (s["batch_id"] != s["batch_id"].shift()).cumsum()
        return s

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

        s = self._segments()
        runs = run_bounds(s["batch_id"])
        ids = s["batch_id"].iloc[runs.starts]
        # An empty batch ID (missing value) is a gap between batches.
        keep = (ids != "").to_numpy(dtype=bool)
        if not keep.any():
            return pd.DataFrame([])
        times = s[self.time_column].array
        start = times[runs.starts[keep]]
        end = times[runs.last[keep]]
        return pd.DataFrame(
            {
                "batch_id": ids[keep].tolist(),
                "start": start,
                "end": end,
                "duration_seconds": total_seconds(end - start),
                "sample_count": runs.lengths[keep],
                "uuid": self.event_uuid,
                "source_uuid": self.batch_uuid,
            }
        )

    # ------------------------------------------------------------------
    # Duration statistics
    # ------------------------------------------------------------------

    if TYPE_CHECKING:

        @deprecated(
            "BatchTrackingEvents.batch_duration_stats() is deprecated, use .batch_duration_statistics()"
        )
        def batch_duration_stats(self) -> pd.DataFrame: ...

    else:
        batch_duration_stats = deprecated_method("batch_duration_statistics")

    @twin_columns({"count": "batch_count"})
    def batch_duration_statistics(self) -> pd.DataFrame:
        """Compute duration statistics grouped by batch type (batch_id).

        Returns:
            DataFrame with columns:
            - batch_id
            - count: deprecated twin of ``batch_count``.
            - batch_count: Number of occurrences of this batch type.
            - min_duration_seconds
            - avg_duration_seconds
            - max_duration_seconds
            - total_duration_seconds

            Canonical twin columns: ``batch_count`` (after ``count``); it holds the same
            values as the deprecated column right before it, which is removed
            in the next major release.
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

        return stats

    # ------------------------------------------------------------------
    # Batch yield
    # ------------------------------------------------------------------

    def batch_yield(
        self,
        counter_uuid: str,
        *,
        value_column_counter: str = "value_integer",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Compute production quantity for each detected batch.

        Uses a monotonic counter signal. With ``handle_resets=True`` the
        reset-aware increments are computed once over the whole counter and
        each is booked to the batch that was running when it was produced: a
        reading reports the production since the previous reading, so it
        belongs to the batch whose ID was last seen strictly before it. A batch
        therefore owns the counter readings after its first sample up to and
        including the next batch-ID change, the production between the last
        sample of one batch and the first sample of the next counts for the
        earlier batch, and the batch quantities add up to the counter total
        (readings before the first batch-ID sample, after the last one, or
        while the batch ID is empty belong to no batch).

        Args:
            counter_uuid: UUID of the monotonic part counter.
            value_column_counter: Column holding counter values.
            handle_resets: When True (default), treat a decreasing counter as a reset and
                count the production after the reset instead of losing it, so
                ``quantity`` is the reset-aware total for the batch. The result
                gains a ``resets`` column with the number of resets observed in
                each batch. When False, ``quantity`` is simply
                ``max(0, last - first)`` of the readings between the batch's
                first and last sample (0 for fewer than two readings): a batch
                containing a reset reports too little and production between
                batches is lost.

        Returns:
            DataFrame with columns:
            - batch_id
            - start
            - end
            - duration_seconds
            - quantity: Parts produced during the batch (see above).
            - uuid
            - source_uuid
            - resets (only when ``handle_resets=True``)
        """
        base_columns = [
            "batch_id",
            "start",
            "end",
            "duration_seconds",
            "quantity",
            "uuid",
            "source_uuid",
        ]
        if handle_resets:
            base_columns.append("resets")

        batches = self.detect_batches()

        if batches.empty:
            return pd.DataFrame(columns=base_columns)

        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if counter_data.empty:
            batches = batches.copy()
            batches["quantity"] = 0
            if handle_resets:
                batches["resets"] = 0
            return batches[base_columns]

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])

        quantities: list[int] = []
        resets: list[int] = []
        if handle_resets:
            # Book every reset-aware increment of the whole counter signal to
            # the batch that was running when it was produced: a reading
            # reports the production since the previous reading, so it belongs
            # to the batch whose ID was last seen strictly before it. Batch k
            # thus owns the readings in (its first sample, the next batch-ID
            # change], and the production between the last sample of one batch
            # and the first sample of the next is no longer lost.
            segs = self._segments()
            labels = segs.groupby("group")["batch_id"].first()
            batch_groups = labels[labels != ""].index
            readings = counter_data[[self.time_column, value_column_counter]]
            readings = pd.merge_asof(
                readings.reset_index(drop=True),
                segs[[self.time_column, "group"]],
                on=self.time_column,
                direction="backward",
                allow_exact_matches=False,
            )
            # After the last batch-ID sample nothing is known about the batch.
            after_end = readings[self.time_column] > segs[self.time_column].iloc[-1]
            readings["group"] = readings["group"].where(~after_end)
            totals = counter_totals(
                readings, value_column_counter, ["group"], handle_resets=True
            ).set_index("group")
            for group in batch_groups:
                if group in totals.index:
                    quantities.append(int(totals.at[group, "quantity"]))
                    resets.append(int(totals.at[group, "resets"]))
                else:
                    quantities.append(0)
                    resets.append(0)
        else:
            for _, batch_row in batches.iterrows():
                mask = (counter_data[self.time_column] >= batch_row["start"]) & (
                    counter_data[self.time_column] <= batch_row["end"]
                )
                batch_counter = counter_data.loc[mask, value_column_counter]
                quantities.append(
                    int(counter_quantity(batch_counter, handle_resets=False))
                )
                resets.append(0)

        batches = batches.copy()
        batches["quantity"] = quantities
        if handle_resets:
            batches["resets"] = resets
        return batches[base_columns].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def batch_transition_matrix(
        self, *, layout: Literal["long", "wide"] = "long"
    ) -> pd.DataFrame:
        """Count which batch follows which.

        Args:
            layout: ``"long"`` (default) returns one row per observed
                transition. ``"wide"`` returns the pivoted matrix.

        Returns:
            With ``layout="long"``, a DataFrame with columns ``from_state``,
            ``to_state`` and ``transition_count`` (int), one row per observed
            ``(from_state, to_state)`` pair, sorted by ``from_state`` and
            ``to_state``. With fewer than two batches the result is empty and
            keeps these columns.

            With ``layout="wide"``, a pivot table where index = from_batch,
            columns = to_batch, values = transition count, plus a ``total``
            column. The batch columns depend on the data, so with fewer than
            two batches the result is an empty frame with only the ``total``
            column (index named ``from_batch``).

        Raises:
            ValueError: If ``layout`` is not ``"long"`` or ``"wide"``.
        """
        if layout not in ("long", "wide"):
            raise ValueError(f"layout must be 'long' or 'wide', got {layout!r}")
        batches = self.detect_batches()

        if batches.empty or len(batches) < 2:
            if layout == "long":
                return pd.DataFrame(
                    {
                        "from_state": pd.Series(dtype=object),
                        "to_state": pd.Series(dtype=object),
                        "transition_count": pd.Series(dtype="int64"),
                    }
                )
            return pd.DataFrame(
                columns=["total"], index=pd.Index([], name="from_batch")
            )

        from_batch = batches["batch_id"].iloc[:-1].values
        to_batch = batches["batch_id"].iloc[1:].values

        transitions = pd.DataFrame({"from_batch": from_batch, "to_batch": to_batch})
        if layout == "long":
            return (
                transitions.groupby(["from_batch", "to_batch"])
                .size()
                .astype("int64")
                .reset_index(name="transition_count")
                .rename(columns={"from_batch": "from_state", "to_batch": "to_state"})
            )
        matrix = (
            transitions.groupby(["from_batch", "to_batch"]).size().unstack(fill_value=0)
        )
        matrix["total"] = matrix.sum(axis=1)
        return matrix
