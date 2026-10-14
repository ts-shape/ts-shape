"""Production schedule adherence: planned quantity vs. what the line made.

OEE answers "how well did the line run?" but not "did it make what the plan
asked for?". Those come apart routinely: a line at 92 % OEE running the wrong
order is still behind schedule, and a line that lost two hours to a changeover
can be on plan if the plan accounted for it.

This detector joins a planned schedule -- period start/end, planned quantity,
order id -- against actual production derived from a monotonic counter, and
reports per period how far ahead or behind the line is, plus the rate it would
have to hold for the rest of the period to recover.

Actual quantities are reset-aware by default (see
:mod:`ts_shape.utils.counters`): a counter that drops to zero at a part change
inside a scheduled period would otherwise report the period as catastrophically
behind. Pass ``handle_resets=False`` for the naive ``last - first`` reading.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.counters import (
    count_resets,
    counter_increments,
    counter_quantity,
)

logger = logging.getLogger(__name__)


class ProductionScheduleAdherenceEvents(Base):
    """Production: schedule vs. actual adherence.

    The schedule is a plain DataFrame — one row per planned period::

        schedule = pd.DataFrame({
            "start":       pd.to_datetime(["2024-05-02 06:00", "2024-05-02 14:00"]),
            "end":         pd.to_datetime(["2024-05-02 14:00", "2024-05-02 22:00"]),
            "planned_qty": [400, 400],
            "order_id":    ["WO-1001", "WO-1002"],
        })

        adherence = ProductionScheduleAdherenceEvents(df, schedule)
        adherence.adherence_by_period("counter:line-1")

    Column names are configurable via the constructor, so an MES export can be
    passed through without renaming.

    Methods:
    - adherence_by_period: Planned vs. actual and recovery rate per period.
    - deviation_events: Point events for periods breaching the deviation bands.
    - adherence_summary: Roll-up across the scheduled horizon.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        schedule: pd.DataFrame,
        *,
        event_uuid: str = "prod:schedule_adherence",
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        schedule_start_column: str = "start",
        schedule_end_column: str = "end",
        planned_qty_column: str = "planned_qty",
        order_id_column: str = "order_id",
    ) -> None:
        """Initialize the adherence tracker.

        Args:
            dataframe: Timeseries frame holding the production counter.
            schedule: Planned schedule, one row per period.
            event_uuid: UUID stamped on emitted events.
            time_column: Timestamp column of ``dataframe``.
            uuid_column: Signal-identifier column of ``dataframe``.
            value_column: Counter-value column of ``dataframe``.
            schedule_start_column: Period-start column of ``schedule``.
            schedule_end_column: Period-end column of ``schedule``.
            planned_qty_column: Planned-quantity column of ``schedule``.
            order_id_column: Optional order-identifier column of ``schedule``.
                Missing is fine — the emitted ``order_id`` is then ``None``.

        Raises:
            TypeError: If ``schedule`` is not a DataFrame.
            ValueError: If ``schedule`` lacks a required column, or a period
                ends before it starts.
        """
        super().__init__(dataframe, time_column=time_column)
        if not isinstance(schedule, pd.DataFrame):
            raise TypeError(
                f"schedule must be a pandas DataFrame, got {type(schedule).__name__}"
            )

        self.event_uuid = event_uuid
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.value_column = value_column
        self.schedule_start_column = schedule_start_column
        self.schedule_end_column = schedule_end_column
        self.planned_qty_column = planned_qty_column
        self.order_id_column = order_id_column
        self.schedule = self._prepare_schedule(schedule)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _prepare_schedule(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the planned-schedule frame."""
        required = (
            self.schedule_start_column,
            self.schedule_end_column,
            self.planned_qty_column,
        )
        missing = [c for c in required if c not in schedule.columns]
        if missing:
            raise ValueError(
                f"schedule is missing required column(s) {missing}. "
                f"Got columns: {list(schedule.columns)}"
            )
        if schedule.empty:
            return schedule.copy()

        out = schedule.copy()
        out[self.schedule_start_column] = pd.to_datetime(
            out[self.schedule_start_column]
        )
        out[self.schedule_end_column] = pd.to_datetime(out[self.schedule_end_column])
        out[self.planned_qty_column] = to_numeric_checked(
            out[self.planned_qty_column],
            what=f"schedule column {self.planned_qty_column!r}",
        )
        inverted = out[self.schedule_end_column] < out[self.schedule_start_column]
        if inverted.any():
            raise ValueError(
                f"{int(inverted.sum())} schedule period(s) end before they start; "
                f"check {self.schedule_start_column!r}/{self.schedule_end_column!r}."
            )
        return out.sort_values(self.schedule_start_column).reset_index(drop=True)

    def _counter_frame(self, counter_uuid: str) -> pd.DataFrame:
        """Return the sorted ``(time, value)`` frame for the counter signal."""
        sel = self.dataframe[self.dataframe[self.uuid_column] == counter_uuid]
        if sel.empty:
            return pd.DataFrame(columns=[self.time_column, self.value_column])
        out = sel[[self.time_column, self.value_column]].copy()
        out[self.time_column] = pd.to_datetime(out[self.time_column])
        out[self.value_column] = to_numeric_checked(
            out[self.value_column],
            what=f"counter {counter_uuid!r} ({self.value_column!r})",
        )
        return out.dropna(subset=[self.value_column]).sort_values(self.time_column)

    @staticmethod
    def _classify(
        adherence_pct: float,
        *,
        ahead_pct: float,
        behind_pct: float,
        critical_pct: float,
    ) -> str:
        """Map an adherence percentage onto a deviation band."""
        if np.isnan(adherence_pct):
            return "on_track"
        if adherence_pct < critical_pct:
            return "critical_behind"
        if adherence_pct < behind_pct:
            return "behind"
        if adherence_pct > ahead_pct:
            return "ahead"
        return "on_track"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adherence_by_period(
        self,
        counter_uuid: str,
        *,
        handle_resets: bool = True,
        as_of: pd.Timestamp | str | None = None,
        ahead_pct: float = 105.0,
        behind_pct: float = 95.0,
        critical_pct: float = 80.0,
    ) -> pd.DataFrame:
        """Compare planned against actual production for every scheduled period.

        ``recovery_required_rate`` is the units per hour the line must hold
        over the time still left in the period to finish on plan, evaluated at
        ``as_of``. It is ``0`` for a period already at or above plan and
        ``NaN`` once the period is over — a finished period cannot be
        recovered, and reporting a rate for it would invite planning against a
        number that means nothing.

        Args:
            counter_uuid: Identifier of the monotonic production counter.
            handle_resets: Treat a decreasing counter as a reset and count the
                post-reset value as new production. Default ``True``. The
                increments are computed once over the whole counter signal; a
                reading reports the production since the previous reading, so
                a period ``[start, end]`` gets the increments of the readings
                in ``(start, end]``. The production between the last reading
                before ``start`` and the first reading in the period is
                counted, and a reading on a boundary shared by two periods is
                booked once, to the period ending there. When False,
                ``actual_qty`` is ``max(0, last - first)`` of the readings in
                ``[start, end]``.
            as_of: Evaluation time for the recovery rate. Defaults to the last
                timestamp present in the data.
            ahead_pct: Adherence above this is ``ahead``.
            behind_pct: Adherence below this is ``behind``.
            critical_pct: Adherence below this is ``critical_behind``.

        Returns:
            Summary-shape DataFrame: ``start``, ``end``, ``duration_seconds``,
            ``uuid``, ``source_uuid``, ``order_id``, ``planned_qty``,
            ``actual_qty``, ``schedule_adherence_pct``, ``deviation_units``,
            ``deviation_type``, ``recovery_required_rate``,
            ``remaining_seconds``, ``sample_count``, ``resets``.

        Raises:
            ValueError: If the band thresholds are not ordered
                ``critical_pct <= behind_pct <= ahead_pct``.
        """
        if not critical_pct <= behind_pct <= ahead_pct:
            raise ValueError(
                "thresholds must satisfy critical_pct <= behind_pct <= ahead_pct, "
                f"got {critical_pct}, {behind_pct}, {ahead_pct}"
            )
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "order_id",
            "planned_qty",
            "actual_qty",
            "schedule_adherence_pct",
            "deviation_units",
            "deviation_type",
            "recovery_required_rate",
            "remaining_seconds",
            "sample_count",
            "resets",
        ]
        counter = self._counter_frame(counter_uuid)
        if self.schedule.empty or counter.empty:
            return pd.DataFrame(columns=cols)

        now = (
            pd.to_datetime(as_of)
            if as_of is not None
            else counter[self.time_column].iloc[-1]
        )

        # Reset-aware increments over the whole signal: a reading reports the
        # production since the previous reading, so a period [start, end]
        # owns the increments of the readings in (start, end]. Production
        # between the last reading before ``start`` and the first one inside
        # the period is no longer lost, and a reading exactly on a shared
        # boundary is booked to the period that ends there.
        times = counter[self.time_column]
        increments = counter_increments(counter[self.value_column])
        is_reset = counter[self.value_column].diff() < 0

        rows: list[dict[str, Any]] = []
        for _, period in self.schedule.iterrows():
            start = period[self.schedule_start_column]
            end = period[self.schedule_end_column]
            in_window = (times >= start) & (times <= end)
            window = counter[in_window]
            values = window[self.value_column]
            if handle_resets:
                booked = in_window & (times > start)
                actual = float(increments[booked].sum())
                n_resets = int(is_reset[booked].sum())
            else:
                actual = counter_quantity(values, handle_resets=False)
                n_resets = count_resets(values)
            planned = float(period[self.planned_qty_column])

            adherence = actual / planned * 100.0 if planned > 0 else float("nan")
            deviation = actual - planned
            remaining = max(0.0, (end - now).total_seconds())
            if deviation >= 0:
                recovery_rate = 0.0
            elif remaining > 0:
                recovery_rate = -deviation / (remaining / 3600.0)
            else:
                recovery_rate = float("nan")

            rows.append(
                {
                    "start": start,
                    "end": end,
                    "duration_seconds": (end - start).total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": counter_uuid,
                    "order_id": period.get(self.order_id_column),
                    "planned_qty": planned,
                    "actual_qty": actual,
                    "schedule_adherence_pct": adherence,
                    "deviation_units": deviation,
                    "deviation_type": self._classify(
                        adherence,
                        ahead_pct=ahead_pct,
                        behind_pct=behind_pct,
                        critical_pct=critical_pct,
                    ),
                    "recovery_required_rate": (
                        recovery_rate if pd.notna(recovery_rate) else float("nan")
                    ),
                    "remaining_seconds": remaining,
                    "sample_count": int(len(window)),
                    "resets": n_resets,
                }
            )

        return pd.DataFrame(rows, columns=cols)

    def deviation_events(
        self,
        counter_uuid: str,
        *,
        handle_resets: bool = True,
        as_of: pd.Timestamp | str | None = None,
        ahead_pct: float = 105.0,
        behind_pct: float = 95.0,
        critical_pct: float = 80.0,
        include_ahead: bool = False,
    ) -> pd.DataFrame:
        """Emit one event per period that deviates from plan.

        Args:
            counter_uuid: Identifier of the monotonic production counter.
            handle_resets: See :meth:`adherence_by_period`.
            as_of: See :meth:`adherence_by_period`.
            ahead_pct: Adherence above this is ``ahead``.
            behind_pct: Adherence below this is ``behind``.
            critical_pct: Adherence below this is ``critical_behind``.
            include_ahead: Also emit rows for periods running ahead of plan.
                Off by default — running ahead is rarely an alert, but it is
                worth reporting when the plan feeds a downstream commitment.

        Returns:
            Point-shape DataFrame: ``systime``, ``uuid``, ``source_uuid``,
            ``order_id``, ``planned_qty``, ``actual_qty``,
            ``schedule_adherence_pct``, ``deviation_units``,
            ``deviation_type``, ``recovery_required_rate``, ``severity``,
            ``direction``, ``threshold``.
        """
        cols = [
            "systime",
            "uuid",
            "source_uuid",
            "order_id",
            "planned_qty",
            "actual_qty",
            "schedule_adherence_pct",
            "deviation_units",
            "deviation_type",
            "recovery_required_rate",
            "severity",
            "direction",
            "threshold",
        ]
        periods = self.adherence_by_period(
            counter_uuid,
            handle_resets=handle_resets,
            as_of=as_of,
            ahead_pct=ahead_pct,
            behind_pct=behind_pct,
            critical_pct=critical_pct,
        )
        if periods.empty:
            return pd.DataFrame(columns=cols)

        wanted = {"behind", "critical_behind"}
        if include_ahead:
            wanted.add("ahead")
        flagged = periods[periods["deviation_type"].isin(wanted)]
        if flagged.empty:
            return pd.DataFrame(columns=cols)

        severity = {
            "critical_behind": "critical",
            "behind": "warning",
            "ahead": "info",
        }
        threshold = {
            "critical_behind": critical_pct,
            "behind": behind_pct,
            "ahead": ahead_pct,
        }
        out = pd.DataFrame(
            {
                "systime": flagged["end"].to_numpy(),
                "uuid": self.event_uuid,
                "source_uuid": counter_uuid,
                "order_id": flagged["order_id"].to_numpy(),
                "planned_qty": flagged["planned_qty"].to_numpy(),
                "actual_qty": flagged["actual_qty"].to_numpy(),
                "schedule_adherence_pct": flagged["schedule_adherence_pct"].to_numpy(),
                "deviation_units": flagged["deviation_units"].to_numpy(),
                "deviation_type": flagged["deviation_type"].to_numpy(),
                "recovery_required_rate": flagged["recovery_required_rate"].to_numpy(),
                "severity": flagged["deviation_type"].map(severity).to_numpy(),
                "direction": np.where(
                    flagged["deviation_units"].to_numpy() < 0, "below", "above"
                ),
                "threshold": flagged["deviation_type"].map(threshold).to_numpy(),
            }
        )
        return out[cols].reset_index(drop=True)

    def adherence_summary(
        self,
        counter_uuid: str,
        *,
        handle_resets: bool = True,
        as_of: pd.Timestamp | str | None = None,
        ahead_pct: float = 105.0,
        behind_pct: float = 95.0,
        critical_pct: float = 80.0,
    ) -> pd.DataFrame:
        """Roll the per-period adherence up across the scheduled horizon.

        ``overall_adherence_pct`` is computed from the summed quantities, not
        as the mean of the per-period percentages: a period with a planned
        quantity of 10 should not weigh as much as one planning 1000.

        Args:
            counter_uuid: Identifier of the monotonic production counter.
            handle_resets: See :meth:`adherence_by_period`.
            as_of: See :meth:`adherence_by_period`.
            ahead_pct: Adherence above this is ``ahead``.
            behind_pct: Adherence below this is ``behind``.
            critical_pct: Adherence below this is ``critical_behind``.

        Returns:
            Summary-shape DataFrame with a single row: ``start``, ``end``,
            ``duration_seconds``, ``uuid``, ``source_uuid``,
            ``total_planned_qty``, ``total_actual_qty``,
            ``overall_adherence_pct``, ``deviation_units``, ``periods``,
            ``periods_on_track``, ``periods_behind``,
            ``periods_critical_behind``, ``periods_ahead``,
            ``on_plan_rate_pct``, ``worst_order_id``,
            ``worst_adherence_pct``, ``sample_count``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "total_planned_qty",
            "total_actual_qty",
            "overall_adherence_pct",
            "deviation_units",
            "periods",
            "periods_on_track",
            "periods_behind",
            "periods_critical_behind",
            "periods_ahead",
            "on_plan_rate_pct",
            "worst_order_id",
            "worst_adherence_pct",
            "sample_count",
        ]
        periods = self.adherence_by_period(
            counter_uuid,
            handle_resets=handle_resets,
            as_of=as_of,
            ahead_pct=ahead_pct,
            behind_pct=behind_pct,
            critical_pct=critical_pct,
        )
        if periods.empty:
            return pd.DataFrame(columns=cols)

        counts = periods["deviation_type"].value_counts()
        total_planned = float(periods["planned_qty"].sum())
        total_actual = float(periods["actual_qty"].sum())
        n_periods = int(len(periods))
        on_plan = int(counts.get("on_track", 0) + counts.get("ahead", 0))
        worst_idx = periods["schedule_adherence_pct"].idxmin()

        return pd.DataFrame(
            [
                {
                    "start": periods["start"].iloc[0],
                    "end": periods["end"].iloc[-1],
                    "duration_seconds": (
                        periods["end"].iloc[-1] - periods["start"].iloc[0]
                    ).total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": counter_uuid,
                    "total_planned_qty": total_planned,
                    "total_actual_qty": total_actual,
                    "overall_adherence_pct": (
                        (total_actual / total_planned * 100.0)
                        if total_planned > 0
                        else float("nan")
                    ),
                    "deviation_units": total_actual - total_planned,
                    "periods": n_periods,
                    "periods_on_track": int(counts.get("on_track", 0)),
                    "periods_behind": int(counts.get("behind", 0)),
                    "periods_critical_behind": int(counts.get("critical_behind", 0)),
                    "periods_ahead": int(counts.get("ahead", 0)),
                    "on_plan_rate_pct": on_plan / n_periods * 100.0,
                    "worst_order_id": periods.loc[worst_idx, "order_id"],
                    "worst_adherence_pct": periods.loc[
                        worst_idx, "schedule_adherence_pct"
                    ],
                    "sample_count": int(periods["sample_count"].sum()),
                }
            ],
            columns=cols,
        )
