"""Overall Equipment Effectiveness (OEE) calculator.

OEE = Availability x Performance x Quality

Industry-standard metric for manufacturing productivity:
- Availability: actual run time / planned production time
- Performance: actual throughput / ideal throughput
- Quality: good parts / total parts
"""

import logging

import pandas as pd  # type: ignore

from ts_shape.events._output import (
    COL_DURATION_S,
    COL_END,
    COL_START,
    empty_event_df,
)
from ts_shape.utils.base import Base
from ts_shape.utils.counters import count_resets, counter_quantity

logger = logging.getLogger(__name__)


_DAY_SECONDS = 86400.0


def _daily_summary(rows: list[dict], extra_cols: list[str]) -> pd.DataFrame:
    """Convert a list of {date, ...} rows into the canonical summary schema.

    Each ``date`` becomes ``start`` (midnight) + ``end`` (next midnight) +
    ``duration_seconds = 86400``. The ``date`` key is dropped.
    """
    if not rows:
        return empty_event_df("summary", extra_cols=extra_cols)
    df = pd.DataFrame(rows)
    df[COL_START] = pd.to_datetime(df["date"])
    df[COL_END] = df[COL_START] + pd.Timedelta(days=1)
    df[COL_DURATION_S] = _DAY_SECONDS
    df = df.drop(columns=["date"])
    leading = [COL_START, COL_END, COL_DURATION_S]
    return df[leading + [c for c in df.columns if c not in leading]]


class OEECalculator(Base):
    """Calculate Overall Equipment Effectiveness from timeseries signals.

    Combines availability (from run/idle state), performance (from part counters
    and ideal cycle time), and quality (from total/reject counters) into a single
    OEE metric per day. Each daily row uses the canonical summary schema:
    ``start`` (midnight) / ``end`` (next midnight) / ``duration_seconds`` = 86400.

    Example usage:
        oee = OEECalculator(df)

        # Individual components
        avail = oee.calculate_availability('machine_state')
        perf = oee.calculate_performance('part_counter', ideal_cycle_time=30.0, run_state_uuid='machine_state')
        qual = oee.calculate_quality('total_counter', 'reject_counter')

        # Combined daily OEE
        daily = oee.calculate_oee(
            run_state_uuid='machine_state',
            counter_uuid='part_counter',
            ideal_cycle_time=30.0,
            total_uuid='total_counter',
            reject_uuid='reject_counter',
        )
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
    ) -> None:
        """Initialize OEE calculator.

        Args:
            dataframe: Input DataFrame with timeseries data.
            time_column: Name of timestamp column (default: 'systime').
        """
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def calculate_availability(
        self,
        run_state_uuid: str,
        planned_time_hours: float | None = None,
        *,
        value_column: str = "value_bool",
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate availability percentage from run/idle intervals.

        Availability = run_time / planned_time.  When *planned_time_hours* is
        ``None`` the planned time is derived from the first-to-last timestamp
        span for each day.

        Args:
            run_state_uuid: UUID of the boolean run-state signal (True = running).
            planned_time_hours: Fixed planned production hours per day.  If None,
                the time span covered by data each day is used.
            value_column: Column holding the boolean state.
            max_gap: Optional cap (e.g. "15min"). A sample's duration is the gap
                to the next reading; a gap longer than this is treated as a
                logging outage ("no data") and excluded from both run time and
                planned time. Without it, an overnight or logger gap while the
                last state was ``running`` is booked as run time, inflating
                availability. Only meaningful when ``planned_time_hours`` is
                None (the planned time is derived from the data).

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds (canonical day boundaries; 86400)
            - run_seconds
            - planned_seconds
            - availability_pct
        """
        avail_extra = ["run_seconds", "planned_seconds", "availability_pct"]
        state_data = (
            self.dataframe[self.dataframe["uuid"] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return _daily_summary([], avail_extra)

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        state_data["state"] = state_data[value_column].fillna(False).astype(bool)
        state_data["date"] = state_data[self.time_column].dt.date

        # Duration of each sample = gap to next sample
        state_data["duration"] = (
            state_data[self.time_column].shift(-1) - state_data[self.time_column]
        ).dt.total_seconds()

        # Treat over-long gaps as "no data" so they inflate neither run nor
        # planned time.
        if max_gap is not None:
            cap_seconds = pd.to_timedelta(max_gap).total_seconds()
            state_data.loc[state_data["duration"] > cap_seconds, "duration"] = float(
                "nan"
            )

        # Drop last row per day (NaN duration at boundaries handled by groupby)
        state_data = state_data[state_data["duration"].notna()]

        if state_data.empty:
            return _daily_summary([], avail_extra)

        results = []
        for date, grp in state_data.groupby("date"):
            run_seconds = grp.loc[grp["state"], "duration"].sum()
            if planned_time_hours is not None:
                planned_seconds = planned_time_hours * 3600.0
            else:
                planned_seconds = grp["duration"].sum()

            availability = (
                (run_seconds / planned_seconds * 100.0) if planned_seconds > 0 else 0.0
            )

            results.append(
                {
                    "date": date,
                    "run_seconds": round(run_seconds, 2),
                    "planned_seconds": round(planned_seconds, 2),
                    "availability_pct": round(min(availability, 100.0), 2),
                }
            )

        return _daily_summary(results, avail_extra)

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def calculate_performance(
        self,
        counter_uuid: str,
        ideal_cycle_time: float,
        run_state_uuid: str | None = None,
        *,
        value_column: str = "value_integer",
        run_value_column: str = "value_bool",
        handle_resets: bool = False,
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate performance percentage (actual vs ideal throughput).

        Performance = (actual_parts * ideal_cycle_time) / run_time.
        If *run_state_uuid* is None, the total time span per day is used
        as run time.

        Args:
            counter_uuid: UUID of the monotonic part counter.
            ideal_cycle_time: Ideal cycle time in seconds per part.
            run_state_uuid: Optional UUID for run-state to compute actual run time.
            value_column: Column holding counter values.
            run_value_column: Column holding boolean run state.
            handle_resets: When True, treat a decreasing counter as a reset and
                count the production after the reset instead of losing it, so
                ``actual_parts`` is the reset-aware total. The result gains a
                ``resets`` column with the number of resets observed each day.
                When False (default), ``actual_parts`` is simply
                ``max(0, last - first)`` and a day containing a reset reports
                too little.
            max_gap: Optional cap forwarded to :meth:`calculate_availability`
                when ``run_state_uuid`` is given, so a logging gap does not
                inflate the run time used as the performance denominator.

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds
            - actual_parts
            - ideal_parts
            - run_seconds
            - performance_pct
            - resets (only when ``handle_resets=True``)
        """
        perf_extra = ["actual_parts", "ideal_parts", "run_seconds", "performance_pct"]
        if handle_resets:
            perf_extra.append("resets")
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if counter_data.empty:
            return _daily_summary([], perf_extra)

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])
        counter_data["date"] = counter_data[self.time_column].dt.date

        # Compute run_seconds per day
        run_per_day = {}
        if run_state_uuid is not None:
            avail_df = self.calculate_availability(
                run_state_uuid, value_column=run_value_column, max_gap=max_gap
            )
            for _, row in avail_df.iterrows():
                run_per_day[row[COL_START].date()] = row["run_seconds"]

        results = []
        for date, grp in counter_data.groupby("date"):
            grp = grp.sort_values(self.time_column)
            if handle_resets:
                actual_parts = counter_quantity(grp[value_column], handle_resets=True)
            else:
                actual_parts = max(
                    0, grp[value_column].iloc[-1] - grp[value_column].iloc[0]
                )

            if date in run_per_day:
                run_seconds = run_per_day[date]
            else:
                span = (
                    grp[self.time_column].iloc[-1] - grp[self.time_column].iloc[0]
                ).total_seconds()
                run_seconds = max(span, 0)

            if run_seconds > 0 and ideal_cycle_time > 0:
                ideal_parts = run_seconds / ideal_cycle_time
                performance = (actual_parts / ideal_parts) * 100.0
            else:
                ideal_parts = 0.0
                performance = 0.0

            row = {
                "date": date,
                "actual_parts": int(actual_parts),
                "ideal_parts": round(ideal_parts, 2),
                "run_seconds": round(run_seconds, 2),
                "performance_pct": round(min(performance, 100.0), 2),
            }
            if handle_resets:
                row["resets"] = count_resets(grp[value_column])
            results.append(row)

        return _daily_summary(results, perf_extra)

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def calculate_quality(
        self,
        total_uuid: str,
        reject_uuid: str,
        *,
        value_column: str = "value_integer",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Calculate quality percentage (good parts / total parts).

        Args:
            total_uuid: UUID of the total-parts counter.
            reject_uuid: UUID of the reject-parts counter.
            value_column: Column holding counter values.
            handle_resets: When True, treat a decreasing counter as a reset and
                count the production after the reset for both the total and the
                reject counters, so a day containing a reset is not
                under-reported. The result gains a ``resets`` column with the
                combined number of resets (total + reject) observed each day.
                When False (default), both quantities are ``max(0, last - first)``.

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds
            - total_parts
            - reject_parts
            - good_parts
            - quality_pct
            - resets (only when ``handle_resets=True``)
        """
        qual_extra = ["total_parts", "reject_parts", "good_parts", "quality_pct"]
        if handle_resets:
            qual_extra.append("resets")
        total_data = (
            self.dataframe[self.dataframe["uuid"] == total_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        reject_data = (
            self.dataframe[self.dataframe["uuid"] == reject_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if total_data.empty:
            return _daily_summary([], qual_extra)

        total_data[self.time_column] = pd.to_datetime(total_data[self.time_column])
        total_data["date"] = total_data[self.time_column].dt.date

        reject_by_day = {}
        reject_resets_by_day = {}
        if not reject_data.empty:
            reject_data[self.time_column] = pd.to_datetime(
                reject_data[self.time_column]
            )
            reject_data["date"] = reject_data[self.time_column].dt.date
            for date, grp in reject_data.groupby("date"):
                grp = grp.sort_values(self.time_column)
                if handle_resets:
                    reject_by_day[date] = counter_quantity(
                        grp[value_column], handle_resets=True
                    )
                    reject_resets_by_day[date] = count_resets(grp[value_column])
                else:
                    reject_by_day[date] = max(
                        0, grp[value_column].iloc[-1] - grp[value_column].iloc[0]
                    )

        results = []
        for date, grp in total_data.groupby("date"):
            grp = grp.sort_values(self.time_column)
            if handle_resets:
                total_parts = counter_quantity(grp[value_column], handle_resets=True)
            else:
                total_parts = max(
                    0, grp[value_column].iloc[-1] - grp[value_column].iloc[0]
                )
            reject_parts = reject_by_day.get(date, 0)
            good_parts = max(0, total_parts - reject_parts)
            quality = (good_parts / total_parts * 100.0) if total_parts > 0 else 0.0

            row = {
                "date": date,
                "total_parts": int(total_parts),
                "reject_parts": int(reject_parts),
                "good_parts": int(good_parts),
                "quality_pct": round(min(quality, 100.0), 2),
            }
            if handle_resets:
                row["resets"] = count_resets(grp[value_column]) + int(
                    reject_resets_by_day.get(date, 0)
                )
            results.append(row)

        return _daily_summary(results, qual_extra)

    # ------------------------------------------------------------------
    # Combined OEE
    # ------------------------------------------------------------------

    def calculate_oee(
        self,
        run_state_uuid: str,
        counter_uuid: str,
        ideal_cycle_time: float,
        total_uuid: str | None = None,
        reject_uuid: str | None = None,
        *,
        planned_time_hours: float | None = None,
        handle_resets: bool = False,
        max_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate daily OEE = Availability * Performance * Quality.

        When *total_uuid* / *reject_uuid* are not provided, quality is
        assumed to be 100 %.

        Args:
            run_state_uuid: UUID of the boolean run-state signal.
            counter_uuid: UUID of the monotonic part counter.
            ideal_cycle_time: Ideal cycle time in seconds per part.
            total_uuid: Optional UUID of total-parts counter (for quality).
            reject_uuid: Optional UUID of reject-parts counter (for quality).
            planned_time_hours: Fixed planned production hours per day.
            handle_resets: When True, the part / total / reject counters are
                treated as reset-aware (see :meth:`calculate_performance` and
                :meth:`calculate_quality`), so a day where a counter resets does
                not silently drag OEE toward zero. Forwarded to the performance
                and quality components. Default False.
            max_gap: Optional cap (e.g. "15min") forwarded to the availability
                and performance components so a logging gap is not booked as run
                time (which would inflate availability and understate
                performance). Default None.

        Returns:
            Summary-shape DataFrame with columns:
            - start, end, duration_seconds
            - availability_pct
            - performance_pct
            - quality_pct
            - oee_pct
        """
        oee_extra = [
            "availability_pct",
            "performance_pct",
            "quality_pct",
            "oee_pct",
        ]
        avail_df = self.calculate_availability(
            run_state_uuid, planned_time_hours=planned_time_hours, max_gap=max_gap
        )
        perf_df = self.calculate_performance(
            counter_uuid,
            ideal_cycle_time,
            run_state_uuid=run_state_uuid,
            handle_resets=handle_resets,
            max_gap=max_gap,
        )

        if total_uuid is not None and reject_uuid is not None:
            qual_df = self.calculate_quality(
                total_uuid, reject_uuid, handle_resets=handle_resets
            )
        else:
            qual_df = None

        if avail_df.empty and perf_df.empty:
            return empty_event_df("summary", extra_cols=oee_extra)

        # Merge on canonical day boundaries (start/end/duration_seconds)
        key_cols = [COL_START, COL_END, COL_DURATION_S]
        merged = avail_df[key_cols + ["availability_pct"]].copy()
        if not perf_df.empty:
            merged = merged.merge(
                perf_df[key_cols + ["performance_pct"]], on=key_cols, how="outer"
            )
        else:
            merged["performance_pct"] = 0.0

        if qual_df is not None and not qual_df.empty:
            merged = merged.merge(
                qual_df[key_cols + ["quality_pct"]], on=key_cols, how="outer"
            )
        else:
            merged["quality_pct"] = 100.0

        merged = merged.fillna(0.0)

        merged["oee_pct"] = (
            merged["availability_pct"]
            * merged["performance_pct"]
            * merged["quality_pct"]
            / 10000.0  # three percentages multiplied: /100 /100
        ).round(2)

        return (
            merged[key_cols + oee_extra].sort_values(COL_START).reset_index(drop=True)
        )
