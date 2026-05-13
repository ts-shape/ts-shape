"""Overall Equipment Effectiveness (OEE) calculator.

OEE = Availability x Performance x Quality

Industry-standard metric for manufacturing productivity:
- Availability: actual run time / planned production time
- Performance: actual throughput / ideal throughput
- Quality: good parts / total parts
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class OEECalculator(Base):
    """Calculate Overall Equipment Effectiveness from timeseries signals.

    Combines availability (from run/idle state), performance (from part counters
    and ideal cycle time), and quality (from total/reject counters) into a single
    OEE metric per day.

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
        planned_time_hours: Optional[float] = None,
        *,
        value_column: str = "value_bool",
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

        Returns:
            DataFrame with columns:
            - date
            - run_seconds
            - planned_seconds
            - availability_pct
        """
        state_data = (
            self.dataframe[self.dataframe["uuid"] == run_state_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if state_data.empty:
            return pd.DataFrame(
                columns=["date", "run_seconds", "planned_seconds", "availability_pct"]
            )

        state_data[self.time_column] = pd.to_datetime(state_data[self.time_column])
        state_data["state"] = state_data[value_column].fillna(False).astype(bool)
        state_data["date"] = state_data[self.time_column].dt.date

        # Duration of each sample = gap to next sample
        state_data["duration"] = (
            state_data[self.time_column].shift(-1) - state_data[self.time_column]
        ).dt.total_seconds()

        # Drop last row per day (NaN duration at boundaries handled by groupby)
        state_data = state_data[state_data["duration"].notna()]

        if state_data.empty:
            return pd.DataFrame(
                columns=["date", "run_seconds", "planned_seconds", "availability_pct"]
            )

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

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def calculate_performance(
        self,
        counter_uuid: str,
        ideal_cycle_time: float,
        run_state_uuid: Optional[str] = None,
        *,
        value_column: str = "value_integer",
        run_value_column: str = "value_bool",
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

        Returns:
            DataFrame with columns:
            - date
            - actual_parts
            - ideal_parts
            - run_seconds
            - performance_pct
        """
        counter_data = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if counter_data.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "actual_parts",
                    "ideal_parts",
                    "run_seconds",
                    "performance_pct",
                ]
            )

        counter_data[self.time_column] = pd.to_datetime(counter_data[self.time_column])
        counter_data["date"] = counter_data[self.time_column].dt.date

        # Compute run_seconds per day
        run_per_day = {}
        if run_state_uuid is not None:
            avail_df = self.calculate_availability(
                run_state_uuid, value_column=run_value_column
            )
            for _, row in avail_df.iterrows():
                run_per_day[row["date"]] = row["run_seconds"]

        results = []
        for date, grp in counter_data.groupby("date"):
            grp = grp.sort_values(self.time_column)
            actual_parts = grp[value_column].iloc[-1] - grp[value_column].iloc[0]
            actual_parts = max(0, actual_parts)

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

            results.append(
                {
                    "date": date,
                    "actual_parts": int(actual_parts),
                    "ideal_parts": round(ideal_parts, 2),
                    "run_seconds": round(run_seconds, 2),
                    "performance_pct": round(min(performance, 100.0), 2),
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def calculate_quality(
        self,
        total_uuid: str,
        reject_uuid: str,
        *,
        value_column: str = "value_integer",
    ) -> pd.DataFrame:
        """Calculate quality percentage (good parts / total parts).

        Args:
            total_uuid: UUID of the total-parts counter.
            reject_uuid: UUID of the reject-parts counter.
            value_column: Column holding counter values.

        Returns:
            DataFrame with columns:
            - date
            - total_parts
            - reject_parts
            - good_parts
            - quality_pct
        """
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
            return pd.DataFrame(
                columns=[
                    "date",
                    "total_parts",
                    "reject_parts",
                    "good_parts",
                    "quality_pct",
                ]
            )

        total_data[self.time_column] = pd.to_datetime(total_data[self.time_column])
        total_data["date"] = total_data[self.time_column].dt.date

        reject_by_day = {}
        if not reject_data.empty:
            reject_data[self.time_column] = pd.to_datetime(
                reject_data[self.time_column]
            )
            reject_data["date"] = reject_data[self.time_column].dt.date
            for date, grp in reject_data.groupby("date"):
                grp = grp.sort_values(self.time_column)
                reject_by_day[date] = max(
                    0, grp[value_column].iloc[-1] - grp[value_column].iloc[0]
                )

        results = []
        for date, grp in total_data.groupby("date"):
            grp = grp.sort_values(self.time_column)
            total_parts = max(0, grp[value_column].iloc[-1] - grp[value_column].iloc[0])
            reject_parts = reject_by_day.get(date, 0)
            good_parts = max(0, total_parts - reject_parts)
            quality = (good_parts / total_parts * 100.0) if total_parts > 0 else 0.0

            results.append(
                {
                    "date": date,
                    "total_parts": int(total_parts),
                    "reject_parts": int(reject_parts),
                    "good_parts": int(good_parts),
                    "quality_pct": round(min(quality, 100.0), 2),
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Combined OEE
    # ------------------------------------------------------------------

    def calculate_oee(
        self,
        run_state_uuid: str,
        counter_uuid: str,
        ideal_cycle_time: float,
        total_uuid: Optional[str] = None,
        reject_uuid: Optional[str] = None,
        *,
        planned_time_hours: Optional[float] = None,
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

        Returns:
            DataFrame with columns:
            - date
            - availability_pct
            - performance_pct
            - quality_pct
            - oee_pct
        """
        avail_df = self.calculate_availability(
            run_state_uuid, planned_time_hours=planned_time_hours
        )
        perf_df = self.calculate_performance(
            counter_uuid, ideal_cycle_time, run_state_uuid=run_state_uuid
        )

        if total_uuid is not None and reject_uuid is not None:
            qual_df = self.calculate_quality(total_uuid, reject_uuid)
        else:
            qual_df = None

        if avail_df.empty and perf_df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "availability_pct",
                    "performance_pct",
                    "quality_pct",
                    "oee_pct",
                ]
            )

        # Merge on date
        merged = avail_df[["date", "availability_pct"]].copy()
        if not perf_df.empty:
            merged = merged.merge(
                perf_df[["date", "performance_pct"]], on="date", how="outer"
            )
        else:
            merged["performance_pct"] = 0.0

        if qual_df is not None and not qual_df.empty:
            merged = merged.merge(
                qual_df[["date", "quality_pct"]], on="date", how="outer"
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
            merged[
                [
                    "date",
                    "availability_pct",
                    "performance_pct",
                    "quality_pct",
                    "oee_pct",
                ]
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )
