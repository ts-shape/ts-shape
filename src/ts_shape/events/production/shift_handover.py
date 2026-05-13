"""Automated shift handover report generation.

Standardize shift-to-shift communication:
- Production summary
- Quality summary
- Downtime summary
- Issues to watch

Two modes of operation:
1. From raw signals: ``generate_report(counter_uuid, ok_counter_uuid, ...)``
2. From pre-computed DataFrames: ``from_shift_data(production_df, quality_df, downtime_df)``
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import Optional, Dict, List
from datetime import date as DateType

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ShiftHandoverReport(Base):
    """Generate automated shift handover reports.

    Combines production, quality, and downtime data into a single summary
    suitable for shift handover meetings.

    Merge keys: [date, shift] — the report output is keyed on these columns.

    Two usage patterns:

    **Pattern A — from raw signals (one-step):**

        report = ShiftHandoverReport(df)
        result = report.generate_report(
            counter_uuid='prod', ok_counter_uuid='ok',
            nok_counter_uuid='nok', state_uuid='state',
            targets={'shift_1': 450, 'shift_2': 450},
        )

    **Pattern B — pipeline (compose from upstream modules):**

        # Step 1: compute upstream DataFrames
        prod   = ShiftReporting(df).shift_production('counter')
        qual   = QualityTracking(df).nok_by_shift('ok', 'nok')
        downtime = DowntimeTracking(df).downtime_by_shift('state')

        # Step 2: assemble into handover report
        result = ShiftHandoverReport.from_shift_data(
            production_df=prod,
            quality_df=qual,
            downtime_df=downtime,
            targets={'shift_1': 450, 'shift_2': 450},
        )

    Both patterns return the same output schema.
    """

    MERGE_KEYS = ["date", "shift"]
    OUTPUT_COLUMNS = [
        "date",
        "shift",
        "production",
        "production_target",
        "production_achievement_pct",
        "ok_parts",
        "nok_parts",
        "quality_pct",
        "availability_pct",
        "downtime_minutes",
    ]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        shift_definitions: Optional[Dict[str, tuple[str, str]]] = None,
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.time_column = time_column
        self.shift_definitions = shift_definitions or {
            "shift_1": ("06:00", "14:00"),
            "shift_2": ("14:00", "22:00"),
            "shift_3": ("22:00", "06:00"),
        }

    def _assign_shift(self, timestamp: pd.Timestamp) -> str:
        time = timestamp.time()
        for shift_name, (start, end) in self.shift_definitions.items():
            start_time = pd.to_datetime(start).time()
            end_time = pd.to_datetime(end).time()
            if start_time < end_time:
                if start_time <= time < end_time:
                    return shift_name
            else:
                if time >= start_time or time < end_time:
                    return shift_name
        return "unknown"

    def _counter_by_shift(
        self,
        uuid: str,
        value_column: str,
    ) -> pd.DataFrame:
        """Get counter deltas grouped by date/shift."""
        data = (
            self.dataframe[self.dataframe["uuid"] == uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(columns=["date", "shift", "quantity"])

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = data[self.time_column].apply(self._assign_shift)
        data["date"] = data[self.time_column].dt.date

        results = []
        for (dt, shift), grp in data.groupby(["date", "shift"]):
            grp = grp.sort_values(self.time_column)
            qty = max(0, grp[value_column].iloc[-1] - grp[value_column].iloc[0])
            results.append({"date": dt, "shift": shift, "quantity": int(qty)})

        return pd.DataFrame(results)

    def _availability_by_shift(
        self,
        state_uuid: str,
        running_value: str,
        value_column: str,
    ) -> pd.DataFrame:
        """Compute availability grouped by date/shift."""
        data = (
            self.dataframe[self.dataframe["uuid"] == state_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if data.empty:
            return pd.DataFrame(
                columns=["date", "shift", "availability_pct", "downtime_minutes"]
            )

        data[self.time_column] = pd.to_datetime(data[self.time_column])
        data["shift"] = data[self.time_column].apply(self._assign_shift)
        data["date"] = data[self.time_column].dt.date
        data["is_running"] = data[value_column] == running_value
        data["duration_s"] = data[self.time_column].diff().shift(-1).dt.total_seconds()
        data = data[data["duration_s"].notna()]

        results = []
        for (dt, shift), grp in data.groupby(["date", "shift"]):
            up = grp.loc[grp["is_running"], "duration_s"].sum()
            down = grp.loc[~grp["is_running"], "duration_s"].sum()
            total = up + down
            avail = (up / total * 100) if total > 0 else 0.0
            results.append(
                {
                    "date": dt,
                    "shift": shift,
                    "availability_pct": round(avail, 1),
                    "downtime_minutes": round(down / 60, 1),
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Pipeline entry-point: build report from pre-computed DataFrames
    # ------------------------------------------------------------------

    @staticmethod
    def from_shift_data(
        production_df: pd.DataFrame,
        quality_df: Optional[pd.DataFrame] = None,
        downtime_df: Optional[pd.DataFrame] = None,
        *,
        targets: Optional[Dict[str, float]] = None,
        report_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build a handover report from pre-computed shift-level DataFrames.

        This is the **pipeline-friendly** entry-point.  Instead of reading raw
        signals, it accepts DataFrames that were already computed by upstream
        modules (ShiftReporting, QualityTracking, DowntimeTracking, etc.).

        Args:
            production_df: DataFrame with [date, shift, quantity] — e.g. from
                ``ShiftReporting.shift_production()``.
            quality_df: DataFrame with [date, shift, ok_parts, nok_parts, quality_pct]
                — e.g. from ``QualityTracking.nok_by_shift()``.  Optional.
            downtime_df: DataFrame with [date, shift, availability_pct, downtime_minutes]
                — e.g. from ``DowntimeTracking.downtime_by_shift()``.  Optional.
            targets: Per-shift production targets (dict).
            report_date: Specific date (YYYY-MM-DD).  If None, uses latest.

        Returns:
            DataFrame with columns:
            date, shift, production, production_target, production_achievement_pct,
            ok_parts, nok_parts, quality_pct, availability_pct, downtime_minutes
        """
        output_cols = ShiftHandoverReport.OUTPUT_COLUMNS

        if production_df.empty:
            return pd.DataFrame(columns=output_cols)

        prod = production_df.copy()

        # Normalize column name
        if "quantity" in prod.columns and "production" not in prod.columns:
            prod = prod.rename(columns={"quantity": "production"})

        # Filter to report date
        if report_date:
            target_date = pd.to_datetime(report_date).date()
        else:
            prod["date"] = (
                pd.to_datetime(prod["date"]).dt.date
                if not pd.api.types.is_object_dtype(prod["date"])
                else prod["date"]
            )
            target_date = prod["date"].max()

        result = prod[prod["date"] == target_date].copy()

        if result.empty:
            return pd.DataFrame(columns=output_cols)

        # Merge quality
        if quality_df is not None and not quality_df.empty:
            q = quality_df[quality_df["date"] == target_date].copy()
            q_cols = ["date", "shift"]
            if "ok_parts" in q.columns:
                q_cols.append("ok_parts")
            if "nok_parts" in q.columns:
                q_cols.append("nok_parts")
            if "quality_pct" in q.columns:
                q_cols.append("quality_pct")
            elif "first_pass_yield_pct" in q.columns:
                q = q.rename(columns={"first_pass_yield_pct": "quality_pct"})
                q_cols.append("quality_pct")
            result = result.merge(q[q_cols], on=["date", "shift"], how="left")

        for col in ["ok_parts", "nok_parts", "quality_pct"]:
            if col not in result.columns:
                result[col] = 0 if col != "quality_pct" else 0.0

        # Merge downtime
        if downtime_df is not None and not downtime_df.empty:
            d = downtime_df[downtime_df["date"] == target_date].copy()
            d_cols = ["date", "shift"]
            if "availability_pct" in d.columns:
                d_cols.append("availability_pct")
            if "downtime_minutes" in d.columns:
                d_cols.append("downtime_minutes")
            result = result.merge(d[d_cols], on=["date", "shift"], how="left")

        for col in ["availability_pct", "downtime_minutes"]:
            if col not in result.columns:
                result[col] = 0.0

        result = result.fillna(0)

        # Targets
        if targets:
            result["production_target"] = result["shift"].map(targets).fillna(0)
            result["production_achievement_pct"] = (
                (result["production"] / result["production_target"] * 100)
                .where(result["production_target"] > 0, 0)
                .round(1)
            )
        else:
            result["production_target"] = 0.0
            result["production_achievement_pct"] = 0.0

        # Ensure quality_pct is computed if we have ok/nok but no quality_pct from upstream
        if (result["quality_pct"] == 0).all() and (result["ok_parts"] > 0).any():
            total = result["ok_parts"] + result["nok_parts"]
            result["quality_pct"] = (
                (result["ok_parts"] / total * 100).where(total > 0, 0).round(1)
            )

        # Return with consistent column order
        for col in output_cols:
            if col not in result.columns:
                result[col] = 0.0

        return result[output_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Raw-signal entry-point
    # ------------------------------------------------------------------

    def generate_report(
        self,
        counter_uuid: str,
        ok_counter_uuid: str,
        nok_counter_uuid: str,
        state_uuid: str,
        *,
        targets: Optional[Dict[str, float]] = None,
        quality_target_pct: float = 98.0,
        availability_target_pct: float = 90.0,
        running_value: str = "Running",
        value_column_counter: str = "value_integer",
        value_column_state: str = "value_string",
        report_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate a shift handover report from raw timeseries signals.

        For pipeline usage with pre-computed DataFrames, use
        :meth:`from_shift_data` instead.

        Args:
            counter_uuid: UUID of production counter.
            ok_counter_uuid: UUID of good parts counter.
            nok_counter_uuid: UUID of defective parts counter.
            state_uuid: UUID of machine state signal.
            targets: Per-shift production targets.
            quality_target_pct: Quality target percentage.
            availability_target_pct: Availability target percentage.
            running_value: Value indicating machine is running.
            value_column_counter: Column for counter values.
            value_column_state: Column for state values.
            report_date: Specific date (YYYY-MM-DD). If None, uses latest date.

        Returns:
            DataFrame with columns:
            date, shift, production, production_target, production_achievement_pct,
            ok_parts, nok_parts, quality_pct, availability_pct, downtime_minutes
        """
        # Production
        prod = self._counter_by_shift(counter_uuid, value_column_counter)
        ok = self._counter_by_shift(ok_counter_uuid, value_column_counter)
        nok = self._counter_by_shift(nok_counter_uuid, value_column_counter)
        avail = self._availability_by_shift(
            state_uuid, running_value, value_column_state
        )

        if prod.empty:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS)

        # Filter to report date
        if report_date:
            target_date = pd.to_datetime(report_date).date()
        else:
            target_date = prod["date"].max()

        prod = prod[prod["date"] == target_date]

        # Merge all data
        result = prod.rename(columns={"quantity": "production"})

        if not ok.empty:
            ok_filt = ok[ok["date"] == target_date].rename(
                columns={"quantity": "ok_parts"}
            )
            result = result.merge(
                ok_filt[["date", "shift", "ok_parts"]], on=["date", "shift"], how="left"
            )
        else:
            result["ok_parts"] = 0

        if not nok.empty:
            nok_filt = nok[nok["date"] == target_date].rename(
                columns={"quantity": "nok_parts"}
            )
            result = result.merge(
                nok_filt[["date", "shift", "nok_parts"]],
                on=["date", "shift"],
                how="left",
            )
        else:
            result["nok_parts"] = 0

        if not avail.empty:
            avail_filt = avail[avail["date"] == target_date]
            result = result.merge(
                avail_filt[["date", "shift", "availability_pct", "downtime_minutes"]],
                on=["date", "shift"],
                how="left",
            )
        else:
            result["availability_pct"] = 0.0
            result["downtime_minutes"] = 0.0

        result = result.fillna(0)

        # Targets
        if targets:
            result["production_target"] = result["shift"].map(targets).fillna(0)
            result["production_achievement_pct"] = (
                (result["production"] / result["production_target"] * 100)
                .where(result["production_target"] > 0, 0)
                .round(1)
            )
        else:
            result["production_target"] = 0.0
            result["production_achievement_pct"] = 0.0

        # Quality
        total = result["ok_parts"] + result["nok_parts"]
        result["quality_pct"] = (
            (result["ok_parts"] / total * 100).where(total > 0, 0).round(1)
        )

        return result[self.OUTPUT_COLUMNS].reset_index(drop=True)

    def highlight_issues(
        self,
        counter_uuid: Optional[str] = None,
        ok_counter_uuid: Optional[str] = None,
        nok_counter_uuid: Optional[str] = None,
        state_uuid: Optional[str] = None,
        *,
        report_df: Optional[pd.DataFrame] = None,
        thresholds: Optional[Dict[str, float]] = None,
        targets: Optional[Dict[str, float]] = None,
        running_value: str = "Running",
        value_column_counter: str = "value_integer",
        value_column_state: str = "value_string",
        report_date: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Identify issues that need attention.

        Can be called in two ways:

        1. From raw signals (provide UUIDs):
           ``highlight_issues('prod', 'ok', 'nok', 'state', thresholds={...})``

        2. From a pre-built report DataFrame:
           ``highlight_issues(report_df=my_report, thresholds={...})``

        Args:
            counter_uuid: UUID of production counter (raw-signal mode).
            ok_counter_uuid: UUID of good parts counter (raw-signal mode).
            nok_counter_uuid: UUID of defective parts counter (raw-signal mode).
            state_uuid: UUID of machine state signal (raw-signal mode).
            report_df: Pre-computed report DataFrame (pipeline mode).
                If provided, UUID arguments are ignored.
            thresholds: Minimum acceptable values for each metric.
                Defaults to production_achievement_pct=95, quality_pct=98, availability_pct=90.
            targets: Per-shift production targets.
            running_value: Value indicating machine is running.
            value_column_counter: Column for counter values.
            value_column_state: Column for state values.
            report_date: Specific date (YYYY-MM-DD).

        Returns:
            List of dicts with keys: shift, metric, value, threshold, severity.
            severity is 'warning' (within 5% of threshold) or 'critical'.
        """
        thresholds = thresholds or {
            "production_achievement_pct": 95.0,
            "quality_pct": 98.0,
            "availability_pct": 90.0,
        }

        if report_df is not None:
            report = report_df
        elif counter_uuid and ok_counter_uuid and nok_counter_uuid and state_uuid:
            report = self.generate_report(
                counter_uuid,
                ok_counter_uuid,
                nok_counter_uuid,
                state_uuid,
                targets=targets,
                running_value=running_value,
                value_column_counter=value_column_counter,
                value_column_state=value_column_state,
                report_date=report_date,
            )
        else:
            return []

        if report.empty:
            return []

        issues = []
        for _, row in report.iterrows():
            for metric, threshold in thresholds.items():
                if metric not in row:
                    continue
                value = row[metric]
                if value < threshold:
                    severity = "warning" if value >= threshold * 0.95 else "critical"
                    issues.append(
                        {
                            "shift": row["shift"],
                            "metric": metric,
                            "value": round(float(value), 1),
                            "threshold": threshold,
                            "severity": severity,
                        }
                    )

        return issues
