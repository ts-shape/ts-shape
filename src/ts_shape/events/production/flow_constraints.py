import logging
from typing import Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)


class FlowConstraintEvents(Base):
    """Production: Flow Constraints

    Both detectors work on two boolean run signals only (no buffer level or
    part-presence signal), so they flag the run-state pattern, not its cause:

    - blocked_events: upstream running while downstream is stopped -- the
      upstream station keeps producing into a stopped consumer and its
      output buffer fills.
    - starved_events: downstream running while upstream is stopped -- the
      downstream station keeps consuming with no upstream supply and drains
      its input buffer. This is the onset of starvation, not the common
      "station idle for lack of input" state: once the downstream station
      actually runs dry and stops, both are stopped and the interval is no
      longer reported.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        event_uuid: str = "prod:flow",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.event_uuid = event_uuid

    def _align_bool(self, uuid: str) -> pd.DataFrame:
        s = (
            self.dataframe[self.dataframe["uuid"] == uuid]
            .copy()
            .sort_values(self.time_column)
        )
        s[self.time_column] = pd.to_datetime(s[self.time_column])
        s["state"] = s["value_bool"].fillna(False).astype(bool)
        return s[[self.time_column, "state"]]

    def blocked_events(
        self,
        *,
        roles: dict[str, str],
        tolerance: str = "200ms",
        tolerance_before: str | None = None,
        tolerance_after: str | None = None,
        min_duration: str = "0s",
    ) -> pd.DataFrame:
        """Blocked: upstream_run=True while downstream_run=False.

        Flags intervals where the upstream station runs while its downstream
        consumer is stopped, so the buffer between them fills.

        Args:
            roles: Dictionary mapping role names to UUIDs.
                   Expected keys: 'upstream_run', 'downstream_run'
            tolerance: Default tolerance for time alignment (used if directional tolerances not provided)
            tolerance_before: Tolerance for looking backward in time during alignment
            tolerance_after: Tolerance for looking forward in time during alignment
            min_duration: Minimum duration for an event to be included

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta, type,
            time_alignment_quality, duration, severity

        Example:
            roles = {'upstream_run': 'uuid1', 'downstream_run': 'uuid2'}
        """
        up = self._align_bool(roles["upstream_run"])  # time, state
        dn = self._align_bool(roles["downstream_run"])  # time, state
        if up.empty or dn.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "type",
                    "time_alignment_quality",
                    "duration",
                    "severity",
                ]
            )

        # Use directional tolerances if provided, otherwise use single tolerance
        tol_before = (
            pd.to_timedelta(tolerance_before)
            if tolerance_before
            else pd.to_timedelta(tolerance)
        )
        tol_after = (
            pd.to_timedelta(tolerance_after)
            if tolerance_after
            else pd.to_timedelta(tolerance)
        )
        max_tol = max(tol_before, tol_after)

        # Merge with maximum tolerance and track time differences
        dn = dn.assign(**{self.time_column + "_dn": dn[self.time_column]})
        merged = pd.merge_asof(
            up,
            dn,
            on=self.time_column,
            suffixes=("_up", "_dn"),
            tolerance=max_tol,
            direction="nearest",
        )

        # Store original upstream time for quality calculation
        merged["time_up"] = up[self.time_column].array

        # Apply directional tolerance filtering if asymmetric tolerances are specified
        if tolerance_before or tolerance_after:
            time_diff = merged[self.time_column + "_dn"] - merged[self.time_column]
            # Keep only records within directional tolerance bounds
            valid_mask = (
                (time_diff <= pd.Timedelta(0)) & (time_diff.abs() <= tol_before)
            ) | ((time_diff >= pd.Timedelta(0)) & (time_diff <= tol_after))
            merged["state_dn"] = merged["state_dn"].astype("boolean").where(valid_mask)

        # Calculate alignment quality (percentage of records with matches)
        alignment_quality = (
            merged["state_dn"].notna().sum() / len(merged) if len(merged) > 0 else 0.0
        )

        cond = merged["state_up"] & (~merged["state_dn"].fillna(False))
        gid = (cond.ne(cond.shift())).cumsum()
        min_td = pd.to_timedelta(min_duration)
        rows: list[dict[str, Any]] = []
        for _, seg in merged.groupby(gid):
            m = cond.loc[seg.index]
            if not m.any():
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            duration = end - start
            if duration < min_td:
                continue

            # Calculate severity based on duration
            severity = self._calculate_severity(duration)

            rows.append(
                {
                    "start": start,
                    "end": end,
                    "uuid": self.event_uuid,
                    "source_uuid": roles["upstream_run"],
                    "is_delta": True,
                    "type": "blocked",
                    "time_alignment_quality": alignment_quality,
                    "duration": duration,
                    "severity": severity,
                }
            )
        return pd.DataFrame(rows)

    def starved_events(
        self,
        *,
        roles: dict[str, str],
        tolerance: str = "200ms",
        tolerance_before: str | None = None,
        tolerance_after: str | None = None,
        min_duration: str = "0s",
    ) -> pd.DataFrame:
        """Starved: downstream_run=True while upstream_run=False.

        Flags intervals where the downstream station runs while its upstream
        supplier is stopped, i.e. downstream is consuming its input buffer
        with no resupply. A downstream station that has already stopped for
        lack of input (both stations stopped) is not reported; with run
        signals alone that state cannot be told apart from any other joint
        stop.

        Args:
            roles: Dictionary mapping role names to UUIDs.
                   Expected keys: 'upstream_run', 'downstream_run'
            tolerance: Default tolerance for time alignment (used if directional tolerances not provided)
            tolerance_before: Tolerance for looking backward in time during alignment
            tolerance_after: Tolerance for looking forward in time during alignment
            min_duration: Minimum duration for an event to be included

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta, type,
            time_alignment_quality, duration, severity

        Example:
            roles = {'upstream_run': 'uuid1', 'downstream_run': 'uuid2'}
        """
        up = self._align_bool(roles["upstream_run"])  # time, state
        dn = self._align_bool(roles["downstream_run"])  # time, state
        if up.empty or dn.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "type",
                    "time_alignment_quality",
                    "duration",
                    "severity",
                ]
            )

        # Use directional tolerances if provided, otherwise use single tolerance
        tol_before = (
            pd.to_timedelta(tolerance_before)
            if tolerance_before
            else pd.to_timedelta(tolerance)
        )
        tol_after = (
            pd.to_timedelta(tolerance_after)
            if tolerance_after
            else pd.to_timedelta(tolerance)
        )
        max_tol = max(tol_before, tol_after)

        # Merge with maximum tolerance and track time differences
        up = up.assign(**{self.time_column + "_up": up[self.time_column]})
        merged = pd.merge_asof(
            dn,
            up,
            on=self.time_column,
            suffixes=("_dn", "_up"),
            tolerance=max_tol,
            direction="nearest",
        )

        # Store original downstream time for quality calculation
        merged["time_dn"] = dn[self.time_column].array

        # Apply directional tolerance filtering if asymmetric tolerances are specified
        if tolerance_before or tolerance_after:
            time_diff = merged[self.time_column + "_up"] - merged[self.time_column]
            # Keep only records within directional tolerance bounds
            valid_mask = (
                (time_diff <= pd.Timedelta(0)) & (time_diff.abs() <= tol_before)
            ) | ((time_diff >= pd.Timedelta(0)) & (time_diff <= tol_after))
            merged["state_up"] = merged["state_up"].astype("boolean").where(valid_mask)

        # Calculate alignment quality (percentage of records with matches)
        alignment_quality = (
            merged["state_up"].notna().sum() / len(merged) if len(merged) > 0 else 0.0
        )

        cond = merged["state_dn"] & (~merged["state_up"].fillna(False))
        gid = (cond.ne(cond.shift())).cumsum()
        min_td = pd.to_timedelta(min_duration)
        rows: list[dict[str, Any]] = []
        for _, seg in merged.groupby(gid):
            m = cond.loc[seg.index]
            if not m.any():
                continue
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            duration = end - start
            if duration < min_td:
                continue

            # Calculate severity based on duration
            severity = self._calculate_severity(duration)

            rows.append(
                {
                    "start": start,
                    "end": end,
                    "uuid": self.event_uuid,
                    "source_uuid": roles["downstream_run"],
                    "is_delta": True,
                    "type": "starved",
                    "time_alignment_quality": alignment_quality,
                    "duration": duration,
                    "severity": severity,
                }
            )
        return pd.DataFrame(rows)

    def _calculate_severity(
        self,
        duration: pd.Timedelta,
        warning_threshold: str = "5s",
        critical_threshold: str = "30s",
    ) -> str:
        """Calculate severity level based on duration thresholds.

        Args:
            duration: Event duration as a Timedelta
            warning_threshold: Duration below which an event is 'minor'.
            critical_threshold: Duration below which an event is 'moderate'
                (at or above it, 'severe').

        Returns:
            Severity classification: 'minor', 'moderate', or 'severe'
        """
        minor_td = pd.to_timedelta(warning_threshold)
        moderate_td = pd.to_timedelta(critical_threshold)

        if duration < minor_td:
            return "minor"
        elif duration < moderate_td:
            return "moderate"
        else:
            return "severe"

    @renamed_kwargs(
        minor_threshold="warning_threshold", moderate_threshold="critical_threshold"
    )
    def flow_constraint_analytics(
        self,
        *,
        roles: dict[str, str],
        tolerance: str = "200ms",
        tolerance_before: str | None = None,
        tolerance_after: str | None = None,
        min_duration: str = "0s",
        warning_threshold: str = "5s",
        critical_threshold: str = "30s",
    ) -> dict[str, Any]:
        """Generate comprehensive analytics for flow constraints (blockages and starvations).

        Args:
            roles: Dictionary mapping role names to UUIDs.
                   Expected keys: 'upstream_run', 'downstream_run'
            tolerance: Default tolerance for time alignment (used if directional tolerances not provided)
            tolerance_before: Tolerance for looking backward in time during alignment
            tolerance_after: Tolerance for looking forward in time during alignment
            min_duration: Minimum duration for an event to be included
            warning_threshold: Duration below which an event is 'minor'.
            critical_threshold: Duration below which an event is 'moderate'
                (at or above it, 'severe').

        Returns:
            Dictionary containing analytics for both blocked and starved events:
            - blocked_events: DataFrame of blocked events
            - starved_events: DataFrame of starved events
            - summary: Dictionary with statistics including:
                - blocked_count: Total number of blocked events
                - starved_count: Total number of starved events
                - blocked_total_duration: Total duration of blocked events
                - starved_total_duration: Total duration of starved events
                - blocked_avg_duration: Average duration of blocked events
                - starved_avg_duration: Average duration of starved events
                - blocked_severity_breakdown: Count by severity level
                - starved_severity_breakdown: Count by severity level
                - overall_alignment_quality: Average alignment quality across both types

        Examples:
            >>> roles = {'upstream_run': 'uuid1', 'downstream_run': 'uuid2'}
            >>> analytics = flow.flow_constraint_analytics(roles=roles)
            >>> print(analytics['summary']['blocked_count'])
        """
        # Get blocked and starved events
        blocked_df = self.blocked_events(
            roles=roles,
            tolerance=tolerance,
            tolerance_before=tolerance_before,
            tolerance_after=tolerance_after,
            min_duration=min_duration,
        )

        starved_df = self.starved_events(
            roles=roles,
            tolerance=tolerance,
            tolerance_before=tolerance_before,
            tolerance_after=tolerance_after,
            min_duration=min_duration,
        )

        # Calculate summary statistics
        summary: dict[str, Any] = {}

        # Blocked events statistics
        if not blocked_df.empty:
            summary["blocked_count"] = len(blocked_df)
            summary["blocked_total_duration"] = blocked_df["duration"].sum()
            summary["blocked_avg_duration"] = blocked_df["duration"].mean()
            summary["blocked_severity_breakdown"] = (
                blocked_df["severity"].value_counts().to_dict()
            )
            blocked_alignment_quality = (
                blocked_df["time_alignment_quality"].iloc[0]
                if len(blocked_df) > 0
                else 0.0
            )
        else:
            summary["blocked_count"] = 0
            summary["blocked_total_duration"] = pd.Timedelta(0)
            summary["blocked_avg_duration"] = pd.Timedelta(0)
            summary["blocked_severity_breakdown"] = {
                "minor": 0,
                "moderate": 0,
                "severe": 0,
            }
            blocked_alignment_quality = 0.0

        # Starved events statistics
        if not starved_df.empty:
            summary["starved_count"] = len(starved_df)
            summary["starved_total_duration"] = starved_df["duration"].sum()
            summary["starved_avg_duration"] = starved_df["duration"].mean()
            summary["starved_severity_breakdown"] = (
                starved_df["severity"].value_counts().to_dict()
            )
            starved_alignment_quality = (
                starved_df["time_alignment_quality"].iloc[0]
                if len(starved_df) > 0
                else 0.0
            )
        else:
            summary["starved_count"] = 0
            summary["starved_total_duration"] = pd.Timedelta(0)
            summary["starved_avg_duration"] = pd.Timedelta(0)
            summary["starved_severity_breakdown"] = {
                "minor": 0,
                "moderate": 0,
                "severe": 0,
            }
            starved_alignment_quality = 0.0

        # Overall alignment quality (average of both)
        quality_values = [blocked_alignment_quality, starved_alignment_quality]
        summary["overall_alignment_quality"] = (
            sum(quality_values) / len(quality_values) if quality_values else 0.0
        )

        # Total events
        summary["total_constraint_events"] = (
            summary["blocked_count"] + summary["starved_count"]
        )

        return {
            "blocked_events": blocked_df,
            "starved_events": starved_df,
            "summary": summary,
        }
