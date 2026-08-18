"""Did every facility actually report? The precondition for any global figure.

A consolidated number computed over 34 of 37 plants looks exactly like one
computed over all 37 -- it is simply lower, and nothing in the frame says why.
Silent gaps are the most common defect in cross-site reporting, and the hardest
to spot after the fact.

:class:`ReportingCoverage` compares what arrived against what was expected --
ideally the calendar from
:meth:`~ts_shape.reporting.calendar.FacilityCalendar.period_grid`, which knows
each site's local periods and working days -- and reports completeness,
explicit missing rows, per-column null rates, and staleness. Run it before the
rollup, and quote its ``completeness_pct`` next to the headline number.

Example::

    from ts_shape.reporting import FacilityCalendar, ReportingCoverage

    expected = FacilityCalendar(registry).period_grid("2026-01-01", "2026-01-31")
    coverage = ReportingCoverage(kpi_df, value_columns=["parts", "oee_pct"])

    coverage.completeness(expected)   # per facility
    coverage.missing(expected)        # the actual gaps, one row each
    coverage.freshness(as_of="2026-02-01", max_age_days=2)
    coverage.summary(expected)        # one row to put beside the headline
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from ts_shape.reporting.facility import FACILITY_COLUMN

__all__ = ["ReportingCoverage"]

logger = logging.getLogger(__name__)


class ReportingCoverage:
    """Completeness, gaps and freshness of a multi-facility KPI frame.

    Args:
        dataframe: Long KPI frame, one row per facility and period.
        facility_column: Column holding the site id.
        period_column: Column holding the reporting period.
        value_columns: Metric columns whose null rate is tracked. Defaults to
            every numeric column that is not a key.

    Raises:
        ValueError: If a key column is missing.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str = FACILITY_COLUMN,
        period_column: str = "period",
        value_columns: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(dataframe).__name__}"
            )
        for column in (facility_column, period_column):
            if column not in dataframe.columns:
                raise ValueError(
                    f"Column {column!r} not found. "
                    f"Available: {list(dataframe.columns)}"
                )
        self.dataframe = dataframe.copy()
        self.facility_column = facility_column
        self.period_column = period_column
        if value_columns is None:
            self.value_columns = [
                column
                for column in dataframe.columns
                if column not in (facility_column, period_column)
                and pd.api.types.is_numeric_dtype(dataframe[column])
            ]
        else:
            missing = [c for c in value_columns if c not in dataframe.columns]
            if missing:
                raise ValueError(
                    f"value_columns {missing} not found. "
                    f"Available: {list(dataframe.columns)}"
                )
            self.value_columns = list(value_columns)

    def __repr__(self) -> str:
        return (
            f"ReportingCoverage(rows={len(self.dataframe)}, "
            f"facilities={self.dataframe[self.facility_column].nunique()}, "
            f"periods={self.dataframe[self.period_column].nunique()})"
        )

    # -- expected calendar ------------------------------------------------

    def _expected(self, expected: pd.DataFrame | None) -> pd.DataFrame:
        """Normalise the expected calendar to ``[facility, period]`` rows.

        When ``expected`` is ``None`` the calendar is the cross product of the
        facilities and periods actually observed. That detects a site that
        skipped a period, but *not* one that never reported at all -- pass a
        real grid (see
        :meth:`~ts_shape.reporting.calendar.FacilityCalendar.period_grid`) to
        catch those.
        """
        if expected is not None:
            missing = [
                c
                for c in (self.facility_column, self.period_column)
                if c not in expected.columns
            ]
            if missing:
                raise ValueError(
                    f"Expected calendar is missing column(s) {missing}. "
                    f"Available: {list(expected.columns)}"
                )
            return (
                expected[[self.facility_column, self.period_column]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        facilities = self.dataframe[self.facility_column].dropna().unique()
        periods = self.dataframe[self.period_column].dropna().unique()
        logger.debug(
            "No expected calendar supplied; assuming %d facilities x %d periods",
            len(facilities),
            len(periods),
        )
        index = pd.MultiIndex.from_product(
            [facilities, periods], names=[self.facility_column, self.period_column]
        )
        return index.to_frame(index=False)

    # -- checks ------------------------------------------------------------

    def missing(self, expected: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the facility/period combinations that never arrived.

        Args:
            expected: Expected calendar with facility and period columns.
                Defaults to the observed cross product.

        Returns:
            One row per gap, with the facility and period columns. Empty (with
            those columns) when coverage is complete.
        """
        expected_grid = self._expected(expected)
        actual = (
            self.dataframe[[self.facility_column, self.period_column]]
            .drop_duplicates()
            .assign(_present=True)
        )
        merged = expected_grid.merge(
            actual, on=[self.facility_column, self.period_column], how="left"
        )
        gaps = merged[merged["_present"].isna()].drop(columns=["_present"])
        return gaps.sort_values([self.facility_column, self.period_column]).reset_index(
            drop=True
        )

    def completeness(self, expected: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-facility reporting completeness.

        Returns:
            One row per expected facility with ``periods_expected``,
            ``periods_reported``, ``periods_missing``, ``completeness_pct``,
            ``null_pct`` (share of NaN cells across ``value_columns`` in the
            rows that did arrive) and ``status``: ``"complete"``,
            ``"partial"``, or ``"silent"`` for a site that reported nothing at
            all. Sites that reported but were not expected appear with
            ``status="unexpected"`` rather than being dropped.
        """
        expected_grid = self._expected(expected)
        expected_counts = (
            expected_grid.groupby(self.facility_column)[self.period_column]
            .nunique()
            .rename("periods_expected")
        )
        reported_counts = (
            self.dataframe.groupby(self.facility_column)[self.period_column]
            .nunique()
            .rename("periods_reported")
        )
        frame = (
            pd.concat([expected_counts, reported_counts], axis=1)
            .fillna(0)
            .reset_index()
        )
        frame["periods_expected"] = frame["periods_expected"].astype(int)
        frame["periods_reported"] = frame["periods_reported"].astype(int)
        frame["periods_missing"] = (
            (frame["periods_expected"] - frame["periods_reported"]).clip(lower=0)
        ).astype(int)
        frame["completeness_pct"] = [
            (
                float("nan")
                if expected_n == 0
                else min(reported, expected_n) / expected_n * 100.0
            )
            for expected_n, reported in zip(
                frame["periods_expected"], frame["periods_reported"]
            )
        ]
        frame["null_pct"] = frame[self.facility_column].map(self._null_share())
        frame["status"] = [
            _coverage_status(expected_n, reported)
            for expected_n, reported in zip(
                frame["periods_expected"], frame["periods_reported"]
            )
        ]
        return frame.sort_values(
            ["completeness_pct", self.facility_column], na_position="first"
        ).reset_index(drop=True)

    def _null_share(self) -> dict[object, float]:
        """Share of NaN cells across the tracked value columns, per facility."""
        if not self.value_columns:
            return {}
        grouped = self.dataframe.groupby(self.facility_column)[self.value_columns]
        shares = grouped.apply(
            lambda group: (
                float(group.isna().to_numpy().mean() * 100.0)
                if group.size
                else float("nan")
            )
        )
        return {key: float(value) for key, value in shares.items()}

    def null_report(self) -> pd.DataFrame:
        """Null rate per facility and metric column.

        Returns:
            Long frame with ``facility``, ``metric``, ``rows``, ``nulls`` and
            ``null_pct``. A column that is always NaN for one site -- a signal
            that never got mapped there -- shows as 100 %.
        """
        columns = [self.facility_column, "metric", "rows", "nulls", "null_pct"]
        if not self.value_columns:
            return pd.DataFrame(columns=columns)
        rows: list[dict[str, object]] = []
        for site, group in self.dataframe.groupby(self.facility_column, sort=True):
            for metric in self.value_columns:
                nulls = int(group[metric].isna().sum())
                total = int(len(group))
                rows.append(
                    {
                        self.facility_column: site,
                        "metric": metric,
                        "rows": total,
                        "nulls": nulls,
                        "null_pct": (nulls / total * 100.0) if total else float("nan"),
                    }
                )
        return pd.DataFrame(rows, columns=columns)

    def freshness(
        self,
        *,
        as_of: str | pd.Timestamp | None = None,
        max_age_days: float | None = None,
    ) -> pd.DataFrame:
        """How recently each facility reported.

        Args:
            as_of: Reference moment. Defaults to the latest period in the frame,
                which measures each site against the fleet's leading edge.
            max_age_days: Staleness threshold. When given, an ``is_stale``
                column flags sites whose latest period is older.

        Returns:
            One row per facility with ``last_period``, ``age_days`` and, when
            ``max_age_days`` is given, ``is_stale``. Oldest first. Requires a
            datetime-like period column; a non-datetime period yields NaN ages
            and a warning.
        """
        periods = pd.to_datetime(self.dataframe[self.period_column], errors="coerce")
        if periods.isna().all():
            logger.warning(
                "Period column %r is not datetime-like; freshness ages are NaN.",
                self.period_column,
            )
        work = self.dataframe.assign(_period=periods)
        last = (
            work.groupby(self.facility_column)["_period"]
            .max()
            .rename("last_period")
            .reset_index()
        )
        reference = (
            pd.to_datetime(as_of) if as_of is not None else last["last_period"].max()
        )
        last["age_days"] = (
            reference - last["last_period"]
        ).dt.total_seconds() / 86400.0
        if max_age_days is not None:
            last["is_stale"] = last["age_days"] > float(max_age_days)
        return last.sort_values(
            "age_days", ascending=False, na_position="first"
        ).reset_index(drop=True)

    def summary(self, expected: pd.DataFrame | None = None) -> pd.DataFrame:
        """One-row fleet coverage summary to publish beside the headline number.

        Returns:
            DataFrame with ``facilities_expected``, ``facilities_reporting``,
            ``facilities_silent``, ``periods_expected``, ``rows_expected``,
            ``rows_reported``, ``rows_missing``, ``completeness_pct`` and
            ``null_pct``.
        """
        expected_grid = self._expected(expected)
        gaps = self.missing(expected)
        actual_rows = (
            self.dataframe[[self.facility_column, self.period_column]]
            .drop_duplicates()
            .merge(expected_grid, on=[self.facility_column, self.period_column])
        )
        expected_facilities = int(expected_grid[self.facility_column].nunique())
        reporting = int(
            actual_rows[self.facility_column].nunique() if not actual_rows.empty else 0
        )
        rows_expected = int(len(expected_grid))
        null_shares = list(self._null_share().values())
        row = {
            "facilities_expected": expected_facilities,
            "facilities_reporting": reporting,
            "facilities_silent": max(0, expected_facilities - reporting),
            "periods_expected": int(expected_grid[self.period_column].nunique()),
            "rows_expected": rows_expected,
            "rows_reported": int(len(actual_rows)),
            "rows_missing": int(len(gaps)),
            "completeness_pct": (
                len(actual_rows) / rows_expected * 100.0
                if rows_expected
                else float("nan")
            ),
            "null_pct": (
                float(sum(null_shares) / len(null_shares))
                if null_shares
                else float("nan")
            ),
        }
        return pd.DataFrame([row])


def _coverage_status(expected: int, reported: int) -> str:
    if expected == 0:
        return "unexpected"
    if reported == 0:
        return "silent"
    if reported >= expected:
        return "complete"
    return "partial"
