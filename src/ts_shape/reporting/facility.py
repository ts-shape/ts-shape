"""Facility master data: the registry every cross-site report is keyed on.

A global report is only as trustworthy as the facility master data behind it.
Two plants that call their night shift ``22:00 -> 06:00`` and ``21:45 -> 05:45``
are not comparable by calendar date; a plant whose timezone is missing silently
lands in UTC and reports a day that is eight hours out.

:class:`Facility` captures the per-site attributes a consolidated report needs
-- timezone, region, currency, shift pattern, working days -- and
:class:`FacilityRegistry` holds them, validates them, and hands them to the
reporting layer (:mod:`ts_shape.reporting.calendar`,
:mod:`ts_shape.reporting.rollup`).

Example::

    from ts_shape.reporting import Facility, FacilityRegistry

    registry = FacilityRegistry([
        Facility("DE-STR", name="Stuttgart", timezone="Europe/Berlin",
                 region="EMEA", country="DE", currency="EUR"),
        Facility("US-DTW", name="Detroit", timezone="America/Detroit",
                 region="AMER", country="US", currency="USD"),
        Facility("CN-SHA", name="Shanghai", timezone="Asia/Shanghai",
                 region="APAC", country="CN", currency="CNY"),
    ])

    registry.validate()      # -> DataFrame of master-data issues (empty is good)
    registry.to_frame()      # -> one row per facility
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from ts_shape.utils.shifts import DEFAULT_SHIFTS

__all__ = ["Facility", "FacilityRegistry"]

#: Column names used by :meth:`FacilityRegistry.to_frame` and accepted by
#: :meth:`FacilityRegistry.from_frame`.
FACILITY_COLUMN = "facility"


@dataclass(frozen=True)
class Facility:
    """Master data for one production site.

    Args:
        id: Stable site identifier used as the join key in every KPI frame
            (e.g. ``"DE-STR"``). Required.
        name: Human-readable site name for report headers.
        timezone: IANA timezone name (e.g. ``"Europe/Berlin"``). Drives local
            production days, shifts and period alignment.
        region: Reporting region (e.g. ``"EMEA"``) -- the level a fleet rollup
            aggregates to before the global total.
        country: ISO country code, for regulatory and tax-style groupings.
        currency: ISO 4217 code the site's monetary KPIs are denominated in.
            :class:`~ts_shape.reporting.normalize.MetricNormalizer` converts to
            a single reporting currency.
        shifts: Mapping of shift name to ``(start, end)`` ``"HH:MM"`` strings.
            Defaults to :data:`ts_shape.utils.shifts.DEFAULT_SHIFTS`.
        working_days: Weekday numbers the site normally runs, Monday = 0. Used
            by :class:`~ts_shape.reporting.coverage.ReportingCoverage` so a
            silent Sunday is not reported as a missing day.
        capacity_per_day: Optional nameplate capacity (parts/day). When set it
            is carried into rollups as a weight for capacity-normalised KPIs.
    """

    id: str
    name: str = ""
    timezone: str = "UTC"
    region: str = ""
    country: str = ""
    currency: str = "EUR"
    shifts: Mapping[str, tuple[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_SHIFTS), hash=False
    )
    working_days: tuple[int, ...] = (0, 1, 2, 3, 4)
    capacity_per_day: float | None = None

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("Facility.id must be a non-empty string")

    @property
    def label(self) -> str:
        """``name`` when set, otherwise the id -- what a report header shows."""
        return self.name or self.id

    def tzinfo(self) -> ZoneInfo:
        """Return the site's :class:`zoneinfo.ZoneInfo`.

        Raises:
            ValueError: If ``timezone`` is not a known IANA zone. The message
                names the facility so a 200-site registry is debuggable.
        """
        try:
            return ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Facility {self.id!r} has unknown timezone {self.timezone!r}"
            ) from exc

    def utc_offset_hours(self, at: pd.Timestamp | None = None) -> float:
        """UTC offset in hours at ``at`` (default: now), DST-aware."""
        moment = pd.Timestamp.utcnow() if at is None else pd.Timestamp(at)
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")
        offset = moment.tz_convert(self.tzinfo()).utcoffset()
        return 0.0 if offset is None else offset.total_seconds() / 3600.0

    def as_dict(self) -> dict[str, object]:
        """Flat mapping for :meth:`FacilityRegistry.to_frame`."""
        return {
            FACILITY_COLUMN: self.id,
            "name": self.name,
            "timezone": self.timezone,
            "region": self.region,
            "country": self.country,
            "currency": self.currency,
            "shift_count": len(self.shifts),
            "working_days": len(self.working_days),
            "capacity_per_day": self.capacity_per_day,
        }


def _to_time(value: str) -> _dt.time:
    parsed: _dt.time = pd.to_datetime(value).time()
    return parsed


def _minutes(t: _dt.time) -> int:
    return t.hour * 60 + t.minute


def _shift_issues(facility: Facility) -> list[tuple[str, str]]:
    """Return ``(issue, detail)`` pairs for a site's shift pattern.

    Checks the two failure modes that silently corrupt a shift report: shifts
    that overlap (a reading counted twice) and shifts that leave a hole (a
    reading counted nowhere, landing in ``"unknown"``).
    """
    issues: list[tuple[str, str]] = []
    if not facility.shifts:
        return [("no_shifts", "facility defines no shift pattern")]

    spans: list[tuple[int, int]] = []
    for name, bounds in facility.shifts.items():
        try:
            start, end = _to_time(bounds[0]), _to_time(bounds[1])
        except (ValueError, TypeError, IndexError):
            issues.append(("bad_shift_bounds", f"{name}={bounds!r}"))
            continue
        s, e = _minutes(start), _minutes(end)
        if s == e:
            issues.append(
                ("degenerate_shift", f"{name} starts and ends at {bounds[0]}")
            )
            continue
        # An overnight shift is modelled as two same-day spans.
        if s < e:
            spans.append((s, e))
        else:
            spans.extend([(s, 24 * 60), (0, e)])

    spans.sort()
    covered = 0
    prev_end = 0
    for span_start, span_end in spans:
        if span_start < prev_end:
            issues.append(
                (
                    "overlapping_shifts",
                    f"{span_start // 60:02d}:{span_start % 60:02d} re-covered",
                )
            )
        covered += max(0, span_end - max(span_start, prev_end))
        prev_end = max(prev_end, span_end)
    if covered < 24 * 60:
        issues.append(
            ("shift_gap", f"{24 * 60 - covered} minute(s) of the day fall in no shift")
        )
    return issues


class FacilityRegistry:
    """A validated collection of :class:`Facility` master records.

    The registry is the single source of truth the reporting layer reads
    timezones, regions and currencies from. It is immutable after
    construction; build a new one to change master data.

    Args:
        facilities: The sites to register.

    Raises:
        ValueError: If two facilities share an id.
    """

    def __init__(self, facilities: Iterable[Facility]) -> None:
        records: dict[str, Facility] = {}
        for facility in facilities:
            if facility.id in records:
                raise ValueError(f"Duplicate facility id {facility.id!r} in registry")
            records[facility.id] = facility
        self._facilities = records

    # -- construction ---------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        dataframe: pd.DataFrame,
        *,
        id_column: str = FACILITY_COLUMN,
        shifts: Mapping[str, Mapping[str, tuple[str, str]]] | None = None,
        default_shifts: Mapping[str, tuple[str, str]] | None = None,
    ) -> FacilityRegistry:
        """Build a registry from a master-data DataFrame.

        Args:
            dataframe: One row per site. Recognised optional columns:
                ``name``, ``timezone``, ``region``, ``country``, ``currency``,
                ``working_days`` (comma-separated or iterable of weekday
                numbers), ``capacity_per_day``. Unknown columns are ignored.
            id_column: Column holding the site identifier.
            shifts: Optional per-facility shift patterns, keyed by site id.
            default_shifts: Shift pattern for sites absent from ``shifts``.

        Returns:
            A :class:`FacilityRegistry`.

        Raises:
            ValueError: If ``id_column`` is missing or ids are duplicated.
        """
        if id_column not in dataframe.columns:
            raise ValueError(
                f"Column {id_column!r} not found. Available: {list(dataframe.columns)}"
            )
        fallback = dict(default_shifts) if default_shifts else dict(DEFAULT_SHIFTS)
        per_site = shifts or {}

        facilities: list[Facility] = []
        for row in dataframe.to_dict(orient="records"):
            site_id = str(row[id_column])
            facilities.append(
                Facility(
                    id=site_id,
                    name=str(row.get("name") or ""),
                    timezone=str(row.get("timezone") or "UTC"),
                    region=str(row.get("region") or ""),
                    country=str(row.get("country") or ""),
                    currency=str(row.get("currency") or "EUR"),
                    shifts=dict(per_site.get(site_id, fallback)),
                    working_days=_parse_working_days(row.get("working_days")),
                    capacity_per_day=_parse_optional_float(row.get("capacity_per_day")),
                )
            )
        return cls(facilities)

    # -- mapping-ish access ---------------------------------------------

    def __len__(self) -> int:
        return len(self._facilities)

    def __iter__(self) -> Iterator[Facility]:
        return iter(self._facilities.values())

    def __contains__(self, facility_id: object) -> bool:
        return facility_id in self._facilities

    def __getitem__(self, facility_id: str) -> Facility:
        return self.get(facility_id)

    def __repr__(self) -> str:
        return f"FacilityRegistry(facilities={len(self._facilities)})"

    def get(self, facility_id: str) -> Facility:
        """Return the facility with ``facility_id``.

        Raises:
            KeyError: If the id is not registered. The message lists the known
                ids, which is the usual cause -- a KPI frame keyed on a site
                code the master data spells differently.
        """
        try:
            return self._facilities[facility_id]
        except KeyError:
            raise KeyError(
                f"Unknown facility {facility_id!r}. Registered: {self.ids}"
            ) from None

    @property
    def ids(self) -> list[str]:
        """Registered site ids, sorted."""
        return sorted(self._facilities)

    @property
    def timezones(self) -> dict[str, str]:
        """Mapping of site id to IANA timezone name."""
        return {fid: f.timezone for fid, f in self._facilities.items()}

    @property
    def currencies(self) -> dict[str, str]:
        """Mapping of site id to ISO currency code."""
        return {fid: f.currency for fid, f in self._facilities.items()}

    def regions(self) -> dict[str, list[str]]:
        """Mapping of region name to the sorted site ids it contains."""
        grouped: dict[str, list[str]] = {}
        for facility in self._facilities.values():
            grouped.setdefault(facility.region or "unassigned", []).append(facility.id)
        return {region: sorted(ids) for region, ids in sorted(grouped.items())}

    def filter(
        self, *, region: str | None = None, country: str | None = None
    ) -> FacilityRegistry:
        """Return a new registry narrowed to a region and/or country."""
        selected = [
            f
            for f in self._facilities.values()
            if (region is None or f.region == region)
            and (country is None or f.country == country)
        ]
        return FacilityRegistry(selected)

    # -- frames ----------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Return the registry as a DataFrame, one row per site.

        Columns: ``facility``, ``name``, ``timezone``, ``region``, ``country``,
        ``currency``, ``shift_count``, ``working_days``, ``capacity_per_day``.
        """
        columns = [
            FACILITY_COLUMN,
            "name",
            "timezone",
            "region",
            "country",
            "currency",
            "shift_count",
            "working_days",
            "capacity_per_day",
        ]
        if not self._facilities:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame([f.as_dict() for f in self._facilities.values()])
        return frame[columns].sort_values(FACILITY_COLUMN).reset_index(drop=True)

    def attach(
        self,
        dataframe: pd.DataFrame,
        *,
        facility_column: str = FACILITY_COLUMN,
        columns: Iterable[str] = ("region", "country", "currency", "timezone"),
    ) -> pd.DataFrame:
        """Left-join master-data columns onto a KPI frame.

        Args:
            dataframe: KPI frame carrying a facility key.
            facility_column: Column in ``dataframe`` holding the site id.
            columns: Master-data columns to attach.

        Returns:
            A copy of ``dataframe`` with the requested columns added. Sites
            absent from the registry keep NaN, so an unmapped site is visible
            in the report rather than silently dropped.

        Raises:
            ValueError: If ``facility_column`` is missing.
        """
        if facility_column not in dataframe.columns:
            raise ValueError(
                f"Column {facility_column!r} not found. "
                f"Available: {list(dataframe.columns)}"
            )
        wanted = [c for c in columns]
        master = self.to_frame()
        missing = [c for c in wanted if c not in master.columns]
        if missing:
            raise ValueError(
                f"Unknown master-data column(s) {missing}. "
                f"Available: {[c for c in master.columns if c != FACILITY_COLUMN]}"
            )
        master = master[[FACILITY_COLUMN, *wanted]]
        result = dataframe.copy()
        return result.merge(
            master,
            how="left",
            left_on=facility_column,
            right_on=FACILITY_COLUMN,
            suffixes=("", "_master"),
        ).drop(
            columns=[FACILITY_COLUMN] if facility_column != FACILITY_COLUMN else [],
            errors="ignore",
        )

    def validate(self) -> pd.DataFrame:
        """Return master-data problems, one row per issue.

        Checks performed per facility:

        * ``unknown_timezone`` -- the IANA zone does not resolve.
        * ``missing_region`` / ``missing_currency`` -- fields a consolidated
          report groups or converts on.
        * ``suspicious_currency`` -- not a 3-letter ISO 4217-shaped code.
        * ``no_shifts`` / ``bad_shift_bounds`` / ``degenerate_shift`` /
          ``overlapping_shifts`` / ``shift_gap`` -- shift patterns that would
          double-count or drop readings.
        * ``no_working_days`` -- a site that never expects to report.

        Returns:
            DataFrame with columns ``facility``, ``issue``, ``detail``. Empty
            (with those columns) when the registry is clean, so callers can
            assert ``registry.validate().empty`` in a data-quality gate.
        """
        rows: list[dict[str, str]] = []
        for facility in self._facilities.values():
            try:
                facility.tzinfo()
            except ValueError:
                rows.append(
                    {
                        FACILITY_COLUMN: facility.id,
                        "issue": "unknown_timezone",
                        "detail": facility.timezone,
                    }
                )
            if not facility.region:
                rows.append(
                    {
                        FACILITY_COLUMN: facility.id,
                        "issue": "missing_region",
                        "detail": "region is empty; rollups group it as 'unassigned'",
                    }
                )
            if not facility.currency:
                rows.append(
                    {
                        FACILITY_COLUMN: facility.id,
                        "issue": "missing_currency",
                        "detail": "currency is empty; cost KPIs cannot be converted",
                    }
                )
            elif not (len(facility.currency) == 3 and facility.currency.isalpha()):
                rows.append(
                    {
                        FACILITY_COLUMN: facility.id,
                        "issue": "suspicious_currency",
                        "detail": f"{facility.currency!r} is not a 3-letter code",
                    }
                )
            if not facility.working_days:
                rows.append(
                    {
                        FACILITY_COLUMN: facility.id,
                        "issue": "no_working_days",
                        "detail": "site expects to report on no weekday",
                    }
                )
            for issue, detail in _shift_issues(facility):
                rows.append(
                    {FACILITY_COLUMN: facility.id, "issue": issue, "detail": detail}
                )

        columns = [FACILITY_COLUMN, "issue", "detail"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values([FACILITY_COLUMN, "issue"])
            .reset_index(drop=True)
        )


def _parse_working_days(value: object) -> tuple[int, ...]:
    """Coerce a master-data cell into weekday numbers (Monday = 0)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return (0, 1, 2, 3, 4)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(int(p) for p in parts) if parts else (0, 1, 2, 3, 4)
    if isinstance(value, Iterable):
        return tuple(int(v) for v in value)
    return (int(value),)  # type: ignore[call-overload]


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result
