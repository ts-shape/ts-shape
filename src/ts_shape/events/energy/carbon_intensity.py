import logging
import warnings

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._windows import window_bounds, with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_per_window
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs
from ts_shape.utils.time_accounting import (
    grouper_periods,
    reading_intervals,
    split_intervals,
)
from ts_shape.utils.validate import check_choice

logger = logging.getLogger(__name__)

#: The reading kinds ``signal_kind`` accepts (besides ``"auto"``).
SIGNAL_KINDS = ("interval", "meter", "power")
#: ``"auto"`` reads a continuous signal sampled faster than this as power.
POWER_MAX_PERIOD = pd.Timedelta(minutes=1)


class CarbonIntensityEvents(Base):
    """Energy: Carbon Intensity Tracking (Scope 1 & 2)

    Converts energy and fuel consumption signals to CO2-equivalent emissions
    using configurable emission factors. Supports Scope 1 (direct fuel) and
    Scope 2 (electricity) calculations, plus carbon intensity per unit produced.
    Designed for CSRD and ISO 14064 reporting.

    Supports two data models via constructor parameters:

    - Standard (ts-shape default)::

        CarbonIntensityEvents(df, emission_factors={"meter:elec": 0.233})
        # expects: systime | uuid | value_double

    - Raw CSV::

        CarbonIntensityEvents(df, emission_factors={"sensor_01": 0.233},
                              time_column="time", uuid_column="id")

    ``emission_factors`` is a dict mapping signal identifier → kgCO2e per unit
    of the energy/fuel reading::

        emission_factors = {
            "meter:electricity": 0.233,   # kgCO2e/kWh  (UK grid average 2026)
            "meter:gas":         2.034,   # kgCO2e/m³   (natural gas)
        }

    Each factor's scope is inferred from ``scope_map``. Any uuid not in
    ``scope_map`` defaults to Scope 2.

    Methods:
    - emissions_by_window: CO2e per source per time window.
    - total_emissions_by_window: Aggregated Scope 1 + 2 per window.
    - carbon_intensity_per_unit: kgCO2e per unit produced.
    - emission_factor_audit: Return configured factors for audit trail.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        emission_factors: dict[str, float],
        *,
        scope_map: dict[str, int] | None = None,
        event_uuid: str = "energy:carbon",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        """
        Args:
            dataframe: Input signal DataFrame.
            emission_factors: Mapping of signal identifier → kgCO2e per unit.
                Each factor must be a finite number >= 0; an identifier that
                is not in the data emits a ``DataQualityWarning``.
            scope_map: Optional mapping of signal identifier → scope (1 or 2).
                       Defaults to Scope 2 for any uuid not in the map.
            event_uuid: UUID assigned to output events.
            time_column: Name of the timestamp column.
            uuid_column: Name of the signal identifier column.
        """
        super().__init__(dataframe, time_column=time_column)
        where = "CarbonIntensityEvents"
        for src, factor in emission_factors.items():
            try:
                number = float(factor)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{where}: emission factor of {src!r} must be a number, "
                    f"got {factor!r}"
                ) from None
            if not np.isfinite(number) or number < 0:
                raise ValueError(
                    f"{where}: emission factor of {src!r} must be a finite "
                    f"number >= 0 (kgCO2e per unit), got {factor!r}"
                )
        for src, src_scope in (scope_map or {}).items():
            check_choice(src_scope, (1, 2), name=f"scope_map[{src!r}]", where=where)
        self.emission_factors = emission_factors
        self.scope_map = scope_map or {}
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    def emissions_by_window(
        self,
        *,
        scope: int = 0,
        value_column: str = "value_double",
        window: str = "1D",
        signal_kind: str = "auto",
    ) -> pd.DataFrame:
        """CO2e emissions per configured source per time window.

        Args:
            scope: Filter by scope: 1 = fuel only, 2 = electricity only,
                   0 = all sources (default).
            value_column: Column containing consumption readings.
            window: Aggregation window (e.g. '1D', '1h').
            signal_kind: What a reading is (see :data:`SIGNAL_KINDS`):
                ``"interval"`` -- the consumption since the previous reading
                (summed per window); ``"meter"`` -- a cumulative meter
                (reset-aware increments per window); ``"power"`` -- a rate
                such as kW (integrated over time: value x hours, so kW gives
                kWh); ``"auto"`` (default) -- inferred per source from its
                data (:func:`ts_shape.signals.infer.infer_spec`): a counter is
                a meter, a continuous signal sampled faster than once a
                minute is a power reading, anything else an interval
                reading. A choice other than ``"interval"`` made by
                ``"auto"`` is stated in a ``DataQualityWarning``.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid, scope,
                       consumption, emission_factor, kgco2e. ``end`` is
                       ``start`` plus ``window``.
        """
        check_choice(scope, (0, 1, 2), name="scope", where="CarbonIntensityEvents")
        check_choice(
            signal_kind,
            ("auto", *SIGNAL_KINDS),
            name="signal_kind",
            where="CarbonIntensityEvents",
        )
        frames = []
        for src_uuid, factor in self.emission_factors.items():
            src_scope = self.scope_map.get(src_uuid, 2)
            if scope != 0 and src_scope != scope:
                continue

            raw = (
                self.dataframe[self.dataframe[self._uuid_column] == src_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            if raw.empty:
                logger.debug("No data found for emission source '%s'", src_uuid)
                continue

            raw[self.time_column] = pd.to_datetime(raw[self.time_column])
            kind = self._kind(raw, src_uuid, value_column, signal_kind)
            agg = self._consumption(raw, value_column, window, kind)
            agg["uuid"] = self.event_uuid
            agg["source_uuid"] = src_uuid
            agg["scope"] = src_scope
            agg["emission_factor"] = factor
            agg["kgco2e"] = agg["consumption"] * factor
            frames.append(
                agg[
                    [
                        "start",
                        "uuid",
                        "source_uuid",
                        "scope",
                        "consumption",
                        "emission_factor",
                        "kgco2e",
                    ]
                ]
            )

        if not frames:
            return with_window_bounds(
                pd.DataFrame(
                    columns=[
                        "start",
                        "uuid",
                        "source_uuid",
                        "scope",
                        "consumption",
                        "emission_factor",
                        "kgco2e",
                    ]
                ),
                window,
            )

        return with_window_bounds(
            pd.concat(frames, ignore_index=True)
            .sort_values(["start", "source_uuid"])
            .reset_index(drop=True),
            window,
        )

    def _kind(
        self, raw: pd.DataFrame, uuid: str, value_column: str, signal_kind: str
    ) -> str:
        """The reading kind of one source (``signal_kind`` or inferred)."""
        if signal_kind != "auto":
            return signal_kind
        from ts_shape.signals.infer import infer_spec

        frame = raw[[self.time_column, value_column]].rename(
            columns={value_column: "value_double"}
        )
        frame["value_double"] = pd.to_numeric(frame["value_double"], errors="coerce")
        if frame["value_double"].notna().sum() < 3:
            return "interval"
        spec = infer_spec(frame, uuid, self.time_column)
        period = spec.nominal_period
        if spec.kind == "counter":
            kind, why = "meter", "a cumulative meter (it only counts up)"
        elif (
            spec.kind == "continuous"
            and not spec.is_change_encoded
            and period is not None
            and period < POWER_MAX_PERIOD
        ):
            kind = "power"
            why = (
                f"a power reading (sampled every {period.total_seconds():g} s); "
                "it is integrated over time (value x hours)"
            )
        else:
            return "interval"
        warnings.warn(
            f"CarbonIntensityEvents: source {uuid!r} looks like {why}, so it is "
            f"read as signal_kind={kind!r}. Pass signal_kind='interval' if each "
            "reading is the consumption since the previous one.",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
        return kind

    def _consumption(
        self, raw: pd.DataFrame, value_column: str, window: str, kind: str
    ) -> pd.DataFrame:
        """``start``, ``consumption`` per window for one source."""
        series = pd.to_numeric(
            raw.set_index(self.time_column)[value_column], errors="coerce"
        )
        if kind == "meter":
            per = counter_per_window(series, window, handle_resets=True)
        elif kind == "power":
            values = series.dropna()
            times = pd.Series(values.index)
            if len(values) < 2:
                per = series.resample(window).sum() * 0.0
            else:
                intervals = reading_intervals(times, pd.Series(range(len(values))))
                intervals["value"] = values.to_numpy()[
                    intervals["state"].to_numpy(dtype=int)
                ]
                periods = grouper_periods(times.iloc[0], times.iloc[-1], window)
                pieces = split_intervals(
                    intervals[["start", "end", "value"]], periods, keys=["period"]
                )
                energy = pieces["value"] * pieces["seconds"] / 3600.0
                per = energy.groupby(pieces["period"].to_numpy()).sum()
                per.index = pd.DatetimeIndex(per.index)
                per = per.reindex(series.resample(window).sum().index, fill_value=0.0)
        else:
            per = series.resample(window).sum()
        out = per.rename("consumption").to_frame().reset_index()
        return out.rename(columns={out.columns[0]: "start"})

    def total_emissions_by_window(
        self,
        *,
        value_column: str = "value_double",
        window: str = "1D",
        signal_kind: str = "auto",
    ) -> pd.DataFrame:
        """Aggregate Scope 1 + Scope 2 emissions across all configured meters.

        Args:
            value_column: Column containing consumption readings.
            window: Aggregation window.
            signal_kind: What a reading is; see :meth:`emissions_by_window`.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, scope1_kgco2e,
                       scope2_kgco2e, total_kgco2e
        """
        all_emissions = self.emissions_by_window(
            scope=0, value_column=value_column, window=window, signal_kind=signal_kind
        )
        empty_cols = [
            "start",
            "uuid",
            "scope1_kgco2e",
            "scope2_kgco2e",
            "total_kgco2e",
        ]
        if all_emissions.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        scope1 = (
            all_emissions[all_emissions["scope"] == 1]
            .groupby("start")["kgco2e"]
            .sum()
            .rename("scope1_kgco2e")
        )
        scope2 = (
            all_emissions[all_emissions["scope"] == 2]
            .groupby("start")["kgco2e"]
            .sum()
            .rename("scope2_kgco2e")
        )

        out = pd.concat([scope1, scope2], axis=1).fillna(0.0).reset_index()
        out["total_kgco2e"] = out["scope1_kgco2e"] + out["scope2_kgco2e"]
        out["uuid"] = self.event_uuid
        # ``start`` is already the true window start (emissions_by_window is
        # bounded); take each window's end from there instead of re-bounding.
        ends = all_emissions.groupby("start")["end"].first()
        out.insert(1, "end", out["start"].map(ends))
        out.insert(
            2, "duration_seconds", (out["end"] - out["start"]).dt.total_seconds()
        )
        return (
            out[["start", "end", "duration_seconds", *empty_cols[1:]]]
            .sort_values("start")
            .reset_index(drop=True)
        )

    @renamed_kwargs(counter_column="value_column_counter")
    def carbon_intensity_per_unit(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_double",
        value_column_counter: str = "value_integer",
        window: str = "1D",
        handle_resets: bool = True,
        signal_kind: str = "auto",
    ) -> pd.DataFrame:
        """Carbon intensity per unit produced (kgCO2e / unit).

        Args:
            counter_uuid: Identifier of the production counter signal.
            value_column: Column containing energy/fuel readings.
            value_column_counter: Column containing counter readings.
            window: Aggregation window.
            handle_resets: ``True`` (default) treats a decreasing production
                counter as a reset and sums reset-aware increments per window
                (see :func:`ts_shape.utils.counters.counter_per_window`), so a
                window with a reset still reports what was produced.
                ``False`` uses the per-window ``max - min``.
            signal_kind: What an energy reading is; see
                :meth:`emissions_by_window`.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, total_kgco2e,
                       units_produced, carbon_intensity, trend
        """
        totals = self.total_emissions_by_window(
            value_column=value_column, window=window, signal_kind=signal_kind
        )
        empty_cols = [
            "start",
            "uuid",
            "total_kgco2e",
            "units_produced",
            "carbon_intensity",
            "trend",
        ]
        if totals.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        counter_raw = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if counter_raw.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        counter_raw[self.time_column] = pd.to_datetime(counter_raw[self.time_column])
        counter_raw = counter_raw.set_index(self.time_column)
        counter_agg = (
            counter_per_window(
                counter_raw[value_column_counter], window, handle_resets=handle_resets
            )
            .to_frame("units_produced")
            .reset_index()
            .rename(columns={self.time_column: "start"})
        )
        # ``totals`` carries true window starts; the counter raw resample labels.
        counter_agg["start"] = window_bounds(counter_agg["start"], window)[0]

        merged = totals.merge(counter_agg, on="start", how="inner")
        if merged.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)
        merged["carbon_intensity"] = np.where(
            merged["units_produced"] > 0,
            merged["total_kgco2e"] / merged["units_produced"],
            np.nan,
        )

        rolling = (
            merged["carbon_intensity"]
            .rolling(window=min(7, len(merged)), min_periods=1)
            .mean()
        )
        slope = rolling.diff()
        merged["trend"] = np.where(
            slope < -0.001,
            "improving",
            np.where(slope > 0.001, "worsening", "stable"),
        )
        return merged[["start", "end", "duration_seconds", *empty_cols[1:]]]

    def emission_factor_audit(self) -> pd.DataFrame:
        """Return the configured emission factors for audit and reporting.

        Returns:
            DataFrame: source_uuid, scope, emission_factor_kgco2e_per_unit
        """
        rows = [
            {
                "source_uuid": uid,
                "scope": self.scope_map.get(uid, 2),
                "emission_factor_kgco2e_per_unit": factor,
            }
            for uid, factor in self.emission_factors.items()
        ]
        return pd.DataFrame(
            rows, columns=["source_uuid", "scope", "emission_factor_kgco2e_per_unit"]
        )
