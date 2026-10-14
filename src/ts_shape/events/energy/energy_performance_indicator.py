import logging

import numpy as np
import pandas as pd

from ts_shape.events._windows import with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_per_window
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)


class EnergyPerformanceIndicatorEvents(Base):
    """Energy: Performance Indicator (EnPI) per ISO 50001

    Calculate energy consumed per unit of production output (EnPI = kWh/unit).
    Tracks EnPI against a rolling baseline and identifies improvement/degradation
    trends. Supports comparison across multiple meters / production areas.

    Supports two data models via constructor parameters:

    - Standard (ts-shape default)::

        EnergyPerformanceIndicatorEvents(df)
        # expects: systime | uuid | value_double | value_integer

    - Raw CSV::

        EnergyPerformanceIndicatorEvents(df, time_column="time", uuid_column="id")

    Methods:
    - enpi_by_window: EnPI (energy / units) per time window.
    - enpi_vs_baseline: EnPI vs rolling baseline with anomaly flags.
    - enpi_by_hierarchy: EnPI across multiple meters for area comparison.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "energy:enpi",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column
        self._uuid_column = uuid_column

    @renamed_kwargs(
        counter_column="value_column_counter", energy_column="value_column_energy"
    )
    def enpi_by_window(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_counter: str = "value_integer",
        window: str = "1D",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Calculate energy per unit produced (EnPI) for each time window.

        Args:
            meter_uuid: Identifier of the energy meter signal.
            counter_uuid: Identifier of the production counter signal.
            value_column_energy: Column containing energy readings.
            value_column_counter: Column containing counter readings.
            window: Aggregation window (e.g. '1D', '1h', '1W').
            handle_resets: ``True`` (default) treats a decreasing production
                counter as a reset and sums reset-aware increments per window
                (see :func:`ts_shape.utils.counters.counter_per_window`), so a
                window with a reset still reports what was produced.
                ``False`` uses the per-window ``max - min``.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid,
                       is_delta, energy_kwh, units_produced, enpi. ``end``
                       is ``start`` plus ``window``.
        """
        energy, counter = self._aggregate_pair(
            meter_uuid,
            counter_uuid,
            value_column_energy=value_column_energy,
            value_column_counter=value_column_counter,
            window=window,
            handle_resets=handle_resets,
        )
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "is_delta",
            "energy_kwh",
            "units_produced",
            "enpi",
        ]
        if energy.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        merged = energy.join(counter, how="inner").reset_index()
        merged = merged.rename(columns={self.time_column: "start"})
        merged["enpi"] = np.where(
            merged["units_produced"] > 0,
            merged["energy_kwh"] / merged["units_produced"],
            np.nan,
        )
        merged["uuid"] = self.event_uuid
        merged["source_uuid"] = meter_uuid
        merged["is_delta"] = True
        return with_window_bounds(merged[empty_cols], window)

    @renamed_kwargs(
        counter_column="value_column_counter", energy_column="value_column_energy"
    )
    def enpi_vs_baseline(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_counter: str = "value_integer",
        window: str = "1D",
        handle_resets: bool = True,
        baseline_window: int = 30,
        deviation_threshold: float = 0.1,
        baseline: str = "rolling",
        baseline_period: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """Compare current EnPI against a baseline.

        ``baseline="rolling"`` (default) compares each window with the mean
        EnPI of the preceding ``baseline_window`` windows (itself included).
        A rolling baseline drifts with the process, so it never shows a
        lasting improvement, and kWh/unit is biased by load: a fixed base
        load makes EnPI look worse whenever output is low.

        ``baseline="regression"`` is the ISO 50006 energy baseline: fit
        ``energy = a + b * units`` on a fixed baseline period, then compare
        each window's energy with what the model expects for its output.
        ``baseline_enpi`` is that expected energy per unit and
        ``deviation_pct`` is ``(actual - expected) / expected`` energy, so
        the comparison is normalized for production volume. Two columns are
        added: ``expected_energy_kwh`` and ``energy_savings_kwh``
        (``expected - actual``; its cumulative sum is the CUSUM of savings).

        Args:
            meter_uuid: Identifier of the energy meter signal.
            counter_uuid: Identifier of the production counter signal.
            value_column_energy: Column containing energy readings.
            value_column_counter: Column containing counter readings.
            window: Aggregation window.
            baseline_window: Number of windows for rolling baseline.
            deviation_threshold: Fractional deviation to flag as anomaly (0.1 = 10%).
            baseline: ``"rolling"`` or ``"regression"`` (see above).
            baseline_period: ``(start, end)`` of the regression baseline
                period, end exclusive. Default: the first
                ``baseline_window`` windows.
            handle_resets: ``True`` (default) treats a decreasing production
                counter as a reset and sums reset-aware increments per window
                (see :func:`ts_shape.utils.counters.counter_per_window`), so a
                window with a reset still reports what was produced.
                ``False`` uses the per-window ``max - min``.

        Returns:
            DataFrame: start, end, duration_seconds, uuid, source_uuid, enpi,
                       baseline_enpi,
                       deviation_pct, is_anomaly, trend
        """
        base = self.enpi_by_window(
            meter_uuid,
            counter_uuid,
            value_column_energy=value_column_energy,
            value_column_counter=value_column_counter,
            window=window,
            handle_resets=handle_resets,
        )
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "enpi",
            "baseline_enpi",
            "deviation_pct",
            "is_anomaly",
            "trend",
        ]
        if baseline not in ("rolling", "regression"):
            raise ValueError(
                f"baseline must be 'rolling' or 'regression', got {baseline!r}"
            )
        if baseline == "regression":
            empty_cols = empty_cols + ["expected_energy_kwh", "energy_savings_kwh"]
        if base.empty or len(base) < 2:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        if baseline == "regression":
            expected = self._regression_baseline(base, baseline_window, baseline_period)
            base["expected_energy_kwh"] = expected
            base["energy_savings_kwh"] = expected - base["energy_kwh"]
            base["baseline_enpi"] = np.where(
                base["units_produced"] > 0,
                expected / base["units_produced"].where(base["units_produced"] > 0),
                np.nan,
            )
            base["deviation_pct"] = np.where(
                expected > 0,
                (base["energy_kwh"] - expected) / np.where(expected > 0, expected, 1),
                0.0,
            )
        else:
            base["baseline_enpi"] = (
                base["enpi"].rolling(window=baseline_window, min_periods=1).mean()
            )
            base["deviation_pct"] = np.where(
                base["baseline_enpi"] > 0,
                (base["enpi"] - base["baseline_enpi"]) / base["baseline_enpi"],
                0.0,
            )
        base["is_anomaly"] = base["deviation_pct"].abs() > deviation_threshold

        # Trend: sign of rolling average slope over last few periods
        rolling = base["enpi"].rolling(window=min(7, len(base)), min_periods=1).mean()
        slope = rolling.diff()
        base["trend"] = np.where(
            slope < -0.01,
            "improving",
            np.where(slope > 0.01, "degrading", "stable"),
        )
        return with_window_bounds(base[empty_cols], window)

    @staticmethod
    def _regression_baseline(
        base: pd.DataFrame,
        baseline_window: int,
        baseline_period: tuple[str | pd.Timestamp, str | pd.Timestamp] | None,
    ) -> np.ndarray:
        """Expected energy per window from ``energy = a + b * units``."""
        if baseline_period is None:
            ref = base.iloc[:baseline_window]
        else:
            start = pd.Timestamp(baseline_period[0])
            end = pd.Timestamp(baseline_period[1])
            tz = getattr(base["start"].dt, "tz", None)
            if tz is not None:
                start = start.tz_localize(tz) if start.tzinfo is None else start
                end = end.tz_localize(tz) if end.tzinfo is None else end
            ref = base[(base["start"] >= start) & (base["start"] < end)]
        x = ref["units_produced"].to_numpy(dtype=float)
        y = ref["energy_kwh"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2 or np.ptp(x) == 0:
            raise ValueError(
                "regression baseline needs at least two baseline windows with "
                "different production volumes"
            )
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * base["units_produced"].to_numpy(dtype=float)

    @renamed_kwargs(
        counter_column="value_column_counter", energy_column="value_column_energy"
    )
    def enpi_by_hierarchy(
        self,
        meter_uuids: list[str],
        counter_uuid: str,
        *,
        value_column_energy: str = "value_double",
        value_column_counter: str = "value_integer",
        window: str = "1D",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Calculate EnPI per meter for cross-area comparison.

        Useful for comparing energy intensity across production lines, buildings,
        or hierarchy levels. Combine with series metadata to map meter_uuid to
        label_lvl / hierarchy columns.

        Args:
            meter_uuids: List of energy meter identifiers.
            counter_uuid: Shared production counter identifier.
            value_column_energy: Column containing energy readings.
            value_column_counter: Column containing counter readings.
            window: Aggregation window.
            handle_resets: ``True`` (default) treats a decreasing production
                counter as a reset and sums reset-aware increments per window
                (see :func:`ts_shape.utils.counters.counter_per_window`), so a
                window with a reset still reports what was produced.
                ``False`` uses the per-window ``max - min``.

        Returns:
            DataFrame: start, end, duration_seconds, meter_uuid, energy_kwh,
                       units_produced, enpi
        """
        frames = []
        for meter_uuid in meter_uuids:
            df = self.enpi_by_window(
                meter_uuid,
                counter_uuid,
                value_column_energy=value_column_energy,
                value_column_counter=value_column_counter,
                window=window,
                handle_resets=handle_resets,
            )
            if not df.empty:
                df = df.rename(columns={"source_uuid": "meter_uuid"})
                frames.append(
                    df[
                        [
                            "start",
                            "meter_uuid",
                            "energy_kwh",
                            "units_produced",
                            "enpi",
                        ]
                    ]
                )

        if not frames:
            return with_window_bounds(
                pd.DataFrame(
                    columns=[
                        "start",
                        "meter_uuid",
                        "energy_kwh",
                        "units_produced",
                        "enpi",
                    ]
                ),
                window,
            )

        return with_window_bounds(
            pd.concat(frames, ignore_index=True)
            .sort_values(["start", "meter_uuid"])
            .reset_index(drop=True),
            window,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _aggregate_pair(
        self,
        meter_uuid: str,
        counter_uuid: str,
        *,
        value_column_energy: str,
        value_column_counter: str,
        window: str,
        handle_resets: bool = True,
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """Resample energy and counter to window, return as two indexed DataFrames."""
        energy_raw = (
            self.dataframe[self.dataframe[self._uuid_column] == meter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        counter_raw = (
            self.dataframe[self.dataframe[self._uuid_column] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if energy_raw.empty or counter_raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        energy_raw[self.time_column] = pd.to_datetime(energy_raw[self.time_column])
        counter_raw[self.time_column] = pd.to_datetime(counter_raw[self.time_column])

        energy_idx = energy_raw.set_index(self.time_column)
        counter_idx = counter_raw.set_index(self.time_column)

        energy_agg = (
            energy_idx[value_column_energy]
            .resample(window)
            .sum()
            .to_frame("energy_kwh")
        )
        counter_agg = counter_per_window(
            counter_idx[value_column_counter], window, handle_resets=handle_resets
        ).to_frame("units_produced")

        return energy_agg, counter_agg
