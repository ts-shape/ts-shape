from typing import TYPE_CHECKING, Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import deprecated_method, keyword_only

if TYPE_CHECKING:
    from typing_extensions import deprecated


class EnvironmentalCompensationEvents(Base):
    """Quality: Environmental Compensation for Inline Sensors

    Detect and compensate for environmental effects (temperature,
    humidity, pressure) on inline measurement sensors. Tracks
    correlation, estimates sensitivity coefficients, applies
    compensation, and flags out-of-range conditions.

    Methods:
    - detect_correlation: Pearson correlation per window.
    - sensitivity: Linear regression coefficient per window.
    - compensated_signal: Apply compensation model to signal.
    - detect_env_exceedance: Flag out-of-range environmental conditions.

    ``env_uuid`` (the environment signal) is required; the constructor
    raises ``ValueError`` without it. ``env_value`` is optional and only
    sets the nominal environment value used by ``compensated_signal``.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        env_uuid: str | None = None,
        env_value: float | None = None,
        value_column: str = "value_double",
        event_uuid: str = "quality:env_compensation",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.env_uuid = env_uuid
        self.env_value = env_value
        self.value_column = value_column
        self.event_uuid = event_uuid
        self.time_column = time_column

        if env_uuid is None:
            # Every method reads the environment signal; env_value is only
            # the nominal (reference) value used by compensated_signal().
            raise ValueError(
                "env_uuid is required: every method reads the environment "
                "signal. env_value only sets the nominal environment value "
                "used by compensated_signal()."
            )

        # Extract signal data
        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

        # Extract environmental data
        if self.env_uuid is not None:
            self._env = (
                self.dataframe[self.dataframe["uuid"] == self.env_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            self._env[self.time_column] = pd.to_datetime(self._env[self.time_column])
        else:
            self._env = None

    def _build_aligned(self) -> pd.DataFrame:
        """Build time-aligned DataFrame with signal and env columns."""
        if self._env is None or self._env.empty or self.signal.empty:
            return pd.DataFrame()

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.rename(columns={self.value_column: "signal"})
        sig = sig.set_index(self.time_column)

        env = self._env[[self.time_column, self.value_column]].copy()
        env = env.rename(columns={self.value_column: "env"})
        env = env.set_index(self.time_column)

        # Merge on nearest time
        aligned = pd.merge_asof(
            sig.sort_index(),
            env.sort_index(),
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=pd.Timedelta("5m"),
        )
        return aligned.dropna()

    @keyword_only(1)
    def detect_correlation(self, window: str = "4h") -> pd.DataFrame:
        """Pearson correlation between measurement and env signal per window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, correlation, p_value,
            significant.
        """
        from scipy import stats

        cols = ["window_start", "correlation", "p_value", "significant"]
        aligned = self._build_aligned()
        if aligned.empty:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        for ts, group in aligned.resample(window):
            group = group.dropna()
            if len(group) < 5:
                continue

            r, p = stats.pearsonr(group["signal"], group["env"])

            events.append(
                {
                    "window_start": ts,
                    "correlation": float(r),
                    "p_value": float(p),
                    "significant": bool(p < 0.05),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    if TYPE_CHECKING:

        @deprecated(
            "EnvironmentalCompensationEvents.estimate_sensitivity() is deprecated, use .sensitivity()"
        )
        def estimate_sensitivity(self, window: str = "8h") -> pd.DataFrame: ...

    else:
        estimate_sensitivity = deprecated_method("sensitivity")

    @keyword_only(1)
    def sensitivity(self, window: str = "8h") -> pd.DataFrame:
        """Estimate environmental sensitivity coefficient per window.

        Fits measurement = a × env + b per window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, coefficient, intercept,
            r_squared, unit_change_per_degree.
        """
        from scipy import stats

        cols = [
            "window_start",
            "coefficient",
            "intercept",
            "r_squared",
            "unit_change_per_degree",
        ]
        aligned = self._build_aligned()
        if aligned.empty:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        for ts, group in aligned.resample(window):
            group = group.dropna()
            if len(group) < 5:
                continue

            slope, intercept, r_value, _, _ = stats.linregress(
                group["env"], group["signal"]
            )

            events.append(
                {
                    "window_start": ts,
                    "coefficient": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_value**2),
                    "unit_change_per_degree": float(slope),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    @keyword_only(1)
    def compensated_signal(self, coefficient: float | None = None) -> pd.DataFrame:
        """Apply linear compensation to the measurement signal.

        corrected = measured - coefficient × (env - env_nominal)

        If coefficient is not provided, estimates from data using
        overall linear regression.

        Args:
            coefficient: Environmental sensitivity coefficient. If None,
                estimated from data.

        Returns:
            DataFrame with columns: systime, uuid, value_double,
            compensated_value, correction_applied, source_uuid. ``uuid``
            is the event uuid and ``source_uuid`` the measured signal's uuid.
        """
        from scipy import stats

        cols = [
            "systime",
            "uuid",
            "value_double",
            "compensated_value",
            "correction_applied",
            "source_uuid",
        ]
        aligned = self._build_aligned()
        if aligned.empty:
            return pd.DataFrame(columns=cols)

        # Determine nominal env value
        if self.env_value is not None:
            env_nominal = self.env_value
        else:
            env_nominal = float(aligned["env"].mean())

        # Estimate coefficient if not provided
        if coefficient is None:
            if len(aligned) < 5:
                return pd.DataFrame(columns=cols)
            slope, _, _, _, _ = stats.linregress(aligned["env"], aligned["signal"])
            coefficient = float(slope)

        # Apply compensation
        correction = coefficient * (aligned["env"] - env_nominal)
        compensated = aligned["signal"] - correction

        result = pd.DataFrame(
            {
                "systime": aligned.index,
                "uuid": self.event_uuid,
                "value_double": aligned["signal"].values,
                "compensated_value": compensated.values,
                "correction_applied": correction.values,
                "source_uuid": self.signal_uuid,
            }
        )

        return result

    @keyword_only(2)
    def detect_env_exceedance(
        self, valid_range: tuple[float, float], window: str = "1h"
    ) -> pd.DataFrame:
        """Flag windows where environmental conditions exceed valid range.

        Args:
            valid_range: Tuple (min, max) of acceptable env values.
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, window_end, env_mean,
            env_min, env_max, exceedance_type, severity.
        """
        cols = [
            "window_start",
            "window_end",
            "env_mean",
            "env_min",
            "env_max",
            "exceedance_type",
            "severity",
        ]

        if self._env is None or self._env.empty:
            return pd.DataFrame(columns=cols)

        env = self._env[[self.time_column, self.value_column]].copy()
        env = env.set_index(self.time_column)

        valid_min, valid_max = valid_range
        valid_span = valid_max - valid_min

        events: list[dict[str, Any]] = []
        for ts, group in env.resample(window):
            vals = group[self.value_column].dropna()
            if vals.empty:
                continue

            env_mean = float(vals.mean())
            env_min = float(vals.min())
            env_max = float(vals.max())

            exceedance_type = None
            if env_max > valid_max and env_min < valid_min:
                exceedance_type = "both"
            elif env_max > valid_max:
                exceedance_type = "high"
            elif env_min < valid_min:
                exceedance_type = "low"

            if exceedance_type is None:
                continue

            # Severity based on how far outside range
            if exceedance_type == "high":
                deviation = env_max - valid_max
            elif exceedance_type == "low":
                deviation = valid_min - env_min
            else:
                deviation = max(env_max - valid_max, valid_min - env_min)

            if valid_span > 0:
                deviation_pct = deviation / valid_span
            else:
                deviation_pct = 1.0

            if deviation_pct > 0.5:
                severity = "critical"
            elif deviation_pct > 0.2:
                severity = "high"
            elif deviation_pct > 0.05:
                severity = "medium"
            else:
                severity = "low"

            window_end = ts + pd.to_timedelta(window)
            events.append(
                {
                    "window_start": ts,
                    "window_end": window_end,
                    "env_mean": float(env_mean),
                    "env_min": float(env_min),
                    "env_max": float(env_max),
                    "exceedance_type": exceedance_type,
                    "severity": severity,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
