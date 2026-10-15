import logging
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import (
    caller_stacklevel,
    deprecated_class,
    deprecated_method,
    keyword_only,
    renamed_kwargs,
)

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


def _unique_samples(
    dataframe: pd.DataFrame, signal_uuid: str, value_column: str
) -> pd.DataFrame:
    """Rows of ``signal_uuid`` with exact repeated samples dropped (warns)."""
    rows = dataframe[dataframe["uuid"] == signal_uuid]
    unique = rows.drop_duplicates(subset=["systime", value_column])
    dropped = len(rows) - len(unique)
    if dropped:
        warnings.warn(
            f"{signal_uuid!r}: {dropped} repeated sample(s) (same time and "
            "value) were counted once in the CUSUM.",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )
    return unique.copy()


#: d2 for subgroups of n = 2 (the moving range of two consecutive values),
#: Montgomery, Introduction to Statistical Quality Control, Appendix VI.
D2_N2 = 1.128

SigmaMethod = Literal["moving_range", "std"]
_SIGMA_METHODS = ("moving_range", "std")


def _check_sigma_method(sigma_method: str) -> None:
    if sigma_method not in _SIGMA_METHODS:
        raise ValueError(
            f"Unknown sigma_method: {sigma_method!r}. Use 'moving_range' or 'std'."
        )


def _sigma_hat(values: pd.Series, sigma_method: str = "moving_range") -> float:
    """Process sigma of an individuals series.

    ``"moving_range"`` (the Shewhart individuals chart): ``MR-bar / d2`` with
    ``MR_i = |x_i - x_{i-1}|`` over consecutive values in the given order and
    ``d2 = 1.128`` (n = 2); Montgomery, Introduction to Statistical Quality
    Control, section 6.4. ``"std"``: the sample standard deviation (ddof=1),
    which also contains any between-sample variation (drift, shifts).

    NaN values are dropped first. Fewer than two values give NaN.
    """
    _check_sigma_method(sigma_method)
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(vals) < 2:
        return float("nan")
    if sigma_method == "std":
        return float(vals.std(ddof=1))
    return float(vals.diff().abs().iloc[1:].mean() / D2_N2)


def _cusum(
    values: np.ndarray, target: float, slack: float
) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower CUSUM: ``C_i = max(0, C_{i-1} + deviation_i - slack)``."""
    high = np.zeros(len(values))
    low = np.zeros(len(values))
    hi = lo = 0.0
    for i, value in enumerate(values.tolist()):
        hi = max(0.0, hi + value - target - slack)
        lo = max(0.0, lo + target - value - slack)
        high[i] = hi
        low[i] = lo
    return high, low


class StatisticalProcessControlEvents(Base):
    """
    Inherits from Base and applies SPC rules (Western Electric Rules) to a DataFrame for event detection.
    Processes data based on control limit UUIDs, actual value UUIDs, and generates events with an event UUID.

    ``rule_1`` .. ``rule_8`` are the eight tests of Nelson (1984), "The
    Shewhart Control Chart--Tests for Special Causes", Journal of Quality
    Technology 16(4); rules 1, 5 and 6 are the Western Electric zone rules
    (Montgomery, Introduction to Statistical Quality Control, Table 5.1).
    The center line and sigma come from the ``tolerance_uuid`` rows (the
    reference / phase I data).

    Sigma: by default (``sigma_method="moving_range"``) the textbook
    individuals-chart estimate ``sigma_hat = MR-bar / d2`` with ``d2 = 1.128``,
    where ``MR-bar`` is the mean moving range of consecutive reference values
    (Montgomery, Introduction to Statistical Quality Control, section 6.4,
    "Control charts for individual measurements"). It measures the
    short-term (common-cause) variation only, so a drift or shift inside the
    reference does not widen the limits. ``sigma_method="std"`` restores the
    sample standard deviation (ddof=1) that ts-shape used before October
    2026. The same sigma is used by :meth:`control_limits`, the rules 1-8 zones,
    :meth:`detect_cusum_shifts` and the EWMA chart of
    :meth:`dynamic_control_limits`.

    Example:
        >>> import pandas as pd, ts_shape
        >>> ref = ts_shape.make_timeseries(["ref"], n_points=50, seed=1)  # in-control
        >>> act = ts_shape.make_timeseries(["act"], n_points=50, n_outliers=2)
        >>> spc = ts_shape.StatisticalProcessControlEvents(
        ...     pd.concat([ref, act]), "value_double", "ref", "act", "quality:spc")
        >>> out = spc.detect_rule_violations(selected_rules=["rule_1"])
        >>> out[["systime", "value", "rule", "severity"]].round({"value": 1})
                      systime  value    rule  severity
        0 2025-01-01 00:19:30  112.7  rule_1  critical
        1 2025-01-01 00:24:30   88.1  rule_1  critical
    """

    #: CUSUM: ``high`` is a detected shift, ``critical`` twice the threshold.
    _severity_levels = {"detect_cusum_shifts": {"high": "warn"}}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        value_column: str,
        tolerance_uuid: str,
        actual_uuid: str,
        event_uuid: str,
        *,
        sigma_method: SigmaMethod = "moving_range",
    ) -> None:
        """
        Initializes StatisticalProcessControlEvents with UUIDs for tolerance, actual, and event values.
        Inherits the sorted dataframe from the Base class.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the data to be processed.
            value_column (str): The column containing the values to monitor.
            tolerance_uuid (str): UUID identifier for the reference rows the
                center line and sigma are estimated from.
            actual_uuid (str): UUID identifier for rows containing actual values.
            event_uuid (str): UUID to assign to generated events.
            sigma_method: How sigma is estimated from the reference rows.
                ``"moving_range"`` (default, textbook): ``MR-bar / 1.128``
                from consecutive reference values. ``"std"``: the sample
                standard deviation (ddof=1), the behavior before October 2026.
        """
        _check_sigma_method(sigma_method)
        super().__init__(dataframe)  # Initialize the Base class
        self.sigma_method: str = sigma_method
        self.value_column: str = value_column
        self.tolerance_uuid: str = tolerance_uuid
        self.actual_uuid: str = actual_uuid
        self.event_uuid: str = event_uuid
        if not self.dataframe.empty:
            self._validate_column(self.dataframe, value_column)
            self._validate_uuid(self.dataframe, tolerance_uuid)
            self._validate_uuid(self.dataframe, actual_uuid)

    if TYPE_CHECKING:

        @deprecated(
            "StatisticalProcessControlEvents.calculate_control_limits() is deprecated, use .control_limits()"
        )
        def calculate_control_limits(self) -> pd.DataFrame: ...

    else:
        calculate_control_limits = deprecated_method("control_limits")

    def _reference_values(self) -> pd.Series:
        """Reference (tolerance) values in time order."""
        df = self.dataframe[self.dataframe["uuid"] == self.tolerance_uuid]
        df = df.sort_values(by="systime", kind="stable")
        return pd.to_numeric(df[self.value_column], errors="coerce")

    def control_limits(self) -> pd.DataFrame:
        """
        Shewhart individuals-chart limits (center ± 1σ, 2σ, 3σ) from the reference rows.

        Center line ``mean`` is the mean of the reference values; sigma is
        ``MR-bar / d2`` (``d2 = 1.128``) of consecutive reference values with
        the default ``sigma_method="moving_range"`` (Montgomery, Introduction
        to Statistical Quality Control, section 6.4: ``UCL = x-bar +
        3 MR-bar / d2``, ``LCL = x-bar - 3 MR-bar / d2``), or the sample
        standard deviation with ``sigma_method="std"``.

        Returns:
            pd.DataFrame: One row with columns mean, 1sigma_upper,
                1sigma_lower, 2sigma_upper, 2sigma_lower, 3sigma_upper and
                3sigma_lower.
        """
        df = self._reference_values()
        mean = df.mean()
        sigma = _sigma_hat(df, self.sigma_method)
        if not sigma > 0:
            # Fewer than two tolerance rows (sigma NaN) or a constant signal
            # (sigma 0): the limits are NaN or collapse onto the mean, so the
            # Western Electric rules silently never fire (or always fire).
            warnings.warn(
                f"Control limits for uuid {self.tolerance_uuid!r} are degenerate "
                f"(sigma={sigma} from {len(df)} row(s)); rule checks will not be "
                "meaningful.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )

        control_limits = {
            "mean": mean,
            "1sigma_upper": mean + sigma,
            "1sigma_lower": mean - sigma,
            "2sigma_upper": mean + 2 * sigma,
            "2sigma_lower": mean - 2 * sigma,
            "3sigma_upper": mean + 3 * sigma,
            "3sigma_lower": mean - 3 * sigma,
        }

        return pd.DataFrame([control_limits])

    if TYPE_CHECKING:

        @deprecated(
            "StatisticalProcessControlEvents.calculate_dynamic_control_limits() is deprecated, use .dynamic_control_limits()"
        )
        def calculate_dynamic_control_limits(
            self,
            method: Literal["moving_range", "ewma"] = "moving_range",
            window_size: int = 20,
            ewma_lambda: float | None = None,
            n_sigma: float = 3.0,
        ) -> pd.DataFrame: ...

    else:
        calculate_dynamic_control_limits = deprecated_method("dynamic_control_limits")

    @keyword_only(1)
    @renamed_kwargs(window="window_size")
    def dynamic_control_limits(
        self,
        method: Literal["moving_range", "ewma"] = "moving_range",
        window_size: int = 20,
        ewma_lambda: float | None = None,
        n_sigma: float = 3.0,
    ) -> pd.DataFrame:
        """
        Control limits that adapt over time: a rolling individuals chart or an EWMA chart.

        ``method="moving_range"`` -- rolling Shewhart individuals chart
        (Montgomery, Introduction to Statistical Quality Control, section
        6.4). Each actual point ``x_i`` is tested against limits estimated
        from the ``window_size`` points *before* it, so a point never
        widens its own limits::

            center_i = mean(x_{i-w}, ..., x_{i-1})
            sigma_i  = mean(|x_j - x_{j-1}|, j = i-w+1 .. i-1) / 1.128

        (``sigma_method="std"``: ``sigma_i`` is the sample std of the same
        ``w`` previous points). The first ``window_size`` points have no
        complete history and get NaN limits, as does any point whose
        history contains a NaN value.

        ``method="ewma"`` -- EWMA chart (Montgomery, section 9.2)::

            z_i = lambda * x_i + (1 - lambda) * z_{i-1},   z_0 = mu_0
            limits_i = mu_0 +/- L * sigma * sqrt(lambda / (2 - lambda)
                                              * (1 - (1 - lambda)^(2 i)))

        ``mu_0`` and ``sigma`` come from the reference (``tolerance_uuid``)
        rows, sigma by the instance's ``sigma_method``. ``lambda`` is
        ``ewma_lambda`` when given, else ``2 / (window_size + 1)`` (the
        smoothing of an EWMA with span ``window_size``). Rows with a NaN
        value get NaN ``ewma`` and limits and do not advance ``i``.

        Args:
            method: ``"moving_range"`` (default) or ``"ewma"``.
            window_size: Number of previous points the rolling chart uses
                (at least 2); for ``"ewma"`` the span that sets ``lambda``
                when ``ewma_lambda`` is not given (default: 20).
            ewma_lambda: EWMA weight ``0 < lambda <= 1`` (``"ewma"`` only).
            n_sigma: Width ``L`` of the ``lcl`` / ``ucl`` columns in sigmas
                (default 3.0).

        Returns:
            pd.DataFrame: One row per actual point. Columns: systime, value
                (the actual value), ewma (``z_i``; ``"ewma"`` only), mean
                (the center line: ``center_i``, or ``mu_0`` for EWMA),
                1sigma_upper .. 3sigma_lower (center +/- k sigma_i; for EWMA
                ``mu_0 +/- k`` times the EWMA statistic's sigma), lcl / ucl
                (center -/+ ``n_sigma`` times the same sigma), uuid (the
                event uuid) and source_uuid (the actual signal's uuid). For
                EWMA compare ``ewma`` (not ``value``) with the limits.
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df = df.sort_values(by="systime", kind="stable")
        x = pd.to_numeric(df[self.value_column], errors="coerce").astype(float)
        x = x.reset_index(drop=True)

        if method == "moving_range":
            if window_size < 2:
                raise ValueError("window_size must be at least 2 for 'moving_range'")
            center = x.rolling(window=window_size, min_periods=window_size).mean()
            center = center.shift(1)
            if self.sigma_method == "moving_range":
                mr = x.diff().abs()
                sigma = mr.rolling(
                    window=window_size - 1, min_periods=window_size - 1
                ).mean()
                sigma = sigma.shift(1) / D2_N2
            else:
                sigma = x.rolling(window=window_size, min_periods=window_size).std()
                sigma = sigma.shift(1)
            columns: dict[str, object] = {
                "systime": df["systime"].to_numpy(),
                "value": x,
            }
        elif method == "ewma":
            lam = 2.0 / (window_size + 1) if ewma_lambda is None else ewma_lambda
            if not 0 < lam <= 1:
                raise ValueError(f"ewma_lambda must be in (0, 1], got {lam}")
            ref = self._reference_values()
            mu0 = float(ref.mean())
            ref_sigma = _sigma_hat(ref, self.sigma_method)
            if not ref_sigma > 0:
                warnings.warn(
                    f"EWMA limits for uuid {self.tolerance_uuid!r} are degenerate "
                    f"(sigma={ref_sigma} from {ref.notna().sum()} reference "
                    "value(s)); the limits are NaN.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )
                ref_sigma = float("nan")
            values = x.to_numpy()
            z = np.full(len(values), np.nan)
            steps = np.full(len(values), np.nan)
            prev, i = mu0, 0
            for pos, value in enumerate(values.tolist()):
                if np.isnan(value):
                    continue
                i += 1
                prev = lam * value + (1 - lam) * prev
                z[pos] = prev
                steps[pos] = i
            sigma = pd.Series(
                ref_sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * steps)))
            )
            center = pd.Series(np.where(np.isnan(steps), np.nan, mu0))
            columns = {
                "systime": df["systime"].to_numpy(),
                "value": x,
                "ewma": z,
            }
        else:
            raise ValueError(f"Unknown method: {method}. Use 'moving_range' or 'ewma'.")

        columns["mean"] = center
        for k in (1, 2, 3):
            columns[f"{k}sigma_upper"] = center + k * sigma
            columns[f"{k}sigma_lower"] = center - k * sigma
        columns["lcl"] = center - n_sigma * sigma
        columns["ucl"] = center + n_sigma * sigma
        control_limits = pd.DataFrame(columns)
        control_limits["uuid"] = self.event_uuid
        control_limits["source_uuid"] = self.actual_uuid
        return control_limits.reset_index(drop=True)

    # The rule definitions follow Nelson (1984), "The Shewhart Control
    # Chart--Tests for Special Causes", Journal of Quality Technology 16(4),
    # pp. 237-239, which extends the Western Electric zone rules (Western
    # Electric Statistical Quality Control Handbook, 1956; Montgomery,
    # Introduction to Statistical Quality Control, Table 5.1). Zones are
    # measured from the chart's center line (the tolerance mean): zone C is
    # within 1 sigma, zone B between 1 and 2 sigma, zone A between 2 and 3
    # sigma. Run lengths count *points*: "six points steadily increasing" is
    # five consecutive increases. A rule flags the point that completes the
    # pattern (the last point of every qualifying window).
    _RULE_NAMES = tuple(f"rule_{i}" for i in range(1, 9))

    def _rule_mask(
        self,
        rule: str,
        values: pd.Series,
        limits: pd.DataFrame | None,
        center: float | None = None,
    ) -> pd.Series:
        """Boolean mask of the points that complete a violation of ``rule``.

        Shared by the per-rule methods, :meth:`detect_violations` and
        :meth:`detect_rule_violations` so all paths flag the same points.

        Args:
            rule: One of ``"rule_1"`` .. ``"rule_8"``.
            values: The actual values, sorted by time.
            limits: Control limits as returned by :meth:`control_limits`
                (first row used). Only rules 2-4 work without limits.
            center: Center line for rule 2; defaults to ``limits["mean"]``,
                or to the mean of ``values`` when no limits are given.

        Returns:
            pd.Series: Boolean mask aligned with ``values``.
        """
        if rule in ("rule_2", "rule_3", "rule_4") and limits is None:
            lim: dict[str, float] = {}
        elif limits is None:
            raise ValueError(f"{rule} needs control limits")
        else:
            lim = {c: float(limits[c].values[0]) for c in limits.columns}
        if rule == "rule_1":
            # Nelson 1: one point more than 3 sigma from the center line.
            return (values > lim["3sigma_upper"]) | (values < lim["3sigma_lower"])
        if rule == "rule_2":
            # Nelson 2: nine points in a row on the same side of the center
            # line (a point exactly on the center line breaks the run).
            if center is None:
                center = lim["mean"] if lim else float(values.mean())
            above = (values > center).astype(int).rolling(window=9).sum() == 9
            below = (values < center).astype(int).rolling(window=9).sum() == 9
            return above | below
        if rule == "rule_3":
            # Nelson 3: six points in a row steadily increasing or decreasing
            # = five consecutive strictly positive (or negative) differences.
            diffs = values.diff()
            inc = (diffs > 0).astype(int).rolling(window=5).sum() == 5
            dec = (diffs < 0).astype(int).rolling(window=5).sum() == 5
            return inc | dec
        if rule == "rule_4":
            # Nelson 4: fourteen points in a row alternating up and down
            # = thirteen consecutive non-zero differences whose signs alternate.
            signs = np.sign(values.diff().to_numpy(dtype=float))
            flips = np.zeros(len(signs), dtype=int)
            if len(signs) > 1:
                flips[1:] = (signs[1:] * signs[:-1]) < 0  # NaN/0 compare False
            # 13 alternating differences = 12 consecutive sign flips.
            return pd.Series(flips, index=values.index).rolling(window=12).sum() == 12
        if rule == "rule_5":
            # Nelson 5 / Western Electric 2: two out of three points in a row
            # more than 2 sigma from the center line, on the same side
            # (zone A *or beyond*).
            up = (values > lim["2sigma_upper"]).astype(int).rolling(window=3).sum()
            lo = (values < lim["2sigma_lower"]).astype(int).rolling(window=3).sum()
            return (up >= 2) | (lo >= 2)
        if rule == "rule_6":
            # Nelson 6 / Western Electric 3: four out of five points in a row
            # more than 1 sigma from the center line, on the same side
            # (zone B *or beyond*).
            up = (values > lim["1sigma_upper"]).astype(int).rolling(window=5).sum()
            lo = (values < lim["1sigma_lower"]).astype(int).rolling(window=5).sum()
            return (up >= 4) | (lo >= 4)
        if rule == "rule_7":
            # Nelson 7: fifteen points in a row within 1 sigma (zone C),
            # either side of the center line.
            within = (values < lim["1sigma_upper"]) & (values > lim["1sigma_lower"])
            return within.astype(int).rolling(window=15).sum() == 15
        if rule == "rule_8":
            # Nelson 8: eight points in a row with none within 1 sigma (none
            # in zone C), on either side of the center line.
            outside = (values > lim["1sigma_upper"]) | (values < lim["1sigma_lower"])
            return outside.astype(int).rolling(window=8).sum() == 8
        raise ValueError(f"Unknown rule: {rule!r}")

    def _apply_rule(
        self, rule: str, df: pd.DataFrame, limits: pd.DataFrame | None
    ) -> pd.DataFrame:
        out = df.copy()
        out[rule] = self._rule_mask(rule, out[self.value_column], limits).astype(bool)
        return out[out[rule]]

    def rule_1(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 1 (Nelson 1984, test 1): one point more than 3σ from the center line.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_1`` column.
        """
        return self._apply_rule("rule_1", df, limits)

    def rule_2(
        self, df: pd.DataFrame, *, limits: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Rule 2 (Nelson 1984, test 2): nine points in a row on the same side of the center line.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                the center line is ``limits["mean"]``. When omitted, the
                mean of ``df``'s values is used as the center line.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_2`` column (the ninth point of each run is flagged).
        """
        return self._apply_rule("rule_2", df, limits)

    def rule_3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 3 (Nelson 1984, test 3): six points in a row steadily increasing or decreasing.

        Six points means five consecutive strictly positive (or strictly
        negative) differences; an unchanged value breaks the run.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_3`` column (the sixth point of each run is flagged).
        """
        return self._apply_rule("rule_3", df, None)

    def rule_4(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 4 (Nelson 1984, test 4): fourteen points in a row alternating up and down.

        Fourteen points means thirteen consecutive non-zero differences
        whose signs alternate; an unchanged value breaks the run.

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_4`` column (the fourteenth point of each run is flagged).
        """
        return self._apply_rule("rule_4", df, None)

    def rule_5(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 5 (Nelson 1984, test 5): two out of three points in a row more than 2σ from the center line, same side.

        Points beyond 3σ count as beyond 2σ (zone A or beyond).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_5`` column (the last point of each qualifying
                three-point window is flagged).
        """
        return self._apply_rule("rule_5", df, limits)

    def rule_6(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 6 (Nelson 1984, test 6): four out of five points in a row more than 1σ from the center line, same side.

        Points beyond 2σ or 3σ count as beyond 1σ (zone B or beyond).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_6`` column (the last point of each qualifying
                five-point window is flagged).
        """
        return self._apply_rule("rule_6", df, limits)

    def rule_7(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 7 (Nelson 1984, test 7): fifteen points in a row within 1σ of the center line (either side).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_7`` column.
        """
        return self._apply_rule("rule_7", df, limits)

    def rule_8(self, df: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 8 (Nelson 1984, test 8): eight points in a row with none within 1σ of the center line.

        The points may lie on either side of the center line; what matters
        is that none falls in zone C (within 1σ).

        Args:
            df: Rows of the actual signal, sorted by time. Not modified.
            limits: Control limits as returned by :meth:`control_limits`;
                only the first row is used.

        Returns:
            pd.DataFrame: The violating rows (a copy) with a boolean
                ``rule_8`` column.
        """
        return self._apply_rule("rule_8", df, limits)

    if TYPE_CHECKING:

        @deprecated(
            "StatisticalProcessControlEvents.apply_rules_vectorized() is deprecated, use .detect_rule_violations()"
        )
        def apply_rules_vectorized(
            self, selected_rules: list[str] | None = None
        ) -> pd.DataFrame: ...

    else:
        apply_rules_vectorized = deprecated_method("detect_rule_violations")

    @keyword_only(1)
    def detect_rule_violations(
        self, selected_rules: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Applies SPC rules using vectorized operations with optimized multi-rule processing.
        Processes multiple rules in fewer passes through the data for better performance.

        Args:
            selected_rules (Optional[List[str]]): List of rule names to apply.
                If None, applies all rules.

        Returns:
            pd.DataFrame: DataFrame with rule violations. Columns: systime,
                value, rule, severity, uuid (the event uuid) and source_uuid
                (the actual signal's uuid).
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime").reset_index(drop=True)

        limits = self.control_limits()

        if selected_rules is None:
            selected_rules = [
                "rule_1",
                "rule_2",
                "rule_3",
                "rule_4",
                "rule_5",
                "rule_6",
                "rule_7",
                "rule_8",
            ]

        rule_severity = {
            "rule_1": "critical",
            "rule_2": "medium",
            "rule_3": "medium",
            "rule_4": "medium",
            "rule_5": "high",
            "rule_6": "medium",
            "rule_7": "low",
            "rule_8": "low",
        }
        values = df[self.value_column]
        violations = []
        for rule in self._RULE_NAMES:
            if rule not in selected_rules:
                continue
            mask = self._rule_mask(rule, values, limits)
            for idx in df[mask].index:
                violations.append(
                    {
                        "systime": df.loc[idx, "systime"],
                        "value": df.loc[idx, self.value_column],
                        "rule": rule,
                        "severity": rule_severity[rule],
                    }
                )

        # Create violations DataFrame
        if violations:
            violations_df = pd.DataFrame(violations)
            violations_df["uuid"] = self.event_uuid
            violations_df["source_uuid"] = self.actual_uuid
            return violations_df.drop_duplicates()
        else:
            return pd.DataFrame(
                columns=["systime", "value", "rule", "severity", "uuid", "source_uuid"]
            )

    @keyword_only(1)
    def detect_cusum_shifts(
        self, target: float | None = None, k: float = 0.5, h: float = 5.0
    ) -> pd.DataFrame:
        """
        Detect process shifts using CUSUM (Cumulative Sum) control chart.

        CUSUM charts are effective at detecting small shifts in the process mean
        and are more sensitive than traditional Shewhart control charts.

        Tabular CUSUM (Montgomery, Introduction to Statistical Quality
        Control, section 9.1)::

            C+_i = max(0, x_i - (mu_0 + K) + C+_{i-1})
            C-_i = max(0, (mu_0 - K) - x_i + C-_{i-1}),   C+_0 = C-_0 = 0

        with ``K = k * sigma`` and decision interval ``H = h * sigma``; a
        point is a shift when ``C+_i > H`` or ``C-_i > H``. ``sigma`` is
        estimated from the reference rows by the instance's
        ``sigma_method`` (default ``MR-bar / 1.128``).

        Args:
            target (Optional[float]): Target mean value. If None, uses the mean of tolerance data.
            k (float): Reference value (slack parameter) in sigmas, typically 0.5 to 1.0.
                Smaller k detects smaller shifts. Default: 0.5
            h (float): Decision interval in sigmas. Typical values are 4-5.
                Smaller h gives faster detection but more false alarms. Default: 5.0

        Returns:
            pd.DataFrame: DataFrame with CUSUM statistics and detected shifts.
                Columns: systime, value_column, cusum_high, cusum_low,
                shift_direction, severity, uuid (the event uuid) and
                source_uuid (the actual signal's uuid).
        """
        # A repeated sample (same time, same value) would be summed twice and
        # could cross the threshold on its own: keep one copy of each.
        df = _unique_samples(self.dataframe, self.actual_uuid, self.value_column)
        df = df.sort_values(by="systime").reset_index(drop=True)
        tolerance_df = _unique_samples(
            self.dataframe, self.tolerance_uuid, self.value_column
        )

        # Determine target and sigma
        if target is None:
            target = tolerance_df[self.value_column].mean()

        tolerance_df = tolerance_df.sort_values(by="systime", kind="stable")
        sigma = _sigma_hat(tolerance_df[self.value_column], self.sigma_method)
        result_cols = [
            "systime",
            self.value_column,
            "cusum_high",
            "cusum_low",
            "shift_direction",
            "severity",
            "uuid",
            "source_uuid",
        ]
        if not sigma > 0:
            # sigma NaN (fewer than two rows) or 0 (constant): k*sigma and the
            # decision interval h*sigma are 0/NaN, which flags every deviation
            # (or none). Report nothing rather than a wall of false shifts.
            warnings.warn(
                f"CUSUM skipped for uuid {self.tolerance_uuid!r}: tolerance sigma "
                f"is {sigma}; at least two non-constant tolerance values are needed.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return pd.DataFrame(columns=result_cols)

        cusum_high, cusum_low = _cusum(
            df[self.value_column].to_numpy(dtype=float), float(target), float(k * sigma)
        )

        # Add CUSUM statistics to dataframe
        df["cusum_high"] = cusum_high
        df["cusum_low"] = cusum_low

        # Detect shifts (when CUSUM exceeds threshold h*sigma)
        threshold = h * sigma
        df["shift_detected"] = (cusum_high > threshold) | (cusum_low > threshold)
        df["shift_direction"] = np.where(
            cusum_high > threshold,
            "upward",
            np.where(cusum_low > threshold, "downward", "none"),
        )
        df["severity"] = np.where(
            df["shift_detected"],
            np.where(
                (cusum_high > 2 * threshold) | (cusum_low > 2 * threshold),
                "critical",
                "high",
            ),
            "none",
        )

        # Add event UUID to detected shifts
        df.loc[df["shift_detected"], "uuid"] = self.event_uuid
        df["source_uuid"] = self.actual_uuid

        # Return only rows with detected shifts
        result = df[df["shift_detected"]].copy()
        return result[result_cols]

    def interpret_violations(self, violations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable interpretations to rule violations.

        Args:
            violations_df (pd.DataFrame): DataFrame with rule violations
                (output from detect_rule_violations or process methods)

        Returns:
            pd.DataFrame: Enhanced DataFrame with interpretation and recommendation columns.
        """
        # Rule interpretations
        rule_interpretations = {
            "rule_1": {
                "interpretation": "One or more points beyond 3-sigma control limits",
                "meaning": "Indicates a special cause - an unusual event or significant process change",
                "recommendation": "Investigate immediately for assignable causes such as equipment failure, "
                "operator error, or material defects",
                "default_severity": "critical",
            },
            "rule_2": {
                "interpretation": "Nine consecutive points on one side of the center line",
                "meaning": "Process mean has shifted - indicates a sustained change in process level",
                "recommendation": "Check for systematic changes in materials, methods, equipment settings, "
                "or environmental conditions",
                "default_severity": "medium",
            },
            "rule_3": {
                "interpretation": "Six consecutive points steadily increasing or decreasing",
                "meaning": "Indicates a trend - gradual systematic change in the process",
                "recommendation": "Look for tool wear, temperature drift, operator fatigue, "
                "or gradual equipment degradation",
                "default_severity": "medium",
            },
            "rule_4": {
                "interpretation": "Fourteen consecutive points alternating up and down",
                "meaning": "Indicates systematic oscillation - two alternating causes affecting the process",
                "recommendation": "Check for alternating operators, materials from two sources, "
                "or temperature cycling effects",
                "default_severity": "medium",
            },
            "rule_5": {
                "interpretation": "Two out of three consecutive points beyond 2-sigma "
                "limits on the same side",
                "meaning": "Process variation has increased or mean is shifting",
                "recommendation": "Monitor closely and prepare to investigate. May indicate the start "
                "of a larger problem",
                "default_severity": "high",
            },
            "rule_6": {
                "interpretation": "Four out of five consecutive points beyond 1-sigma "
                "limits on the same side",
                "meaning": "Process variation or mean has likely changed",
                "recommendation": "Check for changes in process inputs or measurement system accuracy",
                "default_severity": "medium",
            },
            "rule_7": {
                "interpretation": "Fifteen consecutive points within 1-sigma of center line",
                "meaning": "Unusually low variation - may indicate stratification or measurement issues",
                "recommendation": "Verify measurement system accuracy and check if data is being "
                "manipulated or averaged incorrectly",
                "default_severity": "low",
            },
            "rule_8": {
                "interpretation": "Eight consecutive points with none within 1-sigma "
                "of the center line (either side)",
                "meaning": "Process variation may be higher than expected",
                "recommendation": "Review process capability and consider if control limits need recalculation",
                "default_severity": "low",
            },
        }

        # Add interpretations to violations
        result = violations_df.copy()

        # Ensure severity column exists
        if "severity" not in result.columns:
            result["severity"] = result["rule"].map(
                lambda r: rule_interpretations.get(r, {}).get(
                    "default_severity", "medium"
                )
            )

        # Add interpretation columns
        result["interpretation"] = result["rule"].map(
            lambda r: rule_interpretations.get(r, {}).get(
                "interpretation", "Unknown rule"
            )
        )
        result["meaning"] = result["rule"].map(
            lambda r: rule_interpretations.get(r, {}).get(
                "meaning", "No description available"
            )
        )
        result["recommendation"] = result["rule"].map(
            lambda r: rule_interpretations.get(r, {}).get(
                "recommendation", "Review process documentation"
            )
        )

        return result

    if TYPE_CHECKING:

        @deprecated(
            "StatisticalProcessControlEvents.process() is deprecated, use .detect_violations()"
        )
        def process(
            self, selected_rules: list[str] | None = None, include_severity: bool = True
        ) -> pd.DataFrame: ...

    else:
        process = deprecated_method("detect_violations")

    @keyword_only(1)
    def detect_violations(
        self, selected_rules: list[str] | None = None, include_severity: bool = True
    ) -> pd.DataFrame:
        """
        Applies the selected SPC rules and generates a DataFrame of events where any rules are violated.

        Args:
            selected_rules (Optional[List[str]]): List of rule names (e.g., ['rule_1', 'rule_3']) to apply.
            include_severity (bool): Include the rule that flagged each
                point (``triggered_rule``) and its ``severity`` (default
                ``True``). ``False`` returns only the point columns.

        Returns:
            pd.DataFrame: One row per violating point (a point that violates
                several rules is reported once, with the first selected rule
                that flagged it). Columns: [systime, value_column, uuid,
                triggered_rule, severity, source_uuid]; without
                ``triggered_rule`` and ``severity`` when
                ``include_severity=False``. ``source_uuid`` is the actual
                signal's uuid. An empty result has the same columns.
        """
        df = self.dataframe[self.dataframe["uuid"] == self.actual_uuid].copy()
        df["systime"] = pd.to_datetime(df["systime"])
        df = df.sort_values(by="systime")

        limits = self.control_limits()

        # Dictionary of rule functions
        rules = {
            "rule_1": lambda df: self.rule_1(df, limits),
            "rule_2": lambda df: self.rule_2(df, limits=limits),
            "rule_3": lambda df: self.rule_3(df),
            "rule_4": lambda df: self.rule_4(df),
            "rule_5": lambda df: self.rule_5(df, limits),
            "rule_6": lambda df: self.rule_6(df, limits),
            "rule_7": lambda df: self.rule_7(df, limits),
            "rule_8": lambda df: self.rule_8(df, limits),
        }

        # Severity mapping for rules
        rule_severity = {
            "rule_1": "critical",
            "rule_2": "medium",
            "rule_3": "medium",
            "rule_4": "medium",
            "rule_5": "high",
            "rule_6": "medium",
            "rule_7": "low",
            "rule_8": "low",
        }

        # If no specific rules are provided, use all rules
        if selected_rules is None:
            selected_rules = list(rules.keys())

        # Apply selected rules and track which rule triggered each event
        all_events = []
        for rule_name in selected_rules:
            if rule_name in rules:
                rule_events = rules[rule_name](df)
                if not rule_events.empty:
                    rule_events["triggered_rule"] = rule_name
                    rule_events["severity"] = rule_severity.get(rule_name, "medium")
                    all_events.append(rule_events)

        if all_events:
            events = pd.concat(all_events).drop_duplicates(
                subset=["systime", self.value_column]
            )
        else:
            events = pd.DataFrame(
                columns=["systime", self.value_column, "triggered_rule", "severity"]
            )

        # Add the event UUID to the detected events
        events["uuid"] = self.event_uuid
        events["source_uuid"] = self.actual_uuid

        if include_severity:
            return events[
                [
                    "systime",
                    self.value_column,
                    "uuid",
                    "triggered_rule",
                    "severity",
                    "source_uuid",
                ]
            ].drop_duplicates()
        return events[
            ["systime", self.value_column, "uuid", "source_uuid"]
        ].drop_duplicates()


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated(
        "StatisticalProcessControlRuleBased is deprecated, use StatisticalProcessControlEvents"
    )
    class StatisticalProcessControlRuleBased(StatisticalProcessControlEvents): ...

else:
    StatisticalProcessControlRuleBased = deprecated_class(
        StatisticalProcessControlEvents, "StatisticalProcessControlRuleBased"
    )
