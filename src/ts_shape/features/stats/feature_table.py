import logging
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ts_shape.utils.frames import as_datetime, require_columns, to_ns

from ._values import bool_values, numeric_values, text_values
from .boolean_stats import BooleanStatistics
from .numeric_stats import NumericStatistics
from .string_stats import StringStatistics

logger = logging.getLogger(__name__)

#: Long-format value columns and the statistics computed for them. They are
#: classified by name, so every signal gets the same keys whatever its values.
_VALUE_KINDS = {
    "value_bool": "boolean",
    "value_double": "numeric",
    "value_integer": "numeric",
    "value_string": "string",
}

#: Columns that are not described per column.
_SKIPPED = {"uuid", "systime", "is_delta"}


def _kind(series: pd.Series) -> str | None:
    """Statistics kind of a column that is not a long-format value column."""
    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_datetime64_any_dtype(series.dtype):
        return None
    if is_numeric_dtype(series.dtype):
        return "numeric"
    if is_object_dtype(series.dtype) or is_string_dtype(series.dtype):
        return "string"
    return None


#: Features of ``compute_windowed`` every signal gets: the ``overall`` keys of
#: ``compute`` plus ``total_time_seconds`` (``total_time`` as float seconds).
_OVERALL_FEATURES: tuple[str, ...] = (
    "total_rows",
    "total_time",
    "total_time_seconds",
    "is_delta_sum",
    "is_delta_avg",
    "is_delta_std",
)

#: Features of ``compute_windowed`` per statistics kind: the keys of the
#: ``numeric_stats`` / ``boolean_stats`` / ``string_stats`` dicts of
#: ``compute``, in the same order.
_KIND_FEATURES: dict[str, tuple[str, ...]] = {
    "numeric": (
        "min",
        "max",
        "mean",
        "median",
        "std",
        "var",
        "sum",
        "kurtosis",
        "skewness",
        "q1",
        "q3",
        "iqr",
        "range",
        "mad",
        "coeff_var",
        "sem",
        "mode",
        "percentile_90",
        "percentile_10",
    ),
    "boolean": (
        "true_count",
        "false_count",
        "true_percentage",
        "false_percentage",
        "mode",
        "is_balanced",
    ),
    "string": (
        "unique_values",
        "most_frequent",
        "count_most_frequent",
        "count_null",
        "average_string_length",
        "longest_string",
        "shortest_string",
        "uppercase_percentage",
        "lowercase_percentage",
        "contains_digit_count",
        "least_common",
        "frequency_least_common",
    ),
}

#: Every feature name ``compute_windowed(metrics=...)`` accepts.
_WINDOWED_FEATURES: tuple[str, ...] = tuple(
    dict.fromkeys(
        _OVERALL_FEATURES + tuple(f for fs in _KIND_FEATURES.values() for f in fs)
    )
)

#: Counts: 0 rather than NaN in a window where the signal has no row.
_COUNT_FEATURES = frozenset(
    {
        "total_rows",
        "true_count",
        "false_count",
        "unique_values",
        "count_most_frequent",
        "count_null",
        "contains_digit_count",
        "frequency_least_common",
    }
)

#: Value column of a signal that fills several of them equally often.
_VALUE_PRIORITY = ("value_double", "value_integer", "value_string", "value_bool")

_WHERE_WINDOWED = "DescriptiveFeatures.compute_windowed"

_Array: TypeAlias = npt.NDArray[Any]


def _zero_out_fperr(values: _Array, tolerance: _Array) -> _Array:
    """``values`` with rounding noise below ``tolerance`` set to 0 (as pandas)."""
    return np.where(np.abs(values) < tolerance, 0.0, values)


def _numeric_features(
    x: _Array, codes: _Array, k: int, wanted: set[str]
) -> dict[str, _Array]:
    """Numeric statistics of ``x`` per group code (``0 .. k-1``).

    Every statistic matches the ``NumericStatistics`` method ``compute`` uses
    (pandas' bias-corrected skewness and kurtosis, linear quantiles, the
    smallest value as the mode on a tie).
    """
    out: dict[str, _Array] = {}
    grp = pd.Series(x).groupby(codes, sort=True)

    def series(values: pd.Series) -> _Array:
        return np.asarray(values.reindex(range(k)).to_numpy(), dtype="float64")

    valid = ~np.isnan(x)
    vcodes = codes[valid]
    count = np.bincount(vcodes, minlength=k).astype("float64")
    mean = series(grp.mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        if wanted & {"min", "range"}:
            out["min"] = series(grp.min())
        if wanted & {"max", "range"}:
            out["max"] = series(grp.max())
        if "range" in wanted:
            out["range"] = out["max"] - out["min"]
        out["mean"] = mean
        if "median" in wanted:
            out["median"] = series(grp.median())
        std = series(grp.std())
        out["std"] = std
        if "var" in wanted:
            out["var"] = series(grp.var())
        if "sum" in wanted:
            out["sum"] = series(grp.sum())
        if wanted & {"q1", "q3", "iqr", "percentile_90", "percentile_10"}:
            # Group-major (group, quantile) order; every code 0 .. k-1 occurs.
            q = grp.quantile([0.1, 0.25, 0.75, 0.9]).to_numpy(dtype="float64")
            q = q.reshape(k, 4)
            out["percentile_10"] = q[:, 0]
            out["q1"] = q[:, 1]
            out["q3"] = q[:, 2]
            out["percentile_90"] = q[:, 3]
            out["iqr"] = out["q3"] - out["q1"]
        if wanted & {"mad", "skewness", "kurtosis"}:
            dev = x[valid] - mean[vcodes]
            out["mad"] = np.bincount(vcodes, np.abs(dev), minlength=k) / count
            if wanted & {"skewness", "kurtosis"}:
                dev2 = dev**2
                m2 = np.bincount(vcodes, dev2, minlength=k)
                max_abs = series(pd.Series(np.abs(x)).groupby(codes).max())
                max_abs = np.nan_to_num(max_abs)
                eps = np.finfo("float64").eps
                m2 = _zero_out_fperr(m2, (eps * max_abs) ** 2 * count)
                n = count
                m3 = np.bincount(vcodes, dev2 * dev, minlength=k)
                m3 = _zero_out_fperr(m3, (eps * max_abs) ** 3 * count)
                skew = (n * (n - 1) ** 0.5 / (n - 2)) * (m3 / m2**1.5)
                skew = np.where(m2 == 0, 0.0, skew)
                out["skewness"] = np.where(n < 3, np.nan, skew)
                m4 = np.bincount(vcodes, dev2**2, minlength=k)
                m4 = _zero_out_fperr(m4, (eps * max_abs) ** 4 * count)
                adj = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
                numerator = _zero_out_fperr(
                    n * (n + 1) * (n - 1) * m4, (eps * max_abs) ** 4 * count
                )
                denominator = (n - 2) * (n - 3) * m2**2
                kurt = np.where(denominator == 0, 0.0, numerator / denominator - adj)
                out["kurtosis"] = np.where(n < 4, np.nan, kurt)
        if "sem" in wanted:
            out["sem"] = std / np.sqrt(count)
        if "coeff_var" in wanted:
            out["coeff_var"] = np.where(mean == 0, np.nan, std / mean)
    if "mode" in wanted:
        mode = np.full(k, np.nan)
        if valid.any():
            pairs = pd.DataFrame({"g": vcodes, "v": x[valid]})
            sizes = pairs.groupby(["g", "v"], sort=False).size().reset_index()
            sizes.columns = ["g", "v", "n"]
            sizes = sizes.sort_values(
                ["g", "n", "v"], ascending=[True, False, True], kind="stable"
            ).drop_duplicates("g")
            mode[sizes["g"].to_numpy()] = sizes["v"].to_numpy(dtype="float64")
        out["mode"] = mode
    return out


def _boolean_features(values: pd.Series, codes: _Array, k: int) -> dict[str, _Array]:
    """Boolean statistics of ``values`` per group code (``0 .. k-1``).

    ``mode`` and ``is_balanced`` are 1.0 / 0.0 (NaN for a mode of no value),
    so the columns stay numeric.
    """
    true = np.bincount(codes, values.eq(True).to_numpy(dtype="float64"), minlength=k)
    false = np.bincount(codes, values.eq(False).to_numpy(dtype="float64"), minlength=k)
    known = np.bincount(codes, values.notna().to_numpy(dtype="float64"), minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        true_pct = np.where(known > 0, true / known * 100, np.nan)
        false_pct = np.where(known > 0, false / known * 100, np.nan)
    return {
        "true_count": true,
        "false_count": false,
        "true_percentage": true_pct,
        "false_percentage": false_pct,
        # The mode is False on a tie, as Series.mode sorts its result.
        "mode": np.where(known > 0, (true > false).astype("float64"), np.nan),
        "is_balanced": ((known > 0) & (true * 2 == known)).astype("float64"),
    }


def _string_features(
    values: pd.Series, codes: _Array, k: int, wanted: set[str]
) -> dict[str, _Array]:
    """String statistics of ``values`` per group code (``0 .. k-1``).

    Matches ``StringStatistics.summary_as_dict``: the most frequent value is
    the smallest one on a tie, the least common one the first seen of the
    least frequent values, the longest / shortest string the first one of
    that length.
    """
    out: dict[str, _Array] = {}
    known = values.notna().to_numpy()
    kcodes = codes[known]
    text = values[known].reset_index(drop=True)
    count = np.bincount(kcodes, minlength=k).astype("float64")
    out["count_null"] = np.bincount(codes[~known], minlength=k).astype("float64")

    def blank(fill: Any = np.nan, dtype: Any = "float64") -> _Array:
        return np.full(k, fill, dtype=dtype)

    if wanted & {
        "unique_values",
        "most_frequent",
        "count_most_frequent",
        "least_common",
        "frequency_least_common",
    }:
        pairs = pd.DataFrame({"g": kcodes, "v": text, "pos": np.arange(len(text))})
        sizes = pairs.groupby(["g", "v"], sort=False).agg(
            n=("pos", "size"), first=("pos", "min")
        )
        sizes = sizes.reset_index()
        unique = np.bincount(sizes["g"].to_numpy(), minlength=k).astype("float64")
        out["unique_values"] = unique
        top = sizes.sort_values(
            ["g", "n", "v"], ascending=[True, False, True], kind="stable"
        ).drop_duplicates("g")
        most = blank(None, object)
        most[top["g"].to_numpy()] = top["v"].to_numpy()
        out["most_frequent"] = most
        top_n = blank(0.0)
        top_n[top["g"].to_numpy()] = top["n"].to_numpy()
        out["count_most_frequent"] = top_n
        low = sizes.sort_values(["g", "n", "first"], kind="stable").drop_duplicates("g")
        least = blank(None, object)
        least[low["g"].to_numpy()] = low["v"].to_numpy()
        out["least_common"] = least
        low_n = blank(0.0)
        low_n[low["g"].to_numpy()] = low["n"].to_numpy()
        out["frequency_least_common"] = low_n
    if wanted & {"average_string_length", "longest_string", "shortest_string"}:
        lengths = text.str.len()
        sized = lengths.notna().to_numpy()
        lcodes = kcodes[sized]
        lvals = lengths[sized].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out["average_string_length"] = np.bincount(
                lcodes, lvals, minlength=k
            ) / np.bincount(lcodes, minlength=k)
        by_length = pd.Series(lvals, index=np.flatnonzero(sized)).groupby(lcodes)
        for name, pick in (
            ("longest_string", by_length.idxmax()),
            ("shortest_string", by_length.idxmin()),
        ):
            column = blank(None, object)
            column[pick.index.to_numpy()] = text.to_numpy()[pick.to_numpy()]
            out[name] = column
    as_text = text.astype(str)
    with np.errstate(invalid="ignore", divide="ignore"):
        if "uppercase_percentage" in wanted:
            upper = as_text.str.isupper().to_numpy(dtype="float64")
            out["uppercase_percentage"] = (
                np.bincount(kcodes, upper, minlength=k) / count * 100
            )
        if "lowercase_percentage" in wanted:
            lower = as_text.str.islower().to_numpy(dtype="float64")
            out["lowercase_percentage"] = (
                np.bincount(kcodes, lower, minlength=k) / count * 100
            )
    if "contains_digit_count" in wanted:
        digit = text.str.contains(r"\d").eq(True).to_numpy(dtype="float64")
        out["contains_digit_count"] = np.bincount(kcodes, digit, minlength=k)
    return out


class DescriptiveFeatures:
    """Compute descriptive statistics for a DataFrame, grouped by UUID.

    Use `compute` to get descriptive statistics for each UUID in the DataFrame.

    Attributes:
        data (pandas.DataFrame): DataFrame containing the data.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """Initialize with the DataFrame to describe.

        Args:
            dataframe: DataFrame containing the data.
        """
        self.data = dataframe

    def overall_stats(self, group: pd.DataFrame) -> dict[str, int | float]:
        """Compute and return overall statistics for the DataFrame group.

        The returned dictionary contains:

        - **total_rows**: Total number of rows in the group.
        - **total_time**: Total time difference from max and min of 'systime' column.
        - **is_delta_sum**: Sum of the 'is_delta' column.
        - **is_delta_avg**: Mean of the 'is_delta' column.
        - **is_delta_std**: Standard deviation of the 'is_delta' column.

        The ``is_delta_*`` values are NaN when the group has no ``is_delta``
        column (or no value in it).

        Args:
            group: DataFrame group (rows of a single UUID) to summarize.

        Returns:
            dict: A dictionary with overall statistics.
        """
        if "systime" in group.columns and len(group):
            times = as_datetime(group["systime"], where="DescriptiveFeatures")
            total_time = times.max() - times.min()
        else:
            total_time = pd.NaT
        if "is_delta" in group.columns:
            delta = bool_values(group, "is_delta", where="DescriptiveFeatures")
            flags = delta.dropna().astype(float)
        else:
            flags = pd.Series(dtype="float64")
        statistics = {
            "total_rows": len(group),
            "total_time": total_time,
            "is_delta_sum": float(flags.sum()) if len(flags) else np.nan,
            "is_delta_avg": float(flags.mean()) if len(flags) else np.nan,
            "is_delta_std": float(flags.std()) if len(flags) > 1 else np.nan,
        }
        return statistics

    def compute_per_group(self, group: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Compute and return statistics for each column in the DataFrame group.

        Args:
            group: DataFrame group (rows of a single UUID) to summarize.

        Returns:
            dict: A dictionary with overall statistics, and string, numeric, and
                boolean statistics per column. The long-format value columns
                are classified by name (``value_bool`` boolean,
                ``value_double`` / ``value_integer`` numeric, ``value_string``
                string), so every signal has the same keys; other columns by
                dtype. ``systime``, ``uuid`` and ``is_delta`` are not described
                per column.
        """
        results: dict[str, dict[str, Any]] = {"overall": self.overall_stats(group)}
        for col in group.columns:
            if col in _SKIPPED:
                continue
            kind = _VALUE_KINDS.get(str(col)) or _kind(group[col])
            if kind == "boolean":
                values = bool_values(group, col, where="DescriptiveFeatures")
                results[col] = {
                    "boolean_stats": BooleanStatistics.summary_as_dict(
                        values.to_frame(col), col
                    )
                }
            elif kind == "numeric":
                numbers = numeric_values(group, col, where="DescriptiveFeatures")
                results[col] = {
                    "numeric_stats": NumericStatistics.summary_as_dict(
                        numbers.to_frame(col), col
                    )
                }
            elif kind == "string":
                results[col] = {
                    "string_stats": StringStatistics.summary_as_dict(group, col)
                }

        return results

    def compute(
        self,
        output_format: str = "dict",
        *,
        window: str | pd.Timedelta | pd.DateOffset | None = None,
    ) -> pd.DataFrame | dict[str, dict[str, dict[str, Any]]]:
        """Compute and return descriptive statistics for each UUID in the DataFrame.

        Args:
            output_format (str, optional): The desired output format: 'dict',
                'dataframe' or 'windowed' (one row per time window, see
                `compute_windowed`). Defaults to 'dict'.
            window: Window size for ``output_format="windowed"`` (default
                ``"1s"``); not allowed with the other formats.

        Returns:
            Union[DataFrame, dict]: A DataFrame or a nested dictionary with the
                UUID as the key and specific statistics related to that UUID's
                data type. An empty frame gives ``{}`` (or an empty DataFrame).
                With ``output_format="windowed"`` the frame of
                `compute_windowed`.

        Raises:
            ValueError: For an unknown ``output_format``, or a ``window`` with
                a format other than 'windowed'.
            ColumnNotFoundError: If the frame has no ``uuid`` column.
        """
        if output_format not in ("dict", "dataframe", "windowed"):
            raise ValueError(
                "Invalid output format. Choose 'dict', 'dataframe' or 'windowed'."
            )
        if output_format == "windowed":
            return self.compute_windowed("1s" if window is None else window)
        if window is not None:
            raise ValueError(
                "window applies only to output_format='windowed', "
                f"not {output_format!r}."
            )
        require_columns(self.data, ["uuid"], where="DescriptiveFeatures.compute")
        if output_format == "dataframe":
            rows_list = []

            # Iterate through each group of UUID
            for uuid, group in self.data.groupby("uuid"):
                stats_per_group = self.compute_per_group(group)

                # Iterate through the nested stats and create flat columns
                row_dict = {}
                for section, stats in stats_per_group.items():
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    column_name = f"{uuid}::{section}::{key}::{sub_key}"
                                    row_dict[column_name] = sub_value
                            else:
                                column_name = f"{uuid}::{section}::{key}"
                                row_dict[column_name] = value
                    else:
                        column_name = f"{uuid}::{section}"
                        row_dict[column_name] = stats

                rows_list.append(row_dict)

            return pd.DataFrame(rows_list)

        return {
            uuid: self.compute_per_group(group)
            for uuid, group in self.data.groupby("uuid", sort=True)
        }

    def compute_windowed(
        self,
        window: str | pd.Timedelta | pd.DateOffset = "1s",
        *,
        metrics: list[str] | None = None,
        column_separator: str = "__",
        time_column: str = "systime",
    ) -> pd.DataFrame:
        """Descriptive statistics per time window, all signals in one row.

        One row per window of ``window`` on a regular grid, from the window
        of the first timestamp to the window of the last one: every window is
        present, also one without any data. The windows are the bins of
        ``pd.Grouper(freq=window)``, wall-clock windows in the data's time
        zone (a ``"1D"`` window on a DST day lasts 23 or 25 hours).

        Each signal gets the statistics `compute` reports for it, under the
        same names: the ``overall`` ones (``total_rows``, ``total_time``,
        ``is_delta_sum``, ``is_delta_avg``, ``is_delta_std``, plus
        ``total_time_seconds``) and those of its value kind, read from the
        long-format value column the signal fills (``value_double`` /
        ``value_integer`` numeric, ``value_bool`` boolean, ``value_string``
        string; the most filled one when it fills several). A cell holds
        what `compute` gives for that signal on the rows of that window.

        A window where a signal has no row gives NaN, except for the counts
        (``total_rows``, ``true_count``, ``false_count``, ``unique_values``,
        ``count_most_frequent``, ``count_null``, ``contains_digit_count``,
        ``frequency_least_common``), which are 0 (int64). The boolean
        ``mode`` and ``is_balanced`` are 1.0 / 0.0, ``total_time`` is a
        Timedelta, the string values (``most_frequent``, ``longest_string``,
        ...) are text; every other column is float64.

        All statistics are grouped aggregations over (signal, window), with
        no loop over windows.

        Args:
            window: Window size, a pandas offset (``"1s"``, ``"15min"``,
                ``"1h"``, ``"1D"``, ``"MS"``) or a ``Timedelta``.
            metrics: Feature names to compute; None computes all. A name that
                does not apply to a signal's kind is skipped for that signal.
            column_separator: Separator between uuid and feature in the
                column names.
            time_column: Timestamp column. Rows without a timestamp are left
                out.

        Returns:
            DataFrame with ``start``, ``end``, ``duration_seconds`` and one
                column ``{uuid}{column_separator}{feature}`` per signal and
                feature, signals sorted by uuid, features in the order of
                `compute`. ``start`` / ``end`` are tz-aware when the input
                is. Only the window columns when the frame has no timestamp.

        Raises:
            ValueError: For a name in ``metrics`` that is no feature.
            ColumnNotFoundError: If the frame has no ``uuid`` or
                ``time_column`` column.

        Example:
            ```python
            import pandas as pd
            from ts_shape.features.stats.feature_table import DescriptiveFeatures

            df = pd.DataFrame({
                "systime": pd.to_datetime(
                    ["2024-01-01 00:00:00.2", "2024-01-01 00:00:00.7",
                     "2024-01-01 00:00:02.5"]
                ),
                "uuid": "temp",
                "value_double": [20.0, 22.0, 21.0],
            })
            out = DescriptiveFeatures(df).compute_windowed(
                "1s", metrics=["total_rows", "mean"]
            )
            print(out[["start", "temp__total_rows", "temp__mean"]])
            ```
        """
        from ts_shape.events._windows import window_bounds

        if metrics is not None:
            unknown = [m for m in metrics if m not in _WINDOWED_FEATURES]
            if unknown:
                raise ValueError(
                    f"{_WHERE_WINDOWED}: unknown metrics {unknown}. "
                    f"Available: {list(_WINDOWED_FEATURES)}"
                )
        wanted = set(_WINDOWED_FEATURES if metrics is None else metrics)
        require_columns(self.data, ["uuid", time_column], where=_WHERE_WINDOWED)

        times = as_datetime(self.data[time_column], where=_WHERE_WINDOWED)
        timed = times.notna().to_numpy()
        data = self.data.loc[timed].reset_index(drop=True)
        times = times[timed].reset_index(drop=True)
        if data.empty:
            empty_times = pd.Series(dtype=times.dtype)
            return pd.DataFrame(
                {
                    "start": empty_times,
                    "end": empty_times.copy(),
                    "duration_seconds": pd.Series(dtype="float64"),
                }
            )

        # The grid: the bins pd.Grouper would make between the first and the
        # last timestamp, empty ones included.
        span = pd.Series(0, index=pd.DatetimeIndex([times.min(), times.max()]))
        labels = pd.Series(span.resample(window).size().index)
        start, end = window_bounds(labels, window)
        n_win = len(start)
        win = np.searchsorted(to_ns(start), to_ns(times), side="right") - 1
        win = np.clip(win, 0, n_win - 1)

        uuid_codes, uuids = pd.factorize(data["uuid"], sort=True)
        gid = uuid_codes.astype("int64") * n_win + win

        # Output columns per uuid index, filled window by window from the
        # per-group results.
        columns: dict[int, dict[str, _Array]] = {u: {} for u in range(len(uuids))}

        def scatter(labels_: _Array, features: dict[str, _Array]) -> None:
            owner = labels_ // n_win
            slot = labels_ % n_win
            bounds = np.searchsorted(owner, np.arange(len(uuids) + 1))
            for u in range(len(uuids)):
                lo, hi = bounds[u], bounds[u + 1]
                if lo == hi:
                    continue
                for name, values in features.items():
                    if name not in wanted:
                        continue
                    if values.dtype == object:
                        full = np.full(n_win, np.nan, dtype=object)
                    elif name in _COUNT_FEATURES:
                        full = np.zeros(n_win, dtype="int64")
                    elif values.dtype.kind == "m":
                        full = np.full(
                            n_win, np.timedelta64("NaT", "ns"), dtype=values.dtype
                        )
                    else:
                        full = np.full(n_win, np.nan)
                    full[slot[lo:hi]] = values[lo:hi]
                    columns[u][name] = full

        # Overall statistics, over every row of a signal.
        codes, group_labels = pd.factorize(gid, sort=True)
        k = len(group_labels)
        overall: dict[str, _Array] = {"total_rows": np.bincount(codes, minlength=k)}
        tns = pd.Series(to_ns(times)).groupby(codes, sort=True)
        span_ns = (tns.max() - tns.min()).to_numpy(dtype="int64")
        overall["total_time"] = span_ns.astype("timedelta64[ns]")
        overall["total_time_seconds"] = span_ns / 1e9
        if "is_delta" in data.columns:
            flags = pd.Series(
                bool_values(data, "is_delta", where=_WHERE_WINDOWED).astype("float64")
            ).groupby(codes, sort=True)
            overall["is_delta_sum"] = flags.sum(min_count=1).to_numpy("float64")
            overall["is_delta_avg"] = flags.mean().to_numpy("float64")
            overall["is_delta_std"] = flags.std().to_numpy("float64")
        else:
            for name in ("is_delta_sum", "is_delta_avg", "is_delta_std"):
                overall[name] = np.full(k, np.nan)
        scatter(np.asarray(group_labels), overall)

        # The value column (and so the kind) of each signal.
        value_columns = [c for c in _VALUE_PRIORITY if c in data.columns]
        signal_column: dict[int, str] = {}
        if value_columns:
            filled = data[value_columns].notna().groupby(uuid_codes).sum()
            for u, row in filled.iterrows():
                if row.max() > 0:
                    signal_column[int(u)] = str(row.idxmax())

        for column in value_columns:
            kind = _VALUE_KINDS[column]
            if not wanted & set(_KIND_FEATURES[kind]):
                continue
            mine = [u for u, c in signal_column.items() if c == column]
            if not mine:
                continue
            rows = np.isin(uuid_codes, mine)
            subset = data.loc[rows, [column]].reset_index(drop=True)
            codes, group_labels = pd.factorize(gid[rows], sort=True)
            k = len(group_labels)
            if kind == "numeric":
                x = numeric_values(subset, column, where=_WHERE_WINDOWED)
                features = _numeric_features(
                    x.to_numpy(dtype="float64"), codes, k, wanted
                )
            elif kind == "boolean":
                values = bool_values(subset, column, where=_WHERE_WINDOWED)
                features = _boolean_features(values, codes, k)
            else:
                values = text_values(subset, column, where=_WHERE_WINDOWED)
                features = _string_features(values, codes, k, wanted)
            scatter(np.asarray(group_labels), features)

        frame = pd.DataFrame(
            {
                "start": start.reset_index(drop=True),
                "end": end.reset_index(drop=True),
                "duration_seconds": (end - start)
                .dt.total_seconds()
                .astype("float64")
                .reset_index(drop=True),
            }
        )
        pieces = [frame]
        for u, uuid in enumerate(uuids):
            signal_kind = _VALUE_KINDS.get(signal_column.get(u, ""))
            order = _OVERALL_FEATURES + (
                _KIND_FEATURES[signal_kind] if signal_kind else ()
            )
            block = {
                f"{uuid}{column_separator}{name}": columns[u][name]
                for name in order
                if name in columns[u]
            }
            pieces.append(pd.DataFrame(block))
        out = pd.concat(pieces, axis=1)
        logger.info(
            f"Windowed features: {len(out)} windows x {len(out.columns)} columns."
        )
        return out
