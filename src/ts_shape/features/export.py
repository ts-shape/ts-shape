"""Feature matrix export utilities for ts-shape.

Converts long-format timeseries DataFrames into wide feature matrices
suitable for machine-learning workflows.
"""

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import caller_stacklevel, renamed_kwargs
from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)


_TEXT_VALUE_COLUMNS = ("value_string", "value_bool")


def _holds_numbers(name: object, column: pd.Series) -> bool:
    """A numeric, non-boolean column, or an ``object`` column of numbers."""
    if ptypes.is_bool_dtype(column.dtype):
        return False
    if ptypes.is_numeric_dtype(column.dtype):
        return True
    if name in _TEXT_VALUE_COLUMNS or not ptypes.is_object_dtype(column.dtype):
        return False
    known = column.dropna()
    if known.empty or known.map(lambda v: isinstance(v, (bool, np.bool_))).any():
        return False
    return bool(pd.to_numeric(known, errors="coerce").notna().any())


class FeatureMatrixExporter:
    """Convert long-format timeseries DataFrames to wide feature matrices.

    Pivots a multiplexed DataFrame (one row per timestamp per signal) into
    a single row per entity (e.g. cycle, segment, batch) with one column per
    ``{uuid}__{value_column}__{agg}`` combination.  The result is ready for
    use with scikit-learn, XGBoost, or any tabular ML framework.

    Usage::

        from ts_shape.features.export import FeatureMatrixExporter

        # Long-format df: columns systime, uuid, value_double
        matrix = FeatureMatrixExporter.to_feature_matrix(df, uuid_column='uuid', value_columns=['value_double'])
    """

    #: pandas reductions: ``std`` is the sample standard deviation (ddof=1).
    _DEFAULT_AGGS: dict[str, str | Callable] = {
        "mean": "mean",
        "std": "std",
        "min": "min",
        "max": "max",
    }

    @classmethod
    @renamed_kwargs(
        df="dataframe",
        uuid_col="uuid_column",
        value_cols="value_columns",
        group_col="group_column",
    )
    def to_feature_matrix(
        cls,
        dataframe: pd.DataFrame,
        uuid_column: str = "uuid",
        value_columns: list[str] | None = None,
        agg_funcs: dict[str, str | Callable] | None = None,
        group_column: str | None = None,
    ) -> pd.DataFrame:
        """Pivot a long-format DataFrame into a wide feature matrix.

        Args:
            dataframe: Long-format DataFrame containing signal data.
            uuid_column: Column whose unique values become column prefixes
                (e.g. ``'uuid'``).
            value_columns: Numeric columns to aggregate. If ``None``, the numeric
                ``value_*`` columns (``value_double``, ``value_integer``) are
                used; flags such as ``is_delta`` are not features. A
                ``value_*`` column (other than ``value_string`` /
                ``value_bool``) stored as ``object`` that holds numbers counts
                as numeric; its values are coerced with
                ``pd.to_numeric(errors="coerce")``.
            agg_funcs: Mapping of ``name -> callable`` or pandas agg string.
                Defaults to ``mean``, ``std`` (sample standard deviation,
                ``ddof=1``), ``min``, ``max``.
            group_column: Optional column to use as the row index of the output
                matrix (e.g. ``'cycle_uuid'``, ``'batch_id'``).  When
                ``None`` the entire DataFrame is treated as one group and a
                single-row matrix is returned.

        Returns:
            Wide-format DataFrame.  Columns are named
            ``{uuid}__{value_column}__{agg}``.  Row index is ``group_column``
            when provided, otherwise 0.

        Raises:
            ColumnNotFoundError: If ``uuid_column``, a ``value_columns`` entry or
                ``group_column`` is not in ``dataframe``.
            ValueError: If no numeric value column is found.
        """
        where = "FeatureMatrixExporter.to_feature_matrix"
        require_columns(dataframe, [uuid_column, group_column], where=where)

        if value_columns is None:
            value_columns = [
                c
                for c in dataframe.columns
                if str(c).startswith("value_")
                and c != uuid_column
                and _holds_numbers(c, dataframe[c])
            ]
        if not value_columns:
            raise ValueError("No numeric value columns found to aggregate.")
        require_columns(dataframe, value_columns, where=where)

        aggs = agg_funcs if agg_funcs is not None else cls._DEFAULT_AGGS

        rows: dict = {}
        if group_column:
            iterator = (
                ((rk, sig), grp)
                for (rk, sig), grp in dataframe.groupby([group_column, uuid_column])
            )
        else:
            iterator = (((0, sig), grp) for sig, grp in dataframe.groupby(uuid_column))

        for (row_key, sig), grp in iterator:
            for vcol in value_columns:
                series = to_numeric_checked(
                    grp[vcol], what=f"column {vcol!r} of signal {sig!r}"
                ).dropna()
                for agg_name, agg_fn in aggs.items():
                    col_name = f"{sig}__{vcol}__{agg_name}"
                    if not len(series):
                        value = float("nan")
                    elif isinstance(agg_fn, str):
                        value = series.agg(agg_fn)
                    else:
                        value = agg_fn(series)
                    rows.setdefault(row_key, {})[col_name] = value

        if not rows:
            warnings.warn(
                "FeatureMatrixExporter.to_feature_matrix: no data to aggregate; "
                "returning an empty feature matrix.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )
            return pd.DataFrame()

        matrix = pd.DataFrame.from_dict(rows, orient="index")
        matrix.index.name = group_column
        matrix = matrix.sort_index(axis=1)

        logger.info(
            f"Feature matrix built: {matrix.shape[0]} rows × {matrix.shape[1]} columns "
            f"({len(dataframe[uuid_column].unique())} signals, {len(value_columns)} value cols, {len(aggs)} aggs)."
        )
        return matrix
