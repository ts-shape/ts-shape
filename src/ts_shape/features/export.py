"""Feature matrix export utilities for ts-shape.

Converts long-format timeseries DataFrames into wide feature matrices
suitable for machine-learning workflows.
"""

import logging
from collections.abc import Callable

import pandas as pd
from pandas.api import types as ptypes

from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)


class FeatureMatrixExporter:
    """Convert long-format timeseries DataFrames to wide feature matrices.

    Pivots a multiplexed DataFrame (one row per timestamp per signal) into
    a single row per entity (e.g. cycle, segment, batch) with one column per
    ``{uuid}__{value_col}__{agg}`` combination.  The result is ready for
    use with scikit-learn, XGBoost, or any tabular ML framework.

    Usage::

        from ts_shape.features.export import FeatureMatrixExporter

        # Long-format df: columns systime, uuid, value_double
        matrix = FeatureMatrixExporter.to_feature_matrix(df, uuid_col='uuid', value_cols=['value_double'])
    """

    #: pandas reductions: ``std`` is the sample standard deviation (ddof=1).
    _DEFAULT_AGGS: dict[str, str | Callable] = {
        "mean": "mean",
        "std": "std",
        "min": "min",
        "max": "max",
    }

    @classmethod
    def to_feature_matrix(
        cls,
        df: pd.DataFrame,
        uuid_col: str = "uuid",
        value_cols: list[str] | None = None,
        agg_funcs: dict[str, str | Callable] | None = None,
        group_col: str | None = None,
    ) -> pd.DataFrame:
        """Pivot a long-format DataFrame into a wide feature matrix.

        Args:
            df: Long-format DataFrame containing signal data.
            uuid_col: Column whose unique values become column prefixes
                (e.g. ``'uuid'``).
            value_cols: Numeric columns to aggregate. If ``None``, the numeric
                ``value_*`` columns (``value_double``, ``value_integer``) are
                used; flags such as ``is_delta`` are not features.
            agg_funcs: Mapping of ``name -> callable`` or pandas agg string.
                Defaults to ``mean``, ``std`` (sample standard deviation,
                ``ddof=1``), ``min``, ``max``.
            group_col: Optional column to use as the row index of the output
                matrix (e.g. ``'cycle_uuid'``, ``'batch_id'``).  When
                ``None`` the entire DataFrame is treated as one group and a
                single-row matrix is returned.

        Returns:
            Wide-format DataFrame.  Columns are named
            ``{uuid}__{value_col}__{agg}``.  Row index is ``group_col``
            when provided, otherwise 0.

        Raises:
            ColumnNotFoundError: If ``uuid_col``, a ``value_cols`` entry or
                ``group_col`` is not in ``df``.
            ValueError: If no numeric value column is found.
        """
        where = "FeatureMatrixExporter.to_feature_matrix"
        require_columns(df, [uuid_col, group_col], where=where)

        if value_cols is None:
            value_cols = [
                c
                for c in df.columns
                if str(c).startswith("value_")
                and c != uuid_col
                and ptypes.is_numeric_dtype(df[c].dtype)
                and not ptypes.is_bool_dtype(df[c].dtype)
            ]
        if not value_cols:
            raise ValueError("No numeric value columns found to aggregate.")
        require_columns(df, value_cols, where=where)

        aggs = agg_funcs if agg_funcs is not None else cls._DEFAULT_AGGS

        rows: dict = {}
        if group_col:
            iterator = (
                ((rk, sig), grp) for (rk, sig), grp in df.groupby([group_col, uuid_col])
            )
        else:
            iterator = (((0, sig), grp) for sig, grp in df.groupby(uuid_col))

        for (row_key, sig), grp in iterator:
            for vcol in value_cols:
                series = pd.to_numeric(grp[vcol], errors="coerce").dropna()
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
            logger.warning("No data to aggregate — returning empty feature matrix.")
            return pd.DataFrame()

        matrix = pd.DataFrame.from_dict(rows, orient="index")
        matrix.index.name = group_col
        matrix = matrix.sort_index(axis=1)

        logger.info(
            f"Feature matrix built: {matrix.shape[0]} rows × {matrix.shape[1]} columns "
            f"({len(df[uuid_col].unique())} signals, {len(value_cols)} value cols, {len(aggs)} aggs)."
        )
        return matrix
