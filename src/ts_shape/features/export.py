"""Feature matrix export utilities for ts-shape.

Converts long-format timeseries DataFrames into wide feature matrices
suitable for machine-learning workflows.
"""

import logging
from typing import Dict, List, Optional, Callable, Union
import pandas as pd  # type: ignore
import numpy as np

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

    _DEFAULT_AGGS: Dict[str, Callable] = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }

    @classmethod
    def to_feature_matrix(
        cls,
        df: pd.DataFrame,
        uuid_col: str = "uuid",
        value_cols: Optional[List[str]] = None,
        agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Pivot a long-format DataFrame into a wide feature matrix.

        Args:
            df: Long-format DataFrame containing signal data.
            uuid_col: Column whose unique values become column prefixes
                (e.g. ``'uuid'``).
            value_cols: Numeric columns to aggregate. If ``None``, all
                numeric columns except ``uuid_col`` are used.
            agg_funcs: Mapping of ``name -> callable`` or pandas agg string.
                Defaults to ``mean``, ``std``, ``min``, ``max``.
            group_col: Optional column to use as the row index of the output
                matrix (e.g. ``'cycle_uuid'``, ``'batch_id'``).  When
                ``None`` the entire DataFrame is treated as one group and a
                single-row matrix is returned.

        Returns:
            Wide-format DataFrame.  Columns are named
            ``{uuid}__{value_col}__{agg}``.  Row index is ``group_col``
            when provided, otherwise 0.

        Raises:
            ValueError: If ``uuid_col`` is not in ``df``.
        """
        if uuid_col not in df.columns:
            raise ValueError(
                f"uuid_col '{uuid_col}' not found. Available columns: {list(df.columns)}"
            )

        if value_cols is None:
            value_cols = [
                c for c in df.select_dtypes(include="number").columns if c != uuid_col
            ]
        if not value_cols:
            raise ValueError("No numeric value columns found to aggregate.")

        aggs = agg_funcs if agg_funcs is not None else cls._DEFAULT_AGGS

        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise ValueError(f"value_cols not found in DataFrame: {missing}")

        if group_col is not None and group_col not in df.columns:
            raise ValueError(
                f"group_col '{group_col}' not found. Available columns: {list(df.columns)}"
            )

        rows: Dict = {}
        if group_col:
            iterator = (
                ((rk, sig), grp) for (rk, sig), grp in df.groupby([group_col, uuid_col])
            )
        else:
            iterator = (((0, sig), grp) for sig, grp in df.groupby(uuid_col))

        for (row_key, sig), grp in iterator:
            for vcol in value_cols:
                series = grp[vcol].dropna()
                for agg_name, agg_fn in aggs.items():
                    col_name = f"{sig}__{vcol}__{agg_name}"
                    value = agg_fn(series) if len(series) else float("nan")
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
