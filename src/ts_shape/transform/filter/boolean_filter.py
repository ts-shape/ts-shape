import logging

import numpy as np
import pandas as pd

from ts_shape.transform.filter._sequence import scatter, time_order
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)


class IsDeltaFilter(Base):
    """
    Provides class methods for filtering is_delta columns in a pandas DataFrame.
    """

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_is_delta_true(
        cls, dataframe: pd.DataFrame, value_column: str = "is_delta"
    ) -> pd.DataFrame:
        """Filters rows where 'is_delta' is True.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"is_delta"``.
        """
        Base._validate_column(dataframe, value_column)
        # ``eq`` keeps missing values out of both filters.
        return dataframe[dataframe[value_column].eq(True)]

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_is_delta_false(
        cls, dataframe: pd.DataFrame, value_column: str = "is_delta"
    ) -> pd.DataFrame:
        """Filters rows where 'is_delta' is False.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"is_delta"``.
        """
        Base._validate_column(dataframe, value_column)
        return dataframe[dataframe[value_column].eq(False)]


def _edges(
    dataframe: pd.DataFrame,
    value_column: str,
    *,
    group_by_uuid: bool,
    time_column: str,
    uuid_column: str,
    rising: bool,
    where: str,
) -> pd.DataFrame:
    """Rows where the state changes False -> True (``rising``) or back.

    Each row is compared with the previous known state of the same signal in
    time order; a missing value holds the previous state.
    """
    Base._validate_column(dataframe, value_column)
    values = dataframe[value_column]
    is_true = values.eq(True).fillna(False).to_numpy(dtype=bool)
    is_false = values.eq(False).fillna(False).to_numpy(dtype=bool)
    state = np.where(is_true, 1.0, np.where(is_false, 0.0, np.nan))
    order, groups = time_order(
        dataframe,
        time_column=time_column,
        uuid_column=uuid_column,
        group_by_uuid=group_by_uuid,
        where=where,
    )
    current = pd.Series(state[order])
    held = current.groupby(groups).ffill()
    previous = held.groupby(groups).shift().to_numpy()
    now = current.to_numpy()
    if rising:
        edge = (previous == 0.0) & (now == 1.0)
    else:
        edge = (previous == 1.0) & (now == 0.0)
    return dataframe[scatter(edge, order)]


class BooleanFilter(Base):
    """
    Provides class methods for filtering boolean columns in a pandas DataFrame,
    particularly focusing on status changes.
    """

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_falling_value_bool(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_bool",
        *,
        group_by_uuid: bool = True,
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> pd.DataFrame:
        """Filters rows where 'value_bool' changes from True to False.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"value_bool"``.
            group_by_uuid: When ``True`` (default) each row is compared with
                the previous row of the same ``uuid`` (when the frame has a
                ``uuid`` column), so a multi-signal frame produces no edge
                between the last row of one uuid and the first of the next.
                ``False`` compares with the previous row of the frame.
            time_column: Rows are compared in the order of this column (when
                the frame has it). Default ``"systime"``.
            uuid_column: Signal column used by ``group_by_uuid``. Default
                ``"uuid"``.

        A missing value holds the previous state, so ``False, None, True``
        has its edge at the ``True`` row.

        Returns:
            The input rows where the edge occurs, in input order.
        """
        return _edges(
            dataframe,
            value_column,
            group_by_uuid=group_by_uuid,
            time_column=time_column,
            uuid_column=uuid_column,
            rising=False,
            where="BooleanFilter.filter_falling_value_bool",
        )

    @classmethod
    @renamed_kwargs(column_name="value_column")
    def filter_raising_value_bool(
        cls,
        dataframe: pd.DataFrame,
        value_column: str = "value_bool",
        *,
        group_by_uuid: bool = True,
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> pd.DataFrame:
        """Filters rows where 'value_bool' changes from False to True.

        Args:
            dataframe: Input DataFrame.
            value_column: Name of the column to filter on. Default ``"value_bool"``.
            group_by_uuid: When ``True`` (default) each row is compared with
                the previous row of the same ``uuid`` (when the frame has a
                ``uuid`` column), so a multi-signal frame produces no edge
                between the last row of one uuid and the first of the next.
                ``False`` compares with the previous row of the frame.
            time_column: Rows are compared in the order of this column (when
                the frame has it). Default ``"systime"``.
            uuid_column: Signal column used by ``group_by_uuid``. Default
                ``"uuid"``.

        A missing value holds the previous state, so ``False, None, True``
        has its edge at the ``True`` row.

        Returns:
            The input rows where the edge occurs, in input order.
        """
        return _edges(
            dataframe,
            value_column,
            group_by_uuid=group_by_uuid,
            time_column=time_column,
            uuid_column=uuid_column,
            rising=True,
            where="BooleanFilter.filter_raising_value_bool",
        )
