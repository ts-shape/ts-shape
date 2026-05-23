"""Engineering unit conversion for ts-shape, backed by the ``pint`` library.

:class:`UnitConverter` is a thin wrapper around
`pint <https://pint.readthedocs.io>`_, the de-facto-standard, actively
maintained open-source Python units library. pint owns all conversion data,
dimensional analysis, affine temperature handling and SI prefixes -- ts-shape
stores no conversion factors of its own.

pint is an optional dependency; install it with::

    pip install ts-shape[units]
"""

import logging

import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

# Optional dependency: mirror the scikit-learn handling in
# events/quality/outlier_detection.py -- import once, expose a flag.
try:
    import pint  # type: ignore

    PINT_AVAILABLE = True
    _UREG = pint.UnitRegistry()
except ImportError:  # pragma: no cover - exercised only without the extra
    PINT_AVAILABLE = False
    _UREG = None

_INSTALL_HINT = (
    "Unit conversion requires the 'pint' library. "
    "Install it with: pip install ts-shape[units]"
)

# Input-shorthand normalisation ONLY -- this is not conversion data. pint reads
# "C" as coulomb and "F" as farad, so the everyday temperature shorthands are
# mapped to pint's unambiguous names. Every actual conversion stays pint's job.
_ALIASES = {
    "C": "degC",
    "F": "degF",
    "K": "kelvin",
    "R": "degR",
    "°C": "degC",
    "°F": "degF",
    "°K": "kelvin",
    "°R": "degR",
}


def _require_pint() -> None:
    if not PINT_AVAILABLE:
        raise ImportError(_INSTALL_HINT)


def _normalize(unit: str) -> str:
    """Map a temperature shorthand to pint's name; pass everything else through."""
    return _ALIASES.get(unit.strip(), unit.strip())


class UnitConverter(Base):
    """Convert engineering units in scalars and DataFrame columns via ``pint``.

    Unit names are whatever ``pint`` understands (e.g. ``"bar"``, ``"psi"``,
    ``"m^3/hour"``, ``"kWh"``, ``"degC"``). The temperature shorthands ``C``,
    ``F``, ``K`` and ``R`` are accepted as conveniences.

    Example usage::

        UnitConverter.convert_value(100, "C", "F")          # 212.0
        UnitConverter.conversion_factor("bar", "psi")        # (14.5037..., 0.0)
        UnitConverter.convert_column(df, "bar", "psi", column_name="value_double")
    """

    @classmethod
    def convert_value(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert a single scalar value from ``from_unit`` to ``to_unit``.

        Args:
            value: The numeric value to convert.
            from_unit: Source unit (any pint unit name).
            to_unit: Target unit (any pint unit name).

        Returns:
            The converted value as a float.

        Raises:
            ImportError: If the ``pint`` library is not installed.
            ValueError: If a unit is unknown or the units are incompatible.
        """
        _require_pint()
        try:
            quantity = _UREG.Quantity(value, _normalize(from_unit))
            return float(quantity.to(_normalize(to_unit)).magnitude)
        except pint.PintError as exc:
            raise ValueError(
                f"Cannot convert '{from_unit}' to '{to_unit}': {exc}"
            ) from exc

    @classmethod
    def conversion_factor(cls, from_unit: str, to_unit: str) -> tuple[float, float]:
        """Return the ``(scale, offset)`` mapping ``from_unit`` -> ``to_unit``.

        The conversion is ``target = value * scale + offset``. ``offset`` is
        ``0.0`` for pure-ratio units and non-zero for affine units such as
        temperature. The pair is derived automatically from ``pint``.

        Args:
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            ``(scale, offset)`` as floats.

        Raises:
            ImportError: If the ``pint`` library is not installed.
            ValueError: If a unit is unknown or the units are incompatible.
        """
        _require_pint()
        zero = cls.convert_value(0.0, from_unit, to_unit)
        one = cls.convert_value(1.0, from_unit, to_unit)
        return (one - zero, zero)

    @classmethod
    def convert_column(
        cls,
        dataframe: pd.DataFrame,
        from_unit: str,
        to_unit: str,
        *,
        column_name: str = "value_double",
        target_column: str | None = None,
    ) -> pd.DataFrame:
        """Convert a numeric DataFrame column to different units.

        Args:
            dataframe: The DataFrame to operate on.
            from_unit: Source unit of the column values.
            to_unit: Target unit.
            column_name: Column holding the values to convert.
            target_column: Column to write the converted values to. Defaults to
                ``column_name`` (in-place on a copy).

        Returns:
            A copy of the DataFrame with the converted column.

        Raises:
            ImportError: If the ``pint`` library is not installed.
            ColumnNotFoundError: If ``column_name`` is missing.
            ValueError: If a unit is unknown or the units are incompatible.
        """
        _require_pint()
        Base._validate_column(dataframe, column_name)
        df = dataframe.copy()
        target = target_column or column_name
        try:
            quantity = _UREG.Quantity(
                df[column_name].to_numpy(dtype=float), _normalize(from_unit)
            )
            df[target] = quantity.to(_normalize(to_unit)).magnitude
        except pint.PintError as exc:
            raise ValueError(
                f"Cannot convert '{from_unit}' to '{to_unit}': {exc}"
            ) from exc
        return df
