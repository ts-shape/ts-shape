"""Synthetic timeseries generators for trying out ts-shape.

These helpers produce DataFrames in the standard ts-shape schema
(``systime``, ``uuid``, ``value_*``, ``is_delta``) so any detector,
transform, or statistic can be exercised without real data or loaders::

    import ts_shape
    df = ts_shape.make_timeseries(["sensor:temp"], n_outliers=4)
    ts_shape.OutlierDetectionEvents(df, value_column="value_double") \\
        .detect_outliers_zscore()
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.loader._utils import empty_signal_frame
from ts_shape.utils.deprecation import caller_stacklevel

_VALUE_COLUMNS = ("value_double", "value_integer", "value_bool")


def _unused_columns(n: int) -> dict[str, pd.Series]:
    """The value columns a generator leaves empty, in nullable dtypes."""
    return {
        "value_bool": pd.Series(pd.array([pd.NA] * n, dtype="boolean")),
        "value_integer": pd.Series(pd.array([pd.NA] * n, dtype="Int64")),
        "value_double": pd.Series(np.full(n, np.nan)),
        "value_string": pd.Series([pd.NA] * n, dtype=object),
    }


def make_timeseries(
    uuids: str | Sequence[str] = ("sensor:signal",),
    *,
    n_points: int = 1000,
    freq: str = "30s",
    start: str = "2025-01-01 00:00:00",
    baseline: float = 100.0,
    noise: float = 1.0,
    drift: float = 0.0,
    n_outliers: int = 0,
    value_column: str = "value_double",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a standard-schema synthetic timeseries DataFrame.

    Args:
        uuids: Signal identifiers (or a single one as a string); one block of
            ``n_points`` rows per uuid. An empty list gives an empty frame with
            the standard columns.
        n_points: Number of samples generated per uuid.
        freq: Pandas offset alias for the sampling interval (e.g. ``"30s"``).
        start: Timestamp of the first sample.
        baseline: Mean level of the generated signal.
        noise: Standard deviation of the Gaussian noise added to the signal.
        drift: Total linear drift applied across the series (tool-wear style).
        n_outliers: Number of large spike outliers injected per uuid. Spikes
            are ``12 * noise`` high, so with ``noise=0`` none are visible (a
            :class:`~ts_shape.errors.DataQualityWarning` says so).
        value_column: Which value column to populate -- one of
            ``value_double``, ``value_integer`` or ``value_bool``.
        seed: Seed for reproducibility; pass ``None`` for fresh randomness.

    Returns:
        DataFrame with columns ``systime``, ``uuid``, ``value_bool``,
        ``value_integer``, ``value_double``, ``value_string`` and ``is_delta``.
        The populated column is ``float64`` / ``int64`` / ``bool``; the unused
        ones are missing, as ``boolean``, ``Int64``, ``float64`` and
        ``object``.

    Raises:
        ValueError: If ``value_column`` is not a supported value column, or
            ``n_points`` is not positive.
    """
    if value_column not in _VALUE_COLUMNS:
        raise ValueError(
            f"value_column must be one of {_VALUE_COLUMNS}; got {value_column!r}"
        )
    if n_points <= 0:
        raise ValueError(f"n_points must be positive; got {n_points}")

    uuid_list = [uuids] if isinstance(uuids, str) else list(uuids)
    if not uuid_list:
        return empty_signal_frame()
    if n_outliers > 0 and noise == 0:
        warnings.warn(
            "n_outliers has no effect with noise=0: outliers are 12 * noise high",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )

    rng = np.random.default_rng(seed)
    frames = []
    for uuid in uuid_list:
        times = pd.date_range(start=start, periods=n_points, freq=freq)
        signal = baseline + rng.normal(0.0, noise, n_points)
        if drift:
            signal = signal + np.linspace(0.0, drift, n_points)
        if n_outliers > 0:
            idx = rng.choice(n_points, size=min(n_outliers, n_points), replace=False)
            signal[idx] += rng.choice([-1.0, 1.0], size=len(idx)) * noise * 12.0

        frame = pd.DataFrame(
            {
                "systime": times,
                "uuid": uuid,
                **_unused_columns(n_points),
                "is_delta": True,
            }
        )
        if value_column == "value_double":
            frame["value_double"] = signal
        elif value_column == "value_integer":
            frame["value_integer"] = signal.round().astype("int64")
        else:  # value_bool
            frame["value_bool"] = signal > baseline
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def make_id_signal(
    uuid: str = "object:id",
    values: Sequence[str] = ("A", "B", "C"),
    *,
    hold: int = 10,
    freq: str = "30s",
    start: str = "2025-01-01 00:00:00",
    value_column: str = "value_string",
    source_uuid: str | None = None,
) -> pd.DataFrame:
    """Generate a categorical *identifier* signal in the standard schema.

    Each value in ``values`` is held for ``hold`` consecutive samples, so the
    signal looks like a real batch / serial / coil / recipe id stream that
    changes over time. Feed it to :mod:`ts_shape.eventlog.objects` to extract
    object instances.

    Args:
        uuid: Signal identifier.
        values: Ordered id values; each held for ``hold`` samples.
        hold: Samples each value persists before the next one starts.
        freq: Pandas offset alias for the sampling interval.
        start: Timestamp of the first sample.
        value_column: Which column carries the id (``value_string`` or
            ``value_integer``).
        source_uuid: Optional ``source_uuid`` to stamp on every row.

    Returns:
        DataFrame with the standard ts-shape columns, ``value_column`` filled.

    Raises:
        ValueError: For an unsupported ``value_column``, ``hold <= 0``, or
            ``value_column="value_integer"`` with ids that are not whole
            numbers.
    """
    if value_column not in ("value_string", "value_integer"):
        raise ValueError("value_column must be 'value_string' or 'value_integer'")
    if hold <= 0:
        raise ValueError(f"hold must be positive; got {hold}")
    if value_column == "value_integer":
        bad = [v for v in values if not _is_integral(v)]
        if bad:
            raise ValueError(
                f"value_column='value_integer' needs whole-number ids; got "
                f"{bad[:5]}. Use value_column='value_string' for text ids."
            )

    n = len(values) * hold
    times = pd.date_range(start=start, periods=n, freq=freq)
    seq = [v for v in values for _ in range(hold)]
    frame = pd.DataFrame(
        {
            "systime": times,
            "uuid": uuid,
            **_unused_columns(n),
            "is_delta": False,
        }
    )
    if value_column == "value_integer":
        frame["value_integer"] = pd.array([int(v) for v in seq], dtype="Int64")
    else:
        frame["value_string"] = pd.array([str(v) for v in seq], dtype="string")
    if source_uuid is not None:
        frame["source_uuid"] = source_uuid
    return frame


def make_object_signals(
    *,
    seed: int = 0,
    days: float = 3,
    include: Iterable[str] | None = None,
    noise: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a labelled benchmark of object-identifier signals.

    Simulates one or two production lines and the plant around them (work
    orders with setups, stops and cycles; batches inside the material; one
    serial per cycle with a companion part counter; carriers circulating in a
    loop; operators logging in at the 06:00 / 14:00 / 22:00 shift changes;
    orders, customers and suppliers stamped at shipping and goods receipt;
    SSCC-labelled boxes put away into storage bins) and samples every signal
    the way a historian stores it: on change, with keep-alive re-sends, or
    periodically, with gaps. Identifiers use real formats (SAP numbers,
    GTIN-13/14 and SSCC with check digits, GS1 application identifiers,
    EPC GRAI-96 hex, UUIDs, date-coded batches); several formats per object
    type are drawn per seed. Negative controls (states, alarm texts,
    measurements, counters, timestamps, constants, plus hard cases such as a
    1/2/3 mode code) are mixed in.

    The uuids are opaque (``"sig:0007"``) and shuffled, so only the values and
    their behaviour over time tell what a signal carries.

    Seeds 0-4 are the dev split; seeds 100-119 the held-out split
    (``ts_shape._object_signals.DEV_SEEDS`` / ``HELDOUT_SEEDS``). Seeds from
    100 on also draw formats that dev seeds never use, plus wider and
    differently shaped hold-time distributions; the ``variant`` label names
    the format.

    Args:
        seed: Seed of the whole simulation; the output is deterministic per
            seed.
        days: Length of the recording in days.
        include: Object types (``"batch"``), roles (``"state"``) or negative
            kinds (``"alarm_text"``) to keep; ``None`` keeps every signal.
        noise: Add realistic read noise: missing readings, empty strings,
            garbage reads (``"NOREAD"``, truncated ids, ``65535``), lower-case
            and stripped leading zeros, and extra per-signal gaps.

    Returns:
        ``(frame, labels)``. ``frame`` is the long-format frame with the
        standard columns. ``labels`` has one row per uuid with ``uuid``,
        ``role`` (identifier, state, free_text, measurement, counter,
        timestamp or constant), ``object_type`` (``None`` unless an
        identifier), ``value_column``, ``parent_uuid`` (the signal it nests
        in: batch in material, serial in batch; else ``None``) and
        ``variant`` (the format or negative kind).

    Raises:
        ValueError: If ``days`` is not positive or ``include`` names an
            unknown object type or role.

    Example:
        >>> import ts_shape
        >>> frame, labels = ts_shape.make_object_signals(seed=0, days=1)
        >>> sorted(labels["role"].unique())
        ['constant', 'counter', 'free_text', 'identifier', 'measurement',
         'state', 'timestamp']
        >>> batch = labels.loc[labels["object_type"] == "batch"].iloc[0]
        >>> labels.set_index("uuid").loc[batch["parent_uuid"], "object_type"]
        'material'
    """
    from ts_shape._object_signals import generate

    return generate(seed=seed, days=days, include=include, noise=noise)


def _is_integral(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | np.integer):
        return True
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number.is_integer()
