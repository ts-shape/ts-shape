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

from collections.abc import Sequence

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

_VALUE_COLUMNS = ("value_double", "value_integer", "value_bool")


def make_timeseries(
    uuids: Sequence[str] = ("sensor:signal",),
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
        uuids: Signal identifiers; one block of ``n_points`` rows per uuid.
        n_points: Number of samples generated per uuid.
        freq: Pandas offset alias for the sampling interval (e.g. ``"30s"``).
        start: Timestamp of the first sample.
        baseline: Mean level of the generated signal.
        noise: Standard deviation of the Gaussian noise added to the signal.
        drift: Total linear drift applied across the series (tool-wear style).
        n_outliers: Number of large spike outliers injected per uuid.
        value_column: Which value column to populate -- one of
            ``value_double``, ``value_integer`` or ``value_bool``.
        seed: Seed for reproducibility; pass ``None`` for fresh randomness.

    Returns:
        DataFrame with columns ``systime``, ``uuid``, ``value_bool``,
        ``value_integer``, ``value_double``, ``value_string`` and ``is_delta``.

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

    rng = np.random.default_rng(seed)
    frames = []
    for uuid in uuids:
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
                "value_bool": pd.NA,
                "value_integer": pd.NA,
                "value_double": np.nan,
                "value_string": pd.NA,
                "is_delta": True,
            }
        )
        if value_column == "value_double":
            frame["value_double"] = np.round(signal, 6)
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
    """
    if value_column not in ("value_string", "value_integer"):
        raise ValueError("value_column must be 'value_string' or 'value_integer'")
    if hold <= 0:
        raise ValueError(f"hold must be positive; got {hold}")

    n = len(values) * hold
    times = pd.date_range(start=start, periods=n, freq=freq)
    seq = [v for v in values for _ in range(hold)]
    frame = pd.DataFrame(
        {
            "systime": times,
            "uuid": uuid,
            "value_bool": pd.NA,
            "value_integer": pd.NA,
            "value_double": np.nan,
            "value_string": pd.NA,
            "is_delta": False,
        }
    )
    if value_column == "value_integer":
        frame["value_integer"] = pd.array(seq, dtype="Int64")
    else:
        frame["value_string"] = pd.array([str(v) for v in seq], dtype="string")
    if source_uuid is not None:
        frame["source_uuid"] = source_uuid
    return frame
