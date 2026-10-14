"""Time-weighted resampling for irregular and change-encoded signals.

``series.resample(window).mean()`` averages *samples*, which is only right
when samples are evenly spaced. On a change-encoded signal (one row per
change) or irregular data the result is driven by how many rows land in a
window, not by how long each value was held: ``On@09:00, Off@09:59`` reads
as 50 % on instead of ~98 %, and a window that holds a value without any
new sample has no mean at all.

:func:`time_weighted_resample` holds each value until the next sample
(zero-order hold) and averages by time inside each window.
"""

import numpy as np
import pandas as pd  # type: ignore

__all__ = ["time_weighted_resample"]


def time_weighted_resample(
    series: pd.Series,
    window: str,
    *,
    data_end: pd.Timestamp | None = None,
    max_gap: str | pd.Timedelta | None = None,
) -> pd.Series:
    """Time-weighted mean of a held signal per window.

    Args:
        series: Values with a ``DatetimeIndex`` (naive or tz-aware). NaN
            values are ignored; for duplicate timestamps the last value wins.
        window: Resample frequency (e.g. ``'1h'``).
        data_end: Hold the last value until this time (e.g. the end of a
            companion signal). Default: the series ends at its last sample.
        max_gap: If set, a value is not held across a gap to the next sample
            longer than this; that span counts as "no data" instead.

    Returns:
        A float Series indexed by window start. Windows without any held
        value are NaN.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return pd.Series(dtype=float)

    end = s.index[-1]
    if data_end is not None and pd.Timestamp(data_end) > end:
        end = pd.Timestamp(data_end)

    bounds = s.index.append(pd.DatetimeIndex([end]))
    window_starts = pd.Series(0, index=bounds).resample(window).count().index
    knots = s.index.union(window_starts).union(pd.DatetimeIndex([end]))
    knots = knots[(knots >= s.index[0]) & (knots <= end)]
    if len(knots) < 2:
        return pd.Series(np.nan, index=window_starts, dtype=float)

    held = s.reindex(knots, method="ffill").to_numpy(dtype=float)[:-1]
    seg_start = knots[:-1]
    seconds = (knots[1:] - knots[:-1]).total_seconds().to_numpy(dtype=float)

    if max_gap is not None:
        sample_times = pd.Series(s.index, index=s.index)
        next_sample = sample_times.shift(-1).fillna(pd.Timestamp(end))
        span = (next_sample - sample_times).reindex(knots, method="ffill")
        valid = (span <= pd.Timedelta(max_gap)).to_numpy()[:-1]
        seconds = np.where(valid, seconds, 0.0)

    weight = pd.Series(seconds, index=seg_start).resample(window).sum()
    weighted = pd.Series(held * seconds, index=seg_start).resample(window).sum()
    out = (weighted / weight).where(weight > 0)
    return out.reindex(window_starts).astype(float)
