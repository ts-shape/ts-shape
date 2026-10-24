"""Recognise misread object-id values (PLC no-reads, placeholders, fillers).

A scanner or PLC that fails to read an id still writes *something* into the
id tag: ``"NOREAD"``, ``"#N/A"``, ``"????????"``, the integer default ``0``
or ``65535``, a run of ``X`` or ``0``, or the first few characters of the id.
Taken at face value each of these becomes an object of its own and splits the
presence of the real id around it. :func:`is_misread` flags such readings so
id consumers (object detection, object identification) can treat them as
"no reading".

The rules (any one makes a value a misread):

* **blank** -- empty or whitespace only (``""``, ``"  "``, ``"\\x00"``);
* **no letter or digit** -- ``"????????"``, ``"--------"``, ``"#"``;
* **PLC default number** -- ``0``, ``-1``, ``32767``, ``-32768``, ``65535``,
  ``2147483647``, ``4294967295``, as a number or as text, leading zeros
  ignored (``"00000000"`` is ``0``);
* **placeholder token**, case-insensitive and ignoring blanks and
  punctuation -- ``NOREAD`` (``"NO READ"``, ``"no_read"``), ``ERR``,
  ``ERROR``, ``#N/A`` / ``N/A`` / ``NA``, ``NULL``, ``NONE``, ``NAN``;
* **filler** -- text (not a number column) of at least
  :data:`FILLER_MIN_LENGTH` characters that repeats one character,
  case-insensitive (``"XXXXXXXX"``, ``"99999999"``); an integer id such as
  ``1111`` is not filler;
* **truncated read** (``truncations=True``; text only, in order) -- a value
  held at most :data:`TRUNCATION_MAX_SAMPLES` consecutive samples that is a
  strict prefix (case-insensitive) of the id read just before it *and* just
  after it (``SN-0042``, ``SN-0``, ``SN-0042``), or of one neighbour when it
  is also shorter than the length shared by at least
  :data:`MODAL_LENGTH_SHARE` of the signal's distinct values (fixed-format
  ids).

Missing values (``NaN`` / ``None`` / ``<NA>``) are *not* flagged -- they are
already "no reading". This module is a superset of the context-free rules of
:mod:`ts_shape.signals.identify` (no letter or digit, PLC default numbers).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "FILLER_MIN_LENGTH",
    "MODAL_LENGTH_SHARE",
    "PLACEHOLDER_TOKENS",
    "PLC_DEFAULTS",
    "TRUNCATION_MAX_SAMPLES",
    "is_misread",
    "is_misread_value",
]

_BOOL: TypeAlias = "npt.NDArray[np.bool_]"
_I64: TypeAlias = "npt.NDArray[np.int64]"

#: PLC / data-type default numbers written when nothing was read.
PLC_DEFAULTS: frozenset[str] = frozenset(
    {"0", "-1", "32767", "-32768", "65535", "2147483647", "4294967295"}
)
#: Placeholder words (upper case, letters and digits only).
PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {"NOREAD", "ERR", "ERROR", "NA", "NULL", "NONE", "NAN"}
)
#: A one-character repeat this long or longer is filler.
FILLER_MIN_LENGTH = 4
#: A truncated read lasts at most this many consecutive samples.
TRUNCATION_MAX_SAMPLES = 2
#: Share of distinct values with one length that makes the format fixed.
MODAL_LENGTH_SHARE = 0.8

_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")
_INTEGER = re.compile(r"[+-]?[0-9]+")
_BLANK = " \t\r\n\x00"


def _text(value: object) -> str:
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _misread_text(text: str, is_str: bool) -> bool:
    """The context-free rules on a stripped, upper-cased reading."""
    if not text:
        return True
    if text.isalnum():  # the common case: a plain id
        if text.isdigit() and str(int(text)) in PLC_DEFAULTS:
            return True
        return text in PLACEHOLDER_TOKENS or (
            is_str and len(text) >= FILLER_MIN_LENGTH and len(set(text)) == 1
        )
    if not any(ch.isalnum() for ch in text):
        return True
    if _INTEGER.fullmatch(text) and str(int(text)) in PLC_DEFAULTS:
        return True
    if _NON_ALNUM.sub("", text) in PLACEHOLDER_TOKENS:
        return True
    return is_str and len(text) >= FILLER_MIN_LENGTH and len(set(text)) == 1


def _missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


def is_misread_value(value: object) -> bool:
    """Whether one value is a misread (context-free rules only).

    Args:
        value: A raw id reading (text or number).

    Returns:
        ``True`` for a blank, letterless/digitless, PLC default, placeholder
        or filler value; ``False`` for a missing value, a boolean and every
        other id.
    """
    if _missing(value) or isinstance(value, bool | np.bool_):
        return False
    return _misread_text(_text(value).strip(_BLANK).upper(), isinstance(value, str))


def is_misread(values: Iterable[object], *, truncations: bool = True) -> _BOOL:
    """Flag misread id readings.

    Args:
        values: The readings of one id signal, **in time order** when
            ``truncations`` is on (a Series, array or list).
        truncations: Also flag truncated reads, which needs the neighbouring
            readings (see the module docstring).

    Returns:
        A boolean array, one entry per value; ``True`` marks a misread.

    Example:
        >>> is_misread(["B1", "NOREAD", "", "????", 0, "B2", None]).tolist()
        [False, True, True, True, True, False, False]
    """
    s = values if isinstance(values, pd.Series) else pd.Series(list(values))
    s = s.reset_index(drop=True)
    n = len(s)
    if n == 0:
        return np.zeros(0, dtype=bool)
    codes, uniques = pd.factorize(s, use_na_sentinel=True)
    codes = np.asarray(codes, dtype=np.int64)
    uvals = list(uniques)
    is_text = np.array([isinstance(u, str) for u in uvals], dtype=bool)
    skip = [isinstance(u, bool | np.bool_) for u in uvals]
    texts = [_text(u).strip(_BLANK).upper() for u in uvals]
    bad_u = np.array(
        [
            not sk and _misread_text(t, bool(st))
            for t, st, sk in zip(texts, is_text, skip, strict=True)
        ],
        dtype=bool,
    )
    out = np.zeros(n, dtype=bool)
    ok = codes >= 0
    out[ok] = bad_u[codes[ok]]
    if not truncations or not is_text.any():
        return out
    # runs of equal readings; a truncation is a short run of text whose
    # neighbouring runs are good ids it is a strict prefix of
    brk = np.ones(n, dtype=bool)
    brk[1:] = codes[1:] != codes[:-1]
    idx = np.flatnonzero(brk)
    rc = codes[idx]
    rlen = np.diff(np.append(idx, n))
    good = (rc >= 0) & ~bad_u[rc.clip(min=0)]
    good_text = good & is_text[rc.clip(min=0)]
    lengths = pd.Series(
        [len(texts[i]) for i in np.flatnonzero(is_text & ~bad_u)], dtype="int64"
    )
    modal_len = -1
    if len(lengths):
        counts = lengths.value_counts()
        if counts.iloc[0] >= MODAL_LENGTH_SHARE * len(lengths):
            modal_len = int(counts.index[0])
    cand = np.flatnonzero(good_text & (rlen <= TRUNCATION_MAX_SAMPLES))
    if len(cand) == 0:
        return out
    prev = np.full(len(cand), -1, dtype=np.int64)
    nxt = np.full(len(cand), -1, dtype=np.int64)
    has_prev = cand > 0
    pc = cand[has_prev] - 1
    prev[has_prev] = np.where(good_text[pc], rc[pc], -1)
    has_next = cand < len(rc) - 1
    nc = cand[has_next] + 1
    nxt[has_next] = np.where(good_text[nc], rc[nc], -1)

    def prefix_of(other: _I64) -> _BOOL:
        return np.array(
            [
                o >= 0
                and len(texts[o]) > len(texts[c])
                and texts[o].startswith(texts[c])
                for o, c in zip(other, rc[cand], strict=True)
            ],
            dtype=bool,
        )

    p_pre, n_pre = prefix_of(prev), prefix_of(nxt)
    x_len = np.array([len(texts[c]) for c in rc[cand]], dtype=np.int64)
    sandwiched = (prev >= 0) & (prev == nxt) & p_pre
    short = (x_len < modal_len) & (p_pre | n_pre)
    flagged = np.zeros(len(rc), dtype=bool)
    flagged[cand] = sandwiched | short
    if flagged.any():
        out |= flagged[np.cumsum(brk) - 1]
    return out


def _integral_float(value: object) -> bool:
    return (
        isinstance(value, float | np.floating)
        and bool(np.isfinite(value))
        and float(value).is_integer()
    )


def id_strings(values: pd.Series) -> pd.Series:
    """Object ids as text (``string`` dtype, missing values stay ``<NA>``).

    Like ``values.astype("string")``, except that an integral float renders
    without ``.0`` (``1001.0`` -> ``"1001"``), as the oids of
    :func:`ts_shape.eventlog.object_intervals` do. Keeps the index.
    """
    out = values.astype("string")
    if values.dtype.kind == "f":
        arr = values.to_numpy(dtype=np.float64, na_value=np.nan)
        integral = np.isfinite(arr) & (arr == np.trunc(arr))
    elif values.dtype == object:
        if pd.api.types.infer_dtype(values, skipna=True) in ("string", "empty"):
            return out
        arr = values.to_numpy(dtype=object)
        integral = np.fromiter(
            (_integral_float(v) for v in arr), dtype=bool, count=len(arr)
        )
    else:
        return out
    if integral.any():
        pos = np.flatnonzero(integral)
        out.iloc[pos] = [str(int(v)) for v in arr[pos].tolist()]
    return out
