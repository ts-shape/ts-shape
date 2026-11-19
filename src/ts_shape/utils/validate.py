"""Checks for configuration arguments (dicts, limits, names, choices).

Detector configurations are plain dicts and numbers: schedules, recipe specs,
line configurations, targets, cost tables, threshold levels. A typo in a key,
limits given the wrong way round or a shift name that does not exist used to
be computed as given and produced a plausible but wrong result. These helpers
make such input fail loudly, with a message that says what is wrong and what
is accepted:

* :func:`check_keys` -- unknown keys raise ``ValueError`` with a did-you-mean;
* :func:`check_order` -- ``lower > upper`` raises ``ValueError``;
* :func:`check_levels_ordered` -- threshold levels in the wrong order raise;
* :func:`check_choice` -- a value outside its choices raises;
* :func:`check_known_names` -- names that do not occur in the data (shifts,
  phases, parts, uuids) raise or emit a
  :class:`~ts_shape.errors.DataQualityWarning`;
* :func:`check_positive` / :func:`check_non_negative`.

Every message starts with ``where`` (e.g. ``"ScrapEvents.scrap_cost"``).
"""

from __future__ import annotations

import difflib
import math
import warnings
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.deprecation import caller_stacklevel

__all__ = [
    "check_choice",
    "check_keys",
    "check_known_names",
    "check_levels_ordered",
    "check_non_negative",
    "check_order",
    "check_positive",
    "suggest",
]


def suggest(name: Any, known: Iterable[Any]) -> str:
    """`` (did you mean 'x'?)`` for the closest of ``known``, or ``""``."""
    names = [str(k) for k in known]
    close = difflib.get_close_matches(str(name), names, n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _prefix(where: str | None) -> str:
    return f"{where}: " if where else ""


def check_keys(
    mapping: Mapping[str, Any] | None,
    allowed: Collection[str],
    *,
    where: str | None = None,
    what: str = "key",
) -> None:
    """Raise ``ValueError`` when ``mapping`` has a key outside ``allowed``.

    Args:
        mapping: The configuration dict (``None`` is accepted: nothing to check).
        allowed: The keys the configuration understands.
        where: Message prefix (the calling method).
        what: What a key is, for the message (``"criterion"``, ``"metric"``).

    Raises:
        ValueError: Naming the unknown key, the closest allowed one and every
            allowed key.
    """
    if not mapping:
        return
    unknown = [k for k in mapping if k not in allowed]
    if not unknown:
        return
    first = unknown[0]
    raise ValueError(
        f"{_prefix(where)}unknown {what} {first!r}{suggest(first, allowed)}"
        + (f" (and {unknown[1:]!r})" if len(unknown) > 1 else "")
        + f". Accepted: {sorted(allowed)}"
    )


def check_order(
    lower: Any,
    upper: Any,
    *,
    names: tuple[str, str] = ("lower", "upper"),
    where: str | None = None,
    strict: bool = False,
) -> None:
    """Raise ``ValueError`` when ``lower > upper`` (or ``>=`` when ``strict``).

    ``None`` or NaN on either side is not checked (an open bound).
    """
    if lower is None or upper is None:
        return
    try:
        if math.isnan(float(lower)) or math.isnan(float(upper)):
            return
    except (TypeError, ValueError):
        pass
    bad = lower >= upper if strict else lower > upper
    if bad:
        relation = "below" if strict else "at most"
        raise ValueError(
            f"{_prefix(where)}{names[0]}={lower!r} must be {relation} "
            f"{names[1]}={upper!r}; are they swapped?"
        )


def check_levels_ordered(
    levels: Sequence[Any],
    *,
    direction: str = "above",
    where: str | None = None,
    what: str = "levels",
) -> None:
    """Raise ``ValueError`` for empty ``levels`` or ones out of order.

    ``direction="above"`` needs increasing levels (a higher level is more
    severe), ``"below"`` decreasing ones.
    """
    if len(levels) == 0:
        raise ValueError(f"{_prefix(where)}{what} is empty")
    values = list(levels)
    pairs = list(zip(values, values[1:], strict=False))
    if direction == "above":
        bad = [(a, b) for a, b in pairs if not a < b]
        order = "increasing"
    else:
        bad = [(a, b) for a, b in pairs if not a > b]
        order = "decreasing"
    if bad:
        raise ValueError(
            f"{_prefix(where)}{what} must be strictly {order} for "
            f"direction={direction!r}; got {values!r}"
        )


def check_choice(
    value: Any,
    choices: Collection[Any],
    *,
    name: str = "value",
    where: str | None = None,
) -> None:
    """Raise ``ValueError`` when ``value`` is not one of ``choices``."""
    if value in choices:
        return
    raise ValueError(
        f"{_prefix(where)}{name}={value!r} is not accepted"
        f"{suggest(value, choices)}. Accepted: {sorted(map(str, choices))}"
    )


def check_known_names(
    names: Iterable[Any],
    known: Collection[Any],
    *,
    kind: str,
    where: str | None = None,
    error: bool = False,
    consequence: str = "",
) -> list[Any]:
    """Report the ``names`` that are not in ``known``.

    Args:
        names: Names the configuration uses (shift names, phases, parts...).
        known: The names that exist (in the data or the library).
        kind: What a name is, for the message (``"shift"``).
        where: Message prefix.
        error: Raise ``ValueError`` instead of warning.
        consequence: What happens to an unknown name (appended to the warning).

    Returns:
        The unknown names (empty when every name is known).

    Raises:
        ValueError: With ``error=True``, when a name is unknown.

    Warns:
        DataQualityWarning: Without ``error``, when a name is unknown.
    """
    known_text = {str(k) for k in known}
    missing = [n for n in dict.fromkeys(names) if str(n) not in known_text]
    if not missing:
        return []
    shown = sorted(known_text)[:12]
    more = f" (+{len(known_text) - 12} more)" if len(known_text) > 12 else ""
    hint = suggest(missing[0], known_text)
    message = (
        f"{_prefix(where)}{kind} {missing!r} not found{hint}; known: " f"{shown}{more}"
    )
    if error:
        raise ValueError(message)
    warnings.warn(
        message + (f"; {consequence}" if consequence else ""),
        DataQualityWarning,
        stacklevel=caller_stacklevel(),
    )
    return missing


def check_positive(value: Any, *, name: str, where: str | None = None) -> None:
    """Raise ``ValueError`` unless ``value > 0``."""
    if value is None or not value > 0:
        raise ValueError(f"{_prefix(where)}{name} must be > 0, got {value!r}")


def check_non_negative(value: Any, *, name: str, where: str | None = None) -> None:
    """Raise ``ValueError`` unless ``value >= 0``."""
    if value is None or not value >= 0:
        raise ValueError(f"{_prefix(where)}{name} must be >= 0, got {value!r}")
