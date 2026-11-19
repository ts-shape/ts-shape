"""Signals that carry personal data (operator log-ins, names, badges).

A machine export can hold the operator who is logged in, a badge number or
the name on a setup sheet. Such signals must not reach a report. The
patterns here are matched, case-insensitively, against a signal's label (or
its uuid when it has none); German and English words are covered:

* worker, operator, personnel, employee, user, login / log-in, badge, name of
  a person;
* Werker, Bediener, Bedienpersonal, Mitarbeiter, Personal(-Nr), Benutzer,
  Anmeldung, Ausweis.

:func:`is_personal` checks one label; :data:`PERSONAL_DATA_PATTERNS` is the
default ``exclude`` of :func:`ts_shape.cycle_profile.prepare_signals` and
:func:`ts_shape.profile`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

__all__ = ["PERSONAL_DATA_PATTERNS", "is_personal", "personal_pattern"]

#: Regular expressions (case-insensitive) of signal labels with personal data.
PERSONAL_DATA_PATTERNS: tuple[str, ...] = (
    r"worker",
    r"operator",
    r"personnel",
    r"employee",
    r"\buser\b",
    r"log[-_ ]?in",
    r"badge",
    r"werker",
    r"bediener",
    r"bedienpersonal",
    r"mitarbeiter",
    r"personal[-_ ]?(?:nr|nummer|number|id)?\b",
    r"benutzer",
    r"anmeldung",
    r"ausweis",
)


def personal_pattern(
    patterns: Sequence[str] = PERSONAL_DATA_PATTERNS,
) -> re.Pattern[str]:
    """One case-insensitive regular expression matching any of ``patterns``."""
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


def is_personal(
    label: object, patterns: Iterable[str] = PERSONAL_DATA_PATTERNS
) -> bool:
    """Whether ``label`` names a signal with personal data."""
    if label is None:
        return False
    return bool(personal_pattern(tuple(patterns)).search(str(label)))
