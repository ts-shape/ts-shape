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
default ``exclude`` of :func:`ts_shape.events.profiling.runner.prepare_signals` and
:func:`ts_shape.profile`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

__all__ = ["PERSONAL_DATA_PATTERNS", "is_personal", "personal_pattern"]

# Words such as "User defined" or "Bediener-Taste" (operator button) name a
# machine function, not a person, and are not matched.
# A word boundary that treats "_", "-", "." and digits as separators
# ("User_ID", "Current_User"), unlike ``\b``.
_L = r"(?<![a-zäöüß])"
_R = r"(?![a-zäöüß])"

#: Regular expressions (case-insensitive) of signal labels with personal data.
PERSONAL_DATA_PATTERNS: tuple[str, ...] = (
    rf"{_L}worker",
    rf"{_L}operator(?:s|name|id)?{_R}",
    rf"{_L}personnel{_R}",
    rf"{_L}employee",
    rf"{_L}user(?:s|name|id)?{_R}(?![-_ ]*(?:defined|def{_R}|programm|button|panel))",
    rf"{_L}log[-_ ]?in(?:name)?{_R}",
    rf"{_L}badge",
    rf"{_L}werker",
    rf"{_L}bediener(?:in|name|nummer|nr)?{_R}(?![-_ ]*(?:taste|panel|feld|seite))",
    r"bedienpersonal",
    rf"{_L}mitarbeiter",
    rf"{_L}personal(?:[-_ .]?(?:nr|nummer|number|no|id|name))?{_R}",
    rf"{_L}benutzer(?:name|id|kennung)?{_R}",
    rf"{_L}anmeldung{_R}",
    rf"{_L}ausweis",
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
