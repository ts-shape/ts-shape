"""Find the signals that identify objects, and say what kind of object.

:func:`identify_objects` screens every signal of a long-format frame for
values that name a physical object or a manufacturing entity -- a serial, a
batch, a work order, a carrier, a tool, an operator badge -- and then
classifies the type of each one. The result is a separate
:class:`ObjectReport`; it never changes a :class:`~ts_shape.signals.SignalSpec`
or what :func:`~ts_shape.signals.infer_spec` returns.

Only the data counts as evidence: the values and how they behave over time.
The uuid, labels, descriptions and other metadata are never read. The one
exception is an explicit ``hints={uuid: object_type}``, which pins the type.

Shared id semantics (the same as the id-consuming detectors and
:mod:`ts_shape.eventlog.objects`): a value is held until the next value; a
change starts a new item; an empty reading (``""``, whitespace, NaN) means
"no item".

Stage 1, screening
------------------
Every signal is reduced to its run-length encoding and its distinct values
(O(n), one signal in memory at a time). A signal first meets the negative
roles, in this order: ``insufficient_data`` (fewer than ``min_samples`` rows,
or no non-empty reading), ``constant``, ``state`` (booleans),
``measurement`` (non-integral floats), ``timestamp``, ``counter``,
``measurement`` (a quantized analog value, or a number that changes at nearly
every sample with no format and no holding), ``state`` (few word-like values
or small integer codes), ``free_text``. What is left gets an
``identifier_score`` from :data:`SCREEN_WEIGHTS`; at
:data:`IDENTIFIER_MIN_SCORE` or more the role is ``identifier``.

Stage 2, identification
-----------------------
Facts about an identifier's values (formats, :data:`FACT_DESCRIPTIONS`) and
its behaviour (cardinality, holds, revisits, recurrence, shift alignment)
add the weights of :data:`TYPE_RULES` to the types they point at. Cross-signal
structure -- nesting (each presence of A lies inside one presence of B),
co-changing event-stamped ids, and changes aligned with a counter -- adds a
second round of facts. The best type wins when it reaches
:data:`TYPE_MIN_SCORE`; otherwise the type is ``identifier`` and the evidence
says what was missing. When the two best types belong to one group of
:data:`IDENTIFIER_TYPES` and no format cue separates them, the group is
reported (``order``, ``business_partner``).

Confidence: ``best / (best + runner_up) * min(1, best / CONFIDENT_SCORE)``,
with ``runner_up`` the best score outside the reported type (or group).

Extend it with :func:`register_identifier_type` (new types and groups, also
registered as OCEL object types) and :func:`register_identifier_pattern`
(site-specific formats).
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.signals.infer import (
    STATE_MAX_CARDINALITY,
    _is_counter,
    _iter_signal_groups,
    most_populated_value_column,
)
from ts_shape.utils.deprecation import caller_stacklevel
from ts_shape.utils.frames import require_columns, to_ns
from ts_shape.utils.shifts import DEFAULT_SHIFTS

if TYPE_CHECKING:
    from ts_shape.eventlog.objects import ObjectSpec

__all__ = [
    "FACT_DESCRIPTIONS",
    "IDENTIFIER_TYPES",
    "ObjectReport",
    "PREFIX_FAMILIES",
    "ROLES",
    "SCREEN_WEIGHTS",
    "TYPE_RULES",
    "identify_objects",
    "register_identifier_pattern",
    "register_identifier_type",
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Every object type the classifier can report, mapped to its group (or
#: None). A group is reported instead of a member when the data cannot
#: separate the members without a format cue. ``identifier`` is the fallback.
IDENTIFIER_TYPES: dict[str, str | None] = {
    # item and material
    "serial": None,  # one produced unit
    "batch": None,  # a production batch
    "lot": None,  # a lot (often of incoming material)
    "material": None,  # material number
    "article": None,  # part / article number
    "recipe": None,  # recipe or product variant
    # orders
    "work_order": None,  # production order
    "sales_order": "order",
    "purchase_order": "order",
    # resources
    "tool": None,  # tool, mold or die
    "fixture": None,
    "carrier": None,  # carrier, container, pallet, tray, kanban bin
    "rfid_tag": None,
    "asset": None,  # machine
    "station": None,
    "location": None,  # storage bin
    # people and partners
    "operator": None,  # operator or badge
    "customer": "business_partner",
    "supplier": "business_partner",
    # units
    "handling_unit": None,  # package or SSCC
    # fallback
    "identifier": None,
}

#: Roles a signal can have in the report.
ROLES: tuple[str, ...] = (
    "identifier",
    "state",
    "free_text",
    "measurement",
    "counter",
    "timestamp",
    "constant",
    "insufficient_data",
)

#: Object types that are resources or people: never part of a nesting, as
#: parent or child (a long-running tool contains every batch without owning
#: it).
_NOT_NESTED = frozenset(
    {"tool", "fixture", "carrier", "rfid_tag", "asset", "station", "location"}
    | {"operator", "customer", "supplier", "business_partner"}
)

# ---------------------------------------------------------------------------
# Thresholds (documented; all values are on the data, never on names)
# ---------------------------------------------------------------------------

#: A signal is an identifier at this screening score or more.
IDENTIFIER_MIN_SCORE = 0.5
#: The best type is reported at this score or more; else ``identifier``.
TYPE_MIN_SCORE = 0.5
#: A type score at which confidence is no longer damped.
CONFIDENT_SCORE = 0.8
#: Two group members closer than this are reported as their group.
GROUP_MARGIN = 0.15
#: A format holds when at least this share of the sampled values match.
FORMAT_MIN_SHARE = 0.9
#: At most this many distinct values (and run pairs) are format-checked.
FORMAT_SAMPLE = 2000
#: A date code is at most this many days older than the value's first
#: appearance.
DATE_CODE_MAX_AGE_DAYS = 31
#: Hold bands, in seconds.
CYCLE_MAX_HOLD_SECONDS = 1800.0
LONG_HOLD_SECONDS = 1800.0
HOURS_HOLD_SECONDS = 3600.0
VERY_LONG_HOLD_SECONDS = 8 * 3600.0
#: Cardinality bands (distinct non-empty values).
VERY_FEW_VALUES = 8
FEW_VALUES = 30
MEDIUM_VALUES = 100
HIGH_CARDINALITY = 50
#: Revisit bands (``revisit_ratio``).
NEVER_REVISITED = 0.02
RARELY_REVISITED = 0.10
OFTEN_REVISITED = 0.40
#: Sequential: the next value's number is 1..SEQ_MAX_STEP above.
SEQ_MAX_STEP = 5
#: A ratio (sequential, increasing, successor, shift alignment) holds here.
RATIO_HOLDS = 0.8
#: Regular recurrence: ``recurrence_cv`` at most this.
REGULAR_RECURRENCE_CV = 0.35
#: Event-stamped: this share of id runs is one sample cleared by an empty.
EVENT_STAMPED_SHARE = 0.8
#: A change within this many seconds of a shift boundary is aligned.
SHIFT_TOLERANCE_SECONDS = 900.0
#: Two changes within this many seconds happen together.
COCHANGE_TOLERANCE_SECONDS = 5.0
#: Nesting: this share of the child's presences lies inside the parent's,
#: the child has at least NEST_MIN_FANOUT times as many presences, and
#: this share of the parent's changes has a child change within
#: NEST_ALIGN_HOLD_SHARE of the child's median hold (at least
#: COCHANGE_TOLERANCE_SECONDS). Without the alignment, any short presence
#: lies trivially inside any long one.
NEST_MIN_SHARE = 0.9
NEST_MIN_FANOUT = 1.5
NEST_ALIGN_SHARE = 0.8
NEST_ALIGN_HOLD_SHARE = 0.1
#: Cross-signal work runs among at most this many identifiers.
MAX_CROSS_SIGNALS = 64
#: Screening thresholds of the negative roles.
#: word-like share of the values for a ``state`` (with <= 16 values)
STATE_WORD_SHARE = 0.8
#: quantized analog ``measurement``: this share of changes are small steps
#: (<= 5 % of the range) and at least QUANTIZED_DIRECTION_SHARE go each way
QUANTIZED_SMALL_STEP_SHARE = 0.6
QUANTIZED_DIRECTION_SHARE = 0.2
#: "changes at nearly every sample": runs / non-empty samples at least this
EVERY_SAMPLE_SHARE = 0.95
#: ``free_text``: this share of values contains a space, the length varies
#: (CV) or is long, and the shapes are irregular
FREE_TEXT_SPACE_SHARE = 0.5
FREE_TEXT_LENGTH_CV = 0.25
FREE_TEXT_MEAN_LENGTH = 20
FREE_TEXT_MAX_REGULARITY = 0.6
#: ``fixed_pool`` screening feature: recurrence CV at most this
FIXED_POOL_CV = 0.5

#: Screening features: name -> weight. The identifier score is their sum,
#: clipped to 0..1. See :data:`FACT_DESCRIPTIONS` for each condition.
SCREEN_WEIGHTS: dict[str, float] = {
    "high_cardinality": 0.30,  # more distinct values than the state band (16)
    "fixed_pool": 0.30,  # 3+ values that recur regularly
    "low_cardinality_codes": 0.20,  # 2..16 values that are codes, not words
    "regular_format": 0.25,  # one shape signature for 80 %+ of the values
    "known_format": 0.35,  # a format detector matches 90 %+ of the values
    "held": 0.15,  # values held over several samples
    "event_stamped": 0.15,  # one-sample ids cleared by an empty reading
    "token": 0.10,  # no spaces, 2..40 characters
    "changes_every_sample": -0.30,  # numeric, new value at nearly every sample
}

#: Prefix families, judged on the value content only (the leading letters of
#: the value). Low weight, never decisive alone.
PREFIX_FAMILIES: dict[str, tuple[str, ...]] = {
    "SN": ("serial",),
    "SER": ("serial",),
    "B": ("batch",),
    "BT": ("batch",),
    "CH": ("batch",),
    "BATCH": ("batch",),
    "LOT": ("lot",),
    "MAT": ("material",),
    "ART": ("article",),
    "PN": ("article",),
    "PART": ("article",),
    "R": ("recipe",),
    "RCP": ("recipe",),
    "REZ": ("recipe",),
    "WO": ("work_order",),
    "FA": ("work_order",),
    "MO": ("work_order",),
    "SO": ("sales_order",),
    "KA": ("sales_order",),
    "PO": ("purchase_order",),
    "BE": ("purchase_order",),
    "T": ("tool",),
    "TL": ("tool",),
    "WZ": ("tool",),
    "MOLD": ("tool",),
    "DIE": ("tool",),
    "FX": ("fixture",),
    "FIX": ("fixture",),
    "WT": ("carrier",),
    "CAR": ("carrier",),
    "PAL": ("carrier",),
    "TRAY": ("carrier",),
    "KLT": ("carrier",),
    "BIN": ("carrier", "location"),
    "MC": ("asset",),
    "MCH": ("asset",),
    "EQ": ("asset",),
    "ST": ("station",),
    "STN": ("station",),
    "OP": ("station",),
    "WS": ("station",),
    "LOC": ("location",),
    "EMP": ("operator",),
    "BDG": ("operator",),
    "USR": ("operator",),
    "CUST": ("customer",),
    "KD": ("customer",),
    "SUP": ("supplier",),
    "VEND": ("supplier",),
    "LF": ("supplier",),
    "HU": ("handling_unit",),
    "SSCC": ("handling_unit",),
    "PKG": ("handling_unit",),
}
PREFIX_WEIGHT = 0.25

#: What each fact means (the evidence wording).
FACT_DESCRIPTIONS: dict[str, str] = {
    "high_cardinality": f"many distinct values (>= {HIGH_CARDINALITY})",
    "medium_cardinality": f"2..{MEDIUM_VALUES} distinct values",
    "few_values": f"few distinct values (<= {FEW_VALUES})",
    "very_few_values": f"very few distinct values (<= {VERY_FEW_VALUES})",
    "never_revisited": f"values never come back (revisit <= {NEVER_REVISITED})",
    "rarely_revisited": f"values rarely come back (revisit <= {RARELY_REVISITED})",
    "occasionally_revisited": (
        f"values occasionally come back ({NEVER_REVISITED} < revisit <= "
        f"{OFTEN_REVISITED})"
    ),
    "often_revisited": f"values often come back (revisit > {OFTEN_REVISITED})",
    "cycle_hold": f"held about one cycle (median <= {CYCLE_MAX_HOLD_SECONDS:.0f} s)",
    "long_hold": f"long holds (median >= {LONG_HOLD_SECONDS:.0f} s)",
    "hours_hold": f"hours-long holds (median >= {HOURS_HOLD_SECONDS:.0f} s)",
    "very_long_hold": f"very long holds (median >= {VERY_LONG_HOLD_SECONDS:.0f} s)",
    "sequential": f"next value is 1..{SEQ_MAX_STEP} above (>= {RATIO_HOLDS:.0%})",
    "increasing": f"values increase (>= {RATIO_HOLDS:.0%} of changes)",
    "regular_recurrence": (
        f"a fixed pool recurring regularly (recurrence CV <= {REGULAR_RECURRENCE_CV})"
    ),
    "fixed_successor": f"each value has a fixed successor (>= {RATIO_HOLDS:.0%})",
    "event_stamped": "one-sample ids cleared by an empty reading",
    "shift_aligned": f"changes at shift boundaries (>= {RATIO_HOLDS:.0%})",
    "positional_steps": "numbers in steps of 10 (station positions)",
    "hierarchical_code": "two-level positional code (e.g. L1-M03)",
    "gs1_sscc": "GS1 element string with AI (00) SSCC",
    "gs1_serial": "GS1 element string with AI (21) serial",
    "gs1_batch": "GS1 element string with AI (10) batch/lot",
    "gs1_gtin": "GS1 element string with AI (01) GTIN",
    "gtin": "GTIN-8/12/13/14 with a valid check digit",
    "sscc": "SSCC-18 with a valid check digit",
    "sap_material": "SAP-style 18-digit zero-padded material number",
    "date_coded": "date-coded (YYMMDD / YYWW / YYDDD + sequence, valid date)",
    "epc_hex": "EPC/RFID hex (24 or 32 hex characters)",
    "guid": "UUID/GUID (type-neutral)",
    "location_code": "aisle-rack-level location code",
    "counter_aligned": "changes once per counted part (aligned with a counter)",
    "nested_child": "each presence lies inside one presence of another id",
    "nested_depth2": "nested two levels deep",
    "parent_of_serial": "contains the presences of a serial-like id",
    "parent_of_batch": "contains the presences of a batch-like id",
    "cochanges_with_unique": "changes together with an id that never repeats",
    "cochanges_with_recurring": "changes together with an id that recurs",
}

#: Type rules: ``(target, fact, weight)``. A target is a type or a group of
#: :data:`IDENTIFIER_TYPES` (a group rule adds to each member). Prefix
#: families (:data:`PREFIX_FAMILIES`, weight :data:`PREFIX_WEIGHT`) and
#: registered patterns add to their types as well.
TYPE_RULES: tuple[tuple[str, str, float], ...] = (
    # serial: very high cardinality, never revisited, about one cycle
    ("serial", "never_revisited", 0.25),
    ("serial", "high_cardinality", 0.25),
    ("serial", "cycle_hold", 0.25),
    ("serial", "sequential", 0.15),
    ("serial", "counter_aligned", 0.50),
    ("serial", "nested_child", 0.10),
    ("serial", "nested_depth2", 0.15),
    ("serial", "gs1_serial", 1.20),
    ("serial", "long_hold", -0.30),
    ("serial", "often_revisited", -0.40),
    ("serial", "event_stamped", -0.20),
    # batch: contiguous long holds, rarely revisited, often increasing
    ("batch", "never_revisited", 0.20),
    ("batch", "long_hold", 0.30),
    ("batch", "increasing", 0.15),
    ("batch", "parent_of_serial", 0.15),
    ("batch", "date_coded", 0.70),
    ("batch", "gs1_batch", 1.20),
    ("batch", "cycle_hold", -0.20),
    ("batch", "often_revisited", -0.30),
    ("batch", "event_stamped", -0.30),
    # lot: long holds, occasionally revisited (interleaved consumption)
    ("lot", "rarely_revisited", 0.10),
    ("lot", "occasionally_revisited", 0.25),
    ("lot", "long_hold", 0.25),
    ("lot", "date_coded", 0.60),
    ("lot", "gs1_batch", 1.00),
    ("lot", "cycle_hold", -0.20),
    ("lot", "event_stamped", -0.30),
    # work order: hours-long holds, occasionally revisited
    ("work_order", "hours_hold", 0.30),
    ("work_order", "occasionally_revisited", 0.25),
    ("work_order", "increasing", 0.10),
    ("work_order", "parent_of_serial", 0.15),
    ("work_order", "parent_of_batch", 0.15),
    ("work_order", "cycle_hold", -0.30),
    ("work_order", "often_revisited", -0.30),
    ("work_order", "event_stamped", -0.30),
    ("work_order", "very_long_hold", -0.15),
    # recipe / material / article: few to medium values, revisited, long holds
    ("recipe", "medium_cardinality", 0.20),
    ("recipe", "often_revisited", 0.30),
    ("recipe", "long_hold", 0.20),
    ("recipe", "very_long_hold", -0.20),
    ("recipe", "event_stamped", -0.30),
    ("recipe", "shift_aligned", -0.30),
    ("material", "medium_cardinality", 0.15),
    ("material", "often_revisited", 0.30),
    ("material", "long_hold", 0.20),
    ("material", "parent_of_batch", 0.35),
    ("material", "sap_material", 1.20),
    ("material", "event_stamped", -0.30),
    ("material", "shift_aligned", -0.30),
    ("article", "medium_cardinality", 0.15),
    ("article", "often_revisited", 0.30),
    ("article", "long_hold", 0.20),
    ("article", "parent_of_serial", 0.25),
    ("article", "gtin", 1.00),
    ("article", "gs1_gtin", 1.20),
    ("article", "event_stamped", -0.30),
    ("article", "shift_aligned", -0.30),
    # tool / fixture: very few values, very long holds
    ("tool", "very_few_values", 0.30),
    ("tool", "very_long_hold", 0.40),
    ("tool", "event_stamped", -0.30),
    ("tool", "shift_aligned", -0.30),
    ("tool", "hierarchical_code", -0.30),
    ("fixture", "very_few_values", 0.27),
    ("fixture", "very_long_hold", 0.36),
    ("fixture", "event_stamped", -0.30),
    ("fixture", "shift_aligned", -0.30),
    ("fixture", "hierarchical_code", -0.30),
    # carrier: a fixed pool recurring regularly
    ("carrier", "regular_recurrence", 0.45),
    ("carrier", "fixed_successor", 0.15),
    ("carrier", "often_revisited", 0.10),
    ("carrier", "positional_steps", -0.30),
    ("carrier", "very_long_hold", -0.30),
    # rfid tag: EPC format
    ("rfid_tag", "epc_hex", 1.20),
    # asset / station / location: few values, positional formats
    ("asset", "hierarchical_code", 0.45),
    ("asset", "very_few_values", 0.15),
    ("asset", "very_long_hold", 0.15),
    ("station", "positional_steps", 0.35),
    ("station", "few_values", 0.10),
    ("station", "fixed_successor", 0.15),
    ("station", "often_revisited", 0.10),
    ("location", "location_code", 1.20),
    # operator: changes at shift boundaries
    ("operator", "shift_aligned", 0.90),
    ("operator", "medium_cardinality", 0.10),
    ("operator", "often_revisited", 0.10),
    ("operator", "event_stamped", -0.30),
    # orders and partners: event-stamped
    ("order", "event_stamped", 0.35),
    ("order", "never_revisited", 0.20),
    ("order", "cochanges_with_recurring", 0.30),
    ("business_partner", "event_stamped", 0.30),
    ("business_partner", "often_revisited", 0.25),
    ("business_partner", "cochanges_with_unique", 0.30),
    # handling unit: SSCC
    ("handling_unit", "sscc", 1.20),
    ("handling_unit", "gs1_sscc", 1.20),
    ("handling_unit", "event_stamped", 0.10),
    ("handling_unit", "never_revisited", 0.10),
)

# Facts that count as a known format in screening.
_KNOWN_FORMATS = frozenset(
    {"gs1_sscc", "gs1_serial", "gs1_batch", "gs1_gtin", "gtin", "sscc"}
    | {"sap_material", "date_coded", "epc_hex", "guid", "location_code"}
)


@dataclass(frozen=True)
class _Pattern:
    name: str
    test: Callable[[str], bool]
    object_type: str
    weight: float


_PATTERNS: dict[str, _Pattern] = {}


def _groups() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, group in IDENTIFIER_TYPES.items():
        if group is not None:
            out.setdefault(group, []).append(name)
    return out


def _known_target(name: str) -> bool:
    return name in IDENTIFIER_TYPES or name in _groups()


def _register_ocel(names: Iterable[str]) -> None:
    from ts_shape.eventlog import schema

    for name in names:
        if not schema.is_known_object_type(name):
            schema.register_object_type(name)


def _register_vocabulary() -> None:
    _register_ocel([*IDENTIFIER_TYPES, *_groups()])


def register_identifier_type(name: str, *, group: str | None = None) -> None:
    """Add an object type to the vocabulary (and to the OCEL object types).

    The new type gets no type rules of its own: give it evidence with
    :func:`register_identifier_pattern`, or a ``group`` whose rules it shares
    (a rule on a group adds to each member).

    Args:
        name: The new type, e.g. ``"coil"``. Lowercase, no ``":"``.
        group: Optional group name; created when new. A group is reported
            instead of a member when the data cannot separate the members.

    Raises:
        ValueError: When ``name`` or ``group`` is empty, contains ``":"``,
            or ``group`` is itself a type.
    """
    for label, value in (("name", name), ("group", group)):
        if value is not None and (not value or ":" in value):
            raise ValueError(f"invalid identifier type {label} {value!r}")
    if group is not None and group in IDENTIFIER_TYPES:
        raise ValueError(f"group {group!r} is an object type, not a group")
    if name in _groups():
        raise ValueError(f"{name!r} is a group, not an object type")
    IDENTIFIER_TYPES[name] = group
    _register_ocel([name] + ([group] if group else []))


def register_identifier_pattern(
    name: str,
    regex_or_callable: str | re.Pattern[str] | Callable[[str], bool],
    object_type: str,
    *,
    weight: float,
) -> None:
    """Register a site-specific id format as evidence for a type.

    When at least :data:`FORMAT_MIN_SHARE` of a signal's sampled values
    match, the signal counts as having a known format (screening) and
    ``object_type`` gains ``weight``. Registering a name again replaces it.

    Args:
        name: Name of the pattern, shown in the evidence.
        regex_or_callable: A regular expression (the whole value must
            match) or a function ``str -> bool``.
        object_type: A type or group of :data:`IDENTIFIER_TYPES`.
        weight: Score added to ``object_type`` (positive; 1.0 or more is a
            decisive format).

    Raises:
        ValueError: For an unknown ``object_type``, a non-positive
            ``weight``, or an empty ``name``.
    """
    if not name:
        raise ValueError("pattern name must not be empty")
    if not _known_target(object_type):
        raise ValueError(
            f"unknown object type {object_type!r}; register it first with "
            "register_identifier_type"
        )
    if not weight > 0:
        raise ValueError(f"weight must be positive; got {weight!r}")
    if callable(regex_or_callable) and not isinstance(regex_or_callable, re.Pattern):
        func = regex_or_callable

        def test(text: str) -> bool:
            return bool(func(text))

    else:
        rx = re.compile(regex_or_callable)

        def test(text: str) -> bool:
            return rx.fullmatch(text) is not None

    _PATTERNS[name] = _Pattern(name, test, object_type, float(weight))


# ---------------------------------------------------------------------------
# Format detectors
# ---------------------------------------------------------------------------


def _mod10_ok(digits: str) -> bool:
    """GS1 mod-10 check digit (GTIN, SSCC)."""
    body, check = digits[:-1], ord(digits[-1]) - 48
    total = 0
    for i, ch in enumerate(reversed(body)):
        total += (ord(ch) - 48) * (3 if i % 2 == 0 else 1)
    return (10 - total % 10) % 10 == check


_FIXED_AI = {"00": 18, "01": 14, "02": 14, "11": 6, "13": 6, "15": 6, "17": 6}
_VAR_AI = {"10", "21", "22", "30", "37", "240", "241", "250", "400"}
_GS = "\x1d"
_PAREN = re.compile(r"\((\d{2,4})\)([^()]*)")


def _gs1_valid(ai: str, data: str) -> bool:
    if ai in _FIXED_AI:
        n = _FIXED_AI[ai]
        if len(data) != n or not data.isdigit():
            return False
        return _mod10_ok(data) if ai in ("00", "01", "02") else True
    return ai in _VAR_AI and 0 < len(data) <= 30


def _gs1_parse(text: str) -> set[str] | None:
    """The application identifiers of a GS1 element string, or None."""
    if text.startswith("("):
        found = _PAREN.findall(text)
        if not found or "".join(f"({a}){d}" for a, d in found) != text:
            return None
        ais = set()
        for ai, data in found:
            if not _gs1_valid(ai, data):
                return None
            ais.add(ai)
        return ais
    if not (text.startswith("00") or text.startswith("01")) or len(text) < 16:
        return None
    pos, ais = 0, set()
    while pos < len(text):
        if text[pos] == _GS:
            pos += 1
            continue
        ai = text[pos : pos + 2]
        if ai in _FIXED_AI:
            n = _FIXED_AI[ai]
            data = text[pos + 2 : pos + 2 + n]
            pos += 2 + n
        else:
            ai3 = text[pos : pos + 3]
            ai = ai3 if ai3 in _VAR_AI else ai
            if ai not in _VAR_AI:
                return None
            end = text.find(_GS, pos + len(ai))
            end = len(text) if end < 0 else end
            data = text[pos + len(ai) : end]
            pos = end
        if not _gs1_valid(ai, data):
            return None
        ais.add(ai)
    return ais or None


_GTIN = re.compile(r"\d{8}|\d{12,14}")
_DIGITS18 = re.compile(r"\d{18}")
_SAP = re.compile(r"0{6,}\d{1,12}")
_EPC = re.compile(r"[0-9A-Fa-f]{24}|[0-9A-Fa-f]{32}")
_EPC_HEADERS = ("30", "31", "32", "33", "34", "35", "36", "2F", "2f", "E2", "e2")
_GUID = re.compile(
    r"\{?[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{12}\}?"
)
_LOCATION = re.compile(
    r"(?:[A-Z]{1,3}\d{0,3}|\d{1,3})([-./])\d{1,3}\1\d{1,3}(?:\1\d{1,3})?"
)
_HIERARCHICAL = re.compile(r"[A-Z]{1,4}\d{1,3}[-_.][A-Z]{1,4}\d{1,3}")
_ISO_TIME = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"|\d{1,2}[./]\d{1,2}[./]\d{4}(?: \d{1,2}:\d{2}(?::\d{2})?)?"
)
_LEAD_LETTERS = re.compile(r"([A-Za-z]+)")
_DIGIT_RUNS = re.compile(r"\d+")
_WORD_LIKE = re.compile(r"[A-Za-z][A-Za-z _/\-]*")


def _date_coded(text: str, seen: pd.Timestamp) -> bool:
    """A date part (YYMMDD, YYDDD or YYWW) then a sequence, dated when seen.

    The date must be valid and lie between :data:`DATE_CODE_MAX_AGE_DAYS`
    before and one day after the value first appears (a week earlier for
    YYWW), so a sequential serial such as ``SN250023`` is not read as day 23
    of 2025.
    """
    runs = _DIGIT_RUNS.findall(text)
    if not runs:
        return False
    head = runs[0]
    rest = len(runs) > 1
    lo = seen.normalize() - pd.Timedelta(days=DATE_CODE_MAX_AGE_DAYS)
    hi = seen.normalize() + pd.Timedelta(days=1)
    if len(head) < 4 or int(head[:2]) not in {lo.year % 100, hi.year % 100}:
        return False
    year = 2000 + int(head[:2])

    def ok(day: pd.Timestamp, slack: int = 0) -> bool:
        return bool(lo - pd.Timedelta(days=slack) <= day <= hi)

    if len(head) >= 6 + (0 if rest else 1):
        month, dom = int(head[2:4]), int(head[4:6])
        if 1 <= month <= 12 and 1 <= dom <= 31:
            try:
                if ok(pd.Timestamp(year=year, month=month, day=dom)):
                    return True
            except ValueError:
                pass
    if len(head) >= 5 + (0 if rest else 1):
        doy = int(head[2:5])
        if 1 <= doy <= 366 and ok(
            pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        ):
            return True
    if len(head) >= 4 + (0 if rest else 1):
        week = int(head[2:4])
        if 1 <= week <= 53 and ok(
            pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(weeks=week - 1), 7
        ):
            return True
    return False


def _format_facts(texts: list[str], seen: list[pd.Timestamp]) -> dict[str, float]:
    """Share of ``texts`` that match each format (only shares > 0).

    ``seen`` is when each value first appears (for date codes).
    """
    n = len(texts)
    if n == 0:
        return {}
    counts: Counter[str] = Counter()
    gs1_ais: Counter[str] = Counter()
    gs1_n = 0
    epc_letters = 0
    for text, first in zip(texts, seen, strict=True):
        ais = _gs1_parse(text) if len(text) >= 16 else None
        if ais:
            gs1_n += 1
            gs1_ais.update(ais)
            continue
        if text.isdigit():
            if len(text) == 18 and _SAP.fullmatch(text):
                counts["sap_material"] += 1
            elif _DIGITS18.fullmatch(text) and _mod10_ok(text):
                counts["sscc"] += 1
            elif _GTIN.fullmatch(text) and _mod10_ok(text):
                counts["gtin"] += 1
        if _EPC.fullmatch(text):
            counts["epc_hex"] += 1
            if not text.isdigit() or text.startswith(_EPC_HEADERS):
                epc_letters += 1
        if _GUID.fullmatch(text):
            counts["guid"] += 1
        if _LOCATION.fullmatch(text):
            counts["location_code"] += 1
        if _HIERARCHICAL.fullmatch(text):
            counts["hierarchical_code"] += 1
        if _date_coded(text, first):
            counts["date_coded"] += 1
    if epc_letters < 0.5 * counts.get("epc_hex", 0):
        counts.pop("epc_hex", None)
    shares = {k: v / n for k, v in counts.items()}
    if gs1_n:
        share = gs1_n / n
        for ai, fact in (("00", "gs1_sscc"), ("21", "gs1_serial"), ("10", "gs1_batch")):
            if gs1_ais[ai] >= FORMAT_MIN_SHARE * gs1_n:
                shares[fact] = share
                break
        else:
            if gs1_ais["01"] >= FORMAT_MIN_SHARE * gs1_n:
                shares["gs1_gtin"] = share
    for pattern in _PATTERNS.values():
        hits = sum(1 for text in texts if _safe(pattern.test, text))
        if hits:
            shares[f"pattern:{pattern.name}"] = hits / n
    lead = Counter(
        m.group(1).upper() if (m := _LEAD_LETTERS.match(t)) else "" for t in texts
    )
    token, hits = lead.most_common(1)[0]
    if token in PREFIX_FAMILIES and token != "":
        rest_ok = sum(1 for t in texts if len(t) > len(token)) / n
        if rest_ok >= FORMAT_MIN_SHARE:
            shares[f"prefix:{token}"] = hits / n
    return shares


def _safe(test: Callable[[str], bool], text: str) -> bool:
    try:
        return bool(test(text))
    except Exception:  # a site pattern must not break the screening
        return False


_SHAPE_TABLE = {
    **{c: "9" for c in "0123456789"},
    **{c: "A" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
    **{c: "a" for c in "abcdefghijklmnopqrstuvwxyz"},
}


def _shape(text: str, coarse: bool = False) -> str:
    """Shape signature: ``A`` upper, ``a`` lower, ``9`` digit, other kept.

    ``coarse`` maps every letter and digit to ``X``.
    """
    mapped = (
        (
            ("X" if ch.isalnum() and ch.isascii() else ch)
            if coarse
            else _SHAPE_TABLE.get(ch, ch)
        )
        for ch in text
    )
    parts = []
    for ch, grp in groupby(mapped):
        k = sum(1 for _ in grp)
        parts.append(f"{ch}{{{k}}}" if k > 1 else ch)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Per-signal screening
# ---------------------------------------------------------------------------

_I64: TypeAlias = "npt.NDArray[np.int64]"


@dataclass
class _Signal:
    uuid: str
    value_column: str | None
    sample_count: int
    role: str = "insufficient_data"
    score: float = math.nan
    metrics: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    screen_evidence: list[str] = field(default_factory=list)
    # presence runs (id runs, empties break them)
    p_start: _I64 = field(default_factory=lambda: np.empty(0, np.int64))
    p_end: _I64 = field(default_factory=lambda: np.empty(0, np.int64))
    # changes: start of every id presence after the first
    changes: _I64 = field(default_factory=lambda: np.empty(0, np.int64))
    increments: _I64 = field(default_factory=lambda: np.empty(0, np.int64))
    tz: Any = None
    scores: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    object_type: str | None = None
    confidence: float = math.nan
    runner_up: str | None = None
    runner_up_confidence: float = math.nan
    parent: str | None = None
    depth: int = 0
    distinct: int = 0
    runs: int = 0


def _id_text(value: object) -> str:
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _sample_positions(n: int, k: int = FORMAT_SAMPLE) -> npt.NDArray[np.intp]:
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(np.intp))


def _digits_number(text: str) -> int | None:
    digits = "".join(_DIGIT_RUNS.findall(text))
    return int(digits) if digits else None


def _pair_ratios(numbers: list[int | None]) -> tuple[float, float, float, float]:
    """Sequential, increasing, small-step and down shares of consecutive pairs."""
    seq = inc = small = down = total = 0
    valid = [x for x in numbers if x is not None]
    span = (max(valid) - min(valid)) if valid else 0
    for a, b in zip(numbers[0::2], numbers[1::2], strict=False):
        if a is None or b is None:
            continue
        total += 1
        d = b - a
        seq += 0 < d <= SEQ_MAX_STEP
        inc += d > 0
        down += d < 0
        small += abs(d) <= max(1, 0.05 * span)
    if total == 0:
        return math.nan, math.nan, math.nan, math.nan
    return seq / total, inc / total, small / total, down / total


def _recurrence(v_codes: _I64, v_start: _I64) -> tuple[float, float]:
    """Median CV of re-appearance intervals, and successor determinism."""
    if len(v_codes) < 3:
        return math.nan, math.nan
    order = np.lexsort((v_start, v_codes))
    c, t = v_codes[order], v_start[order].astype(np.float64)
    same = c[1:] == c[:-1]
    gaps = np.diff(t)[same]
    owner = c[1:][same]
    cv = math.nan
    if len(gaps):
        _, inv, cnt = np.unique(owner, return_inverse=True, return_counts=True)
        mean = np.bincount(inv, gaps) / cnt
        sq = np.bincount(inv, gaps * gaps) / cnt
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        ok = (cnt >= 2) & (mean > 0)
        if ok.any():
            cv = float(np.median(std[ok] / mean[ok]))
    # successor determinism: share of transitions to each value's modal successor
    a, b = v_codes[:-1], v_codes[1:]
    key = a.astype(np.int64) * (int(v_codes.max()) + 1) + b
    keys, kcnt = np.unique(key, return_counts=True)
    src = keys // (int(v_codes.max()) + 1)
    best = pd.Series(kcnt).groupby(src).max().to_numpy()
    succ = float(best.sum() / len(a))
    return cv, succ


def _is_epoch_like(values: npt.NDArray[np.float64], t_ns: _I64) -> bool:
    if len(values) == 0 or len(t_ns) == 0:
        return False
    med_v = float(np.median(values))
    med_t = float(np.median(t_ns)) / 1e9
    for scale in (1.0, 1e3, 1e6, 1e9):
        if abs(med_v / scale - med_t) <= 2 * 86400:
            return True
    return False


def _shift_alignment(
    changes: _I64, tz: Any, shift_definitions: Mapping[str, tuple[str, str]]
) -> float:
    if len(changes) == 0:
        return math.nan
    idx = pd.DatetimeIndex(changes.view("datetime64[ns]"))
    if tz is not None:
        idx = idx.tz_localize("UTC").tz_convert(tz)
    tod = (idx - idx.normalize()).total_seconds().to_numpy()
    bounds = []
    for start, end in shift_definitions.values():
        for clock in (start, end):
            t = pd.to_datetime(clock).time()
            bounds.append(t.hour * 3600 + t.minute * 60 + t.second)
    b = np.unique(np.asarray(bounds, dtype=np.float64))
    dist = np.abs(tod[:, None] - b[None, :])
    dist = np.minimum(dist, 86400 - dist).min(axis=1)
    return float((dist <= SHIFT_TOLERANCE_SECONDS).mean())


def _screen(
    key: str,
    rows: pd.DataFrame,
    *,
    time_column: str,
    min_samples: int,
    shift_definitions: Mapping[str, tuple[str, str]],
    hinted: bool = False,
) -> _Signal:
    from ts_shape.eventlog.objects import _id_codes

    times = rows[time_column]
    rows = rows[times.notna().to_numpy()]
    rows = rows.sort_values(time_column, kind="stable")
    n = len(rows)
    vc = most_populated_value_column(rows) if n else None
    sig = _Signal(uuid=key, value_column=vc, sample_count=n)
    if vc is None:
        sig.metrics = _empty_metrics(n)
        sig.screen_evidence.append("no values in any value column")
        return sig
    times = rows[time_column]
    sig.tz = getattr(times.dt, "tz", None)
    t = to_ns(times)
    raw = rows[vc].reset_index(drop=True)
    numeric = vc in ("value_integer", "value_double") and pd.api.types.is_numeric_dtype(
        raw.dtype
    )
    non_integral = False
    if numeric and vc == "value_double":
        finite = raw.dropna().to_numpy(dtype=float)
        non_integral = bool(len(finite)) and not bool(np.all(np.mod(finite, 1) == 0))
    if vc == "value_bool":
        codes, uniques = pd.factorize(raw, use_na_sentinel=True)
        codes = np.asarray(codes, dtype=np.int64)
    else:
        codes, uniques = _id_codes(raw)

    # presence runs: RLE over codes, empties (-1) break runs
    brk = np.ones(n, dtype=bool)
    brk[1:] = codes[1:] != codes[:-1]
    idx = np.flatnonzero(brk)
    rc = codes[idx]
    rstart = t[idx]
    rlen = np.diff(np.append(idx, n))
    rend = np.append(rstart[1:], t[-1])
    is_id = rc >= 0
    followed_empty = np.append(rc[1:] == -1, False)[is_id]
    p_start, p_end, p_len = rstart[is_id], rend[is_id], rlen[is_id]
    sig.p_start, sig.p_end = p_start, p_end
    sig.changes = p_start[1:]

    present = codes >= 0
    ne_codes = codes[present]
    ne_t = t[present]
    nonempty = len(ne_codes)
    empty_ratio = 1.0 - nonempty / n if n else math.nan
    if nonempty == 0:
        sig.metrics = _empty_metrics(n, empty_ratio=empty_ratio)
        sig.screen_evidence.append("no non-empty readings")
        return sig
    vbrk = np.ones(nonempty, dtype=bool)
    vbrk[1:] = ne_codes[1:] != ne_codes[:-1]
    v_codes, v_start = ne_codes[vbrk], ne_t[vbrk]
    used, first_pos = np.unique(v_codes, return_index=True)
    first_seen = v_start[first_pos]
    distinct, runs = len(used), len(v_codes)
    sig.distinct, sig.runs = distinct, runs
    revisit = (runs - distinct) / runs
    holds = (p_end - p_start) / 1e9
    if len(holds) >= 2 and rc[-1] >= 0:
        holds = holds[:-1]
    mean_hold = float(holds.mean()) if len(holds) else math.nan
    median_hold = float(np.median(holds)) if len(holds) else math.nan
    span = (t[-1] - t[0]) / 1e9 if n else 0.0
    cph = (runs - 1) / (span / 3600) if span > 0 else math.nan
    one_cleared = (p_len == 1) & followed_empty
    event_stamped = len(p_len) >= 3 and float(one_cleared.mean()) >= EVENT_STAMPED_SHARE
    held = len(p_len) > 0 and float(np.median(p_len)) >= 2

    # sampled texts of the distinct values, in order of first appearance
    sample_pos = _sample_positions(distinct)
    sample_codes = used[sample_pos]
    texts = [_id_text(uniques[c]) for c in sample_codes]
    shapes = Counter(_shape(x) for x in texts)
    pattern, pcount = shapes.most_common(1)[0]
    pattern_share = pcount / len(texts)
    coarse = Counter(_shape(x, coarse=True) for x in texts).most_common(1)[0][1]
    regularity = max(pattern_share, coarse / len(texts))
    lengths = np.array([len(x) for x in texts], dtype=float)
    length_cv = float(lengths.std() / lengths.mean()) if lengths.mean() > 0 else 0.0
    spaced = sum(" " in x.strip() for x in texts) / len(texts)
    word_like = sum(_WORD_LIKE.fullmatch(x) is not None for x in texts) / len(texts)
    token = sum((" " not in x) and 2 <= len(x) <= 40 for x in texts) / len(texts)

    pair_pos = _sample_positions(max(runs - 1, 0))
    pair_codes = np.empty(2 * len(pair_pos), dtype=np.int64)
    pair_codes[0::2] = v_codes[pair_pos]
    pair_codes[1::2] = v_codes[pair_pos + 1]
    numbers = [_digits_number(_id_text(uniques[c])) for c in pair_codes]
    seq_ratio, inc_ratio, small_ratio, down_ratio = _pair_ratios(numbers)
    rec_cv, successor = _recurrence(v_codes, v_start)

    sig.metrics = {
        "sample_count": n,
        "distinct_count": distinct,
        "run_count": runs,
        "mean_hold_seconds": mean_hold,
        "median_hold_seconds": median_hold,
        "revisit_ratio": revisit,
        "changes_per_hour": cph,
        "empty_ratio": empty_ratio,
        "pattern": pattern,
        "pattern_share": pattern_share,
        "sequential_ratio": seq_ratio,
        "recurrence_cv": rec_cv,
    }

    seen = [pd.Timestamp(int(x)) for x in first_seen[sample_pos]]
    formats = _format_facts(texts, seen)
    known = sorted(
        k
        for k, v in formats.items()
        if v >= FORMAT_MIN_SHARE and (k in _KNOWN_FORMATS or k.startswith("pattern:"))
    )

    def facts() -> dict[str, float]:
        out = {k: v for k, v in formats.items() if v >= FORMAT_MIN_SHARE}
        flags = {
            "high_cardinality": distinct >= HIGH_CARDINALITY,
            "medium_cardinality": 2 <= distinct <= MEDIUM_VALUES,
            "few_values": distinct <= FEW_VALUES,
            "very_few_values": distinct <= VERY_FEW_VALUES,
            "never_revisited": revisit <= NEVER_REVISITED,
            "rarely_revisited": revisit <= RARELY_REVISITED,
            "occasionally_revisited": NEVER_REVISITED < revisit <= OFTEN_REVISITED,
            "often_revisited": revisit > OFTEN_REVISITED,
            "event_stamped": event_stamped,
            "cycle_hold": not event_stamped and median_hold <= CYCLE_MAX_HOLD_SECONDS,
            "long_hold": not event_stamped and median_hold >= LONG_HOLD_SECONDS,
            "hours_hold": not event_stamped and median_hold >= HOURS_HOLD_SECONDS,
            "very_long_hold": (
                not event_stamped and median_hold >= VERY_LONG_HOLD_SECONDS
            ),
            "sequential": seq_ratio >= RATIO_HOLDS,
            "increasing": inc_ratio >= RATIO_HOLDS,
            "regular_recurrence": (
                distinct >= 3
                and revisit > OFTEN_REVISITED
                and not math.isnan(rec_cv)
                and rec_cv <= REGULAR_RECURRENCE_CV
            ),
            "fixed_successor": (
                runs >= 2 * distinct and not math.isnan(successor) and successor >= 0.9
            ),
            "positional_steps": 2 <= distinct <= FEW_VALUES and _positional(texts),
        }
        shift_share = _shift_alignment(sig.changes, sig.tz, shift_definitions)
        flags["shift_aligned"] = (
            len(sig.changes) >= 2
            and not math.isnan(shift_share)
            and shift_share >= RATIO_HOLDS
        )
        out.update({k: 1.0 for k, v in flags.items() if v})
        return out

    def role(name: str, why: str) -> _Signal:
        sig.role = name
        sig.screen_evidence.append(why)
        if hinted:
            sig.facts = facts()
        return sig

    if n < min_samples:
        return role("insufficient_data", f"{n} rows < min_samples={min_samples}")
    if distinct == 1:
        return role("constant", f"one value ({texts[0]!r})")
    if vc == "value_bool":
        return role("state", "boolean values")
    if non_integral:
        return role("measurement", "non-integral floats")
    num_values: npt.NDArray[np.float64] | None = None
    if numeric:
        num_values = raw.dropna().to_numpy(dtype=float)
    stamp_share = sum(_ISO_TIME.fullmatch(x) is not None for x in texts) / len(texts)
    if stamp_share >= FORMAT_MIN_SHARE:
        return role("timestamp", f"{stamp_share:.0%} of values parse as dates/times")
    if num_values is not None and _is_epoch_like(num_values, t):
        return role("timestamp", "numbers track the sample time (epoch)")
    if num_values is not None:
        series = pd.Series(num_values)
        is_counter, why = _is_counter(series)
        if is_counter and not _serial_like(texts, seq_ratio, series):
            inc = np.flatnonzero(np.diff(num_values) > 0) + 1
            sig.increments = t[raw.notna().to_numpy()][inc]
            return role("counter", f"counter: {why}")
        if (
            distinct > STATE_MAX_CARDINALITY
            and small_ratio >= QUANTIZED_SMALL_STEP_SHARE
            and down_ratio >= QUANTIZED_DIRECTION_SHARE
            and inc_ratio >= QUANTIZED_DIRECTION_SHARE
        ):
            return role(
                "measurement",
                f"quantized analog: {small_ratio:.0%} small steps up and down",
            )
        if (
            runs >= EVERY_SAMPLE_SHARE * nonempty
            and distinct >= 0.5 * runs
            and not (inc_ratio >= RATIO_HOLDS)
            and not known
        ):
            return role(
                "measurement", "a new number at nearly every sample, no format, no hold"
            )
    small_codes = (
        num_values is not None
        and bool(np.all(np.abs(num_values) <= 99))
        and not (_positional(texts) and distinct >= 3)
    )
    if (
        distinct <= STATE_MAX_CARDINALITY
        and not known
        and (word_like >= STATE_WORD_SHARE or small_codes)
    ):
        what = (
            "word-like values"
            if word_like >= STATE_WORD_SHARE
            else "small integer codes"
        )
        return role(
            "state", f"{distinct} {what} (state band <= {STATE_MAX_CARDINALITY})"
        )
    if (
        spaced >= FREE_TEXT_SPACE_SHARE
        and (
            length_cv >= FREE_TEXT_LENGTH_CV or lengths.mean() >= FREE_TEXT_MEAN_LENGTH
        )
        and regularity < FREE_TEXT_MAX_REGULARITY
    ):
        return role(
            "free_text",
            f"{spaced:.0%} of values contain spaces, length CV {length_cv:.2f}, "
            f"shape regularity {regularity:.0%}",
        )

    # identifier score
    feats: list[tuple[str, str]] = []
    if distinct > STATE_MAX_CARDINALITY:
        feats.append(
            (
                "high_cardinality",
                f"{distinct} distinct values (> {STATE_MAX_CARDINALITY})",
            )
        )
    elif (
        distinct >= 3
        and revisit > OFTEN_REVISITED
        and not math.isnan(rec_cv)
        and rec_cv <= FIXED_POOL_CV
    ):
        feats.append(
            ("fixed_pool", f"pool of {distinct} values recurring (CV {rec_cv:.2f})")
        )
    elif word_like < STATE_WORD_SHARE:
        feats.append(("low_cardinality_codes", f"{distinct} code values"))
    if regularity >= RATIO_HOLDS:
        feats.append(("regular_format", f"shape {pattern} ({regularity:.0%} regular)"))
    if known:
        feats.append(("known_format", "format " + ", ".join(known)))
    if held:
        feats.append(
            ("held", f"held {float(np.median(p_len)):.0f} samples per run (median)")
        )
    elif event_stamped:
        feats.append(("event_stamped", "one-sample ids cleared by an empty reading"))
    if token >= 0.95:
        feats.append(("token", "single tokens of 2..40 characters"))
    if num_values is not None and runs >= EVERY_SAMPLE_SHARE * nonempty and not known:
        feats.append(("changes_every_sample", "a new number at nearly every sample"))
    score = sum(SCREEN_WEIGHTS[f] for f, _ in feats)
    sig.score = float(min(1.0, max(0.0, score)))
    sig.screen_evidence.extend(f"{why} {SCREEN_WEIGHTS[f]:+.2f}" for f, why in feats)
    if sig.score < IDENTIFIER_MIN_SCORE:
        fallback = (
            "measurement"
            if num_values is not None
            else "state" if distinct <= STATE_MAX_CARDINALITY else "free_text"
        )
        return role(
            fallback,
            f"identifier score {sig.score:.2f} < {IDENTIFIER_MIN_SCORE}",
        )
    sig.role = "identifier"
    sig.facts = facts()
    return sig


def _positional(texts: Sequence[str]) -> bool:
    nums = [_digits_number(x) for x in texts]
    valid = [x for x in nums if x is not None]
    if len(valid) < FORMAT_MIN_SHARE * len(texts) or not valid:
        return False
    return sum(1 for x in valid if x % 10 == 0 and x > 0) >= FORMAT_MIN_SHARE * len(
        valid
    )


def _serial_like(texts: Sequence[str], seq_ratio: float, values: pd.Series) -> bool:
    """A monotonic integer that is a serial rather than a counter.

    Every value has the same number of digits (5 or more) and the value
    changes by +1..+SEQ_MAX_STEP, without a reset.
    """
    widths = {len(x) for x in texts}
    return (
        len(widths) == 1
        and min(widths) >= 5
        and seq_ratio >= RATIO_HOLDS
        and bool((values.diff().dropna() >= 0).all())
    )


def _empty_metrics(n: int, *, empty_ratio: float = math.nan) -> dict[str, Any]:
    return {
        "sample_count": n,
        "distinct_count": 0,
        "run_count": 0,
        "mean_hold_seconds": math.nan,
        "median_hold_seconds": math.nan,
        "revisit_ratio": math.nan,
        "changes_per_hour": math.nan,
        "empty_ratio": empty_ratio if n else math.nan,
        "pattern": None,
        "pattern_share": math.nan,
        "sequential_ratio": math.nan,
        "recurrence_cv": math.nan,
    }


# ---------------------------------------------------------------------------
# Stage 2: type scores
# ---------------------------------------------------------------------------


def _fact_text(fact: str) -> str:
    if fact.startswith("prefix:"):
        return f"prefix family {fact[7:]!r}"
    if fact.startswith("pattern:"):
        return f"pattern {fact[8:]!r}"
    return FACT_DESCRIPTIONS.get(fact, fact)


def _score_types(sig: _Signal) -> None:
    groups = _groups()
    scores: dict[str, float] = {t: 0.0 for t in IDENTIFIER_TYPES if t != "identifier"}
    contrib: dict[str, list[tuple[str, float]]] = {t: [] for t in scores}

    def add(target: str, fact: str, weight: float) -> None:
        for member in groups.get(target, [target]):
            if member in scores:
                scores[member] += weight
                contrib[member].append((fact, weight))

    for target, fact, weight in TYPE_RULES:
        if fact in sig.facts:
            add(target, fact, weight)
    for fact in sig.facts:
        if fact.startswith("prefix:"):
            for target in PREFIX_FAMILIES.get(fact[7:], ()):
                add(target, fact, PREFIX_WEIGHT)
        elif fact.startswith("pattern:"):
            pattern = _PATTERNS.get(fact[8:])
            if pattern is not None:
                add(pattern.object_type, fact, pattern.weight)
    sig.scores = {k: max(0.0, v) for k, v in scores.items()}
    sig.contributions = contrib


def _ranked(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    order = {t: i for i, t in enumerate(IDENTIFIER_TYPES)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order.get(kv[0], 0)))


def _confidence(best: float, other: float) -> float:
    if best <= 0:
        return 0.0
    return best / (best + other) * min(1.0, best / CONFIDENT_SCORE)


def _base_type(sig: _Signal) -> str | None:
    ranked = _ranked(sig.scores)
    if not ranked or ranked[0][1] < TYPE_MIN_SCORE:
        return None
    return ranked[0][0]


def _decide(sig: _Signal) -> None:
    ranked = _ranked(sig.scores)
    groups = _groups()
    best, best_score = ranked[0]
    reported = best
    members = {best}
    group = IDENTIFIER_TYPES.get(best)
    if group is not None:
        close = [
            t
            for t, s in ranked[1:]
            if t in groups[group] and best_score - s < GROUP_MARGIN
        ]
        if close:
            reported = group
            members = set(groups[group])
    outside = [(t, s) for t, s in ranked if t not in members]
    runner, runner_score = outside[0] if outside else (None, 0.0)
    reasons = [
        f"{_fact_text(f)} {w:+.2f}" for f, w in sig.contributions.get(best, []) if w
    ]
    if best_score < TYPE_MIN_SCORE:
        sig.object_type = "identifier"
        sig.confidence = sig.score
        sig.runner_up = best if best_score > 0 else None
        sig.runner_up_confidence = (
            _confidence(best_score, runner_score) if best_score > 0 else math.nan
        )
        missing = [
            _fact_text(f)
            for target, f, w in TYPE_RULES
            if w > 0
            and target in (best, IDENTIFIER_TYPES.get(best))
            and f not in sig.facts
        ][:3]
        sig.evidence.append(
            f"no type reached {TYPE_MIN_SCORE}: best {best} {best_score:.2f}"
            + (f"; missing: {', '.join(missing)}" if missing and best_score > 0 else "")
            + (
                "; no format cue"
                if not any(k in _KNOWN_FORMATS for k in sig.facts)
                else ""
            )
        )
        return
    sig.object_type = reported
    sig.confidence = _confidence(best_score, runner_score)
    sig.runner_up = runner if runner_score > 0 else None
    sig.runner_up_confidence = (
        _confidence(runner_score, best_score) if runner_score > 0 else math.nan
    )
    if reported != best:
        sig.evidence.append(
            f"{' / '.join(sorted(members))} not separable without a format cue: "
            f"reported as {reported}"
        )
    sig.evidence.append(f"{reported} {best_score:.2f}: " + "; ".join(reasons))


# ---------------------------------------------------------------------------
# Cross-signal structure
# ---------------------------------------------------------------------------


def _matched_share(a: _I64, b: _I64, tolerance_ns: int) -> float:
    """Share of the times ``a`` with a time of ``b`` within the tolerance."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pos = np.searchsorted(b, a)
    left = b[np.clip(pos - 1, 0, len(b) - 1)]
    right = b[np.clip(pos, 0, len(b) - 1)]
    near = np.minimum(np.abs(a - left), np.abs(right - a)) <= tolerance_ns
    return float(near.mean())


def _contained_share(child: _Signal, parent: _Signal) -> float:
    ps, pe = parent.p_start, parent.p_end
    cs, ce = child.p_start, child.p_end
    if len(ps) == 0 or len(cs) == 0:
        return 0.0
    k = np.searchsorted(ps, cs, side="right") - 1
    ok = k >= 0
    kk = np.clip(k, 0, len(ps) - 1)
    inside = ok & (cs >= ps[kk]) & (ce <= pe[kk])
    return float(inside.mean())


def _cross_signal(ids: list[_Signal], counters: list[_Signal]) -> None:
    tol = int(COCHANGE_TOLERANCE_SECONDS * 1e9)
    base = {s.uuid: _base_type(s) for s in ids}
    # nesting (held ids only: an event-stamped presence is a point in time)
    held = [
        s
        for s in ids
        if "event_stamped" not in s.facts
        and len(s.p_start)
        and base[s.uuid] not in _NOT_NESTED
    ]
    for child in held:
        best: _Signal | None = None
        hold = child.metrics.get("median_hold_seconds", math.nan)
        align_ns = int(
            max(
                COCHANGE_TOLERANCE_SECONDS,
                NEST_ALIGN_HOLD_SHARE * (0.0 if math.isnan(hold) else hold),
            )
            * 1e9
        )
        for parent in held:
            if parent is child:
                continue
            if len(parent.p_start) < 2:
                continue
            if len(child.p_start) < NEST_MIN_FANOUT * len(parent.p_start):
                continue
            if _contained_share(child, parent) < NEST_MIN_SHARE:
                continue
            lo, hi = child.p_start[0], child.p_end[-1]
            edges = parent.changes[(parent.changes > lo) & (parent.changes < hi)]
            if len(edges) == 0:
                continue
            if _matched_share(edges, child.changes, align_ns) < NEST_ALIGN_SHARE:
                continue
            if best is None or len(parent.p_start) > len(best.p_start):
                best = parent
        if best is not None:
            child.parent = best.uuid
    by_uuid = {s.uuid: s for s in ids}
    for sig in ids:
        depth, cur, seen = 0, sig, {sig.uuid}
        while cur.parent is not None and cur.parent not in seen:
            seen.add(cur.parent)
            depth += 1
            cur = by_uuid[cur.parent]
        sig.depth = depth
        if depth >= 1:
            sig.facts["nested_child"] = 1.0
        if depth >= 2:
            sig.facts["nested_depth2"] = 1.0
        if sig.parent is not None:
            parent = by_uuid[sig.parent]
            if base[sig.uuid] == "serial":
                parent.facts["parent_of_serial"] = 1.0
            if base[sig.uuid] in ("batch", "lot"):
                parent.facts["parent_of_batch"] = 1.0
    # co-changing event-stamped ids (an order and its partner)
    stamped = [s for s in ids if "event_stamped" in s.facts]
    for a in stamped:
        for b in stamped:
            if a is b or len(a.changes) < 2:
                continue
            if _matched_share(a.changes, b.changes, tol) < RATIO_HOLDS:
                continue
            if "never_revisited" in b.facts and "never_revisited" not in a.facts:
                a.facts["cochanges_with_unique"] = 1.0
                b.facts["cochanges_with_recurring"] = 1.0
    # changes aligned with a counter (one serial per counted part)
    for sig in ids:
        if len(sig.changes) < 3:
            continue
        for counter in counters:
            inc = counter.increments
            if len(inc) < 3:
                continue
            lo, hi = sig.changes[0], sig.changes[-1]
            within = inc[(inc >= lo) & (inc <= hi)]
            if len(within) == 0:
                continue
            ratio = len(sig.changes) / len(within)
            if (
                0.8 <= ratio <= 1.25
                and _matched_share(sig.changes, inc, tol) >= RATIO_HOLDS
            ):
                sig.facts["counter_aligned"] = 1.0
                break


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

REPORT_COLUMNS = [
    "uuid",
    "value_column",
    "role",
    "identifier_score",
    "object_type",
    "type_confidence",
    "runner_up_type",
    "runner_up_confidence",
    "sample_count",
    "distinct_count",
    "run_count",
    "mean_hold_seconds",
    "median_hold_seconds",
    "revisit_ratio",
    "changes_per_hour",
    "empty_ratio",
    "pattern",
    "pattern_share",
    "sequential_ratio",
    "recurrence_cv",
    "parent_uuid",
    "evidence",
]

_FLOAT_COLUMNS = (
    "identifier_score",
    "type_confidence",
    "runner_up_confidence",
    "mean_hold_seconds",
    "median_hold_seconds",
    "revisit_ratio",
    "changes_per_hour",
    "empty_ratio",
    "pattern_share",
    "sequential_ratio",
    "recurrence_cv",
)


@dataclass(frozen=True)
class ObjectReport:
    """Result of :func:`identify_objects`.

    Attributes:
        table: One row per signal (columns :data:`REPORT_COLUMNS`), sorted by
            uuid.
        type_scores: ``{uuid: {object_type: score}}`` for every identifier,
            before the decision (the raw evidence sums).

    Example:
        >>> import pandas as pd
        >>> from ts_shape.signals import identify_objects
        >>> t = pd.date_range("2025-03-01", periods=300, freq="1min")
        >>> df = pd.DataFrame({"systime": t, "uuid": "line:id",
        ...     "value_string": [f"SN{100000 + i // 3}" for i in range(300)]})
        >>> report = identify_objects(df)
        >>> report.table[["uuid", "role", "object_type"]]
              uuid        role object_type
        0  line:id  identifier      serial
    """

    table: pd.DataFrame
    type_scores: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """The per-signal table (a copy)."""
        return self.table.copy()

    def identifiers(self) -> pd.DataFrame:
        """The rows whose role is ``identifier``."""
        out = self.table[self.table["role"] == "identifier"]
        return out.reset_index(drop=True)

    def to_object_specs(
        self, min_confidence: float = 0.6, *, include_generic: bool = False
    ) -> list[ObjectSpec]:
        """Object specs for :func:`~ts_shape.eventlog.detect_objects`.

        Args:
            min_confidence: Keep identifiers whose ``type_confidence`` is at
                least this.
            include_generic: Also keep identifiers typed ``identifier`` (their
                confidence is the identifier score).

        Returns:
            One :class:`~ts_shape.eventlog.ObjectSpec` per kept identifier,
            in uuid order.
        """
        from ts_shape.eventlog.objects import ObjectSpec

        _register_vocabulary()
        specs = []
        for row in self.identifiers().itertuples(index=False):
            if row.object_type is None or not (row.type_confidence >= min_confidence):
                continue
            if row.object_type == "identifier" and not include_generic:
                continue
            specs.append(
                ObjectSpec(
                    uuid=str(row.uuid),
                    object_type=str(row.object_type),
                    value_column=str(row.value_column),
                )
            )
        return specs

    def explain(self, signal_uuid: str) -> str:
        """Why a signal got its role and type.

        Args:
            signal_uuid: The signal to explain.

        Returns:
            A few lines of text: role, type, runner-up, the measured
            features and each piece of evidence.

        Raises:
            KeyError: When ``signal_uuid`` is not in the report.
        """
        rows = self.table[self.table["uuid"] == str(signal_uuid)]
        if rows.empty:
            raise KeyError(
                f"signal {signal_uuid!r} is not in the report; "
                f"uuids: {self.table['uuid'].tolist()[:10]}"
            )
        r = rows.iloc[0]
        lines = [f"{r['uuid']}: {r['role']} ({r['value_column']})"]
        if r["role"] == "identifier":
            lines.append(
                f"  identifier score {r['identifier_score']:.2f}; type "
                f"{r['object_type']} (confidence {r['type_confidence']:.2f})"
                + (
                    f", runner-up {r['runner_up_type']} "
                    f"({r['runner_up_confidence']:.2f})"
                    if r["runner_up_type"] is not None
                    else ""
                )
            )
        lines.append(
            f"  {r['sample_count']} samples, {r['distinct_count']} distinct, "
            f"{r['run_count']} runs, median hold {r['median_hold_seconds']:.0f} s, "
            f"revisit {r['revisit_ratio']:.2f}, empty {r['empty_ratio']:.2f}, "
            f"pattern {r['pattern']} ({r['pattern_share']:.0%})"
        )
        lines.extend(f"  - {e}" for e in r["evidence"])
        scores = self.type_scores.get(str(r["uuid"]), {})
        top = [f"{t} {s:.2f}" for t, s in _ranked(scores)[:5] if s > 0]
        if top:
            lines.append("  type scores: " + ", ".join(top))
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable overview: roles, then each identifier and its type."""
        n = len(self.table)
        lines = [f"ts-shape object identification: {n} signal(s)"]
        if n:
            roles = self.table["role"].value_counts().to_dict()
            lines.append(
                "  roles: " + ", ".join(f"{v} {k}" for k, v in sorted(roles.items()))
            )
        for row in self.identifiers().itertuples(index=False):
            parent = f" in {row.parent_uuid}" if row.parent_uuid is not None else ""
            lines.append(
                f"  {row.uuid}: {row.object_type} "
                f"({row.type_confidence:.2f}){parent}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _row(sig: _Signal) -> dict[str, Any]:
    m = sig.metrics
    is_id = sig.role == "identifier"
    return {
        "uuid": sig.uuid,
        "value_column": sig.value_column,
        "role": sig.role,
        "identifier_score": sig.score,
        "object_type": sig.object_type if is_id else None,
        "type_confidence": sig.confidence if is_id else math.nan,
        "runner_up_type": sig.runner_up if is_id else None,
        "runner_up_confidence": sig.runner_up_confidence if is_id else math.nan,
        **m,
        "parent_uuid": sig.parent if is_id else None,
        "evidence": tuple(sig.screen_evidence + sig.evidence),
    }


def _empty_table() -> pd.DataFrame:
    table = pd.DataFrame({c: pd.Series(dtype=object) for c in REPORT_COLUMNS})
    for c in _FLOAT_COLUMNS:
        table[c] = table[c].astype(float)
    for c in ("sample_count", "distinct_count", "run_count"):
        table[c] = table[c].astype("int64")
    return table


def identify_objects(
    dataframe: pd.DataFrame,
    *,
    signal_uuids: Sequence[str] | None = None,
    hints: Mapping[str, str] | None = None,
    min_samples: int = 20,
    shift_definitions: Mapping[str, tuple[str, str]] | None = None,
    time_column: str = "systime",
    uuid_column: str = "uuid",
) -> ObjectReport:
    """Screen every signal for object identifiers and classify their type.

    Only the values and their behaviour over time are evidence; the uuid,
    labels and metadata are never read. See the module docstring for the
    two stages, and :data:`SCREEN_WEIGHTS` / :data:`TYPE_RULES` for the
    weights.

    Args:
        dataframe: Long-format frame (``systime``, ``uuid``, ``value_*``).
        signal_uuids: Screen only these signals; unknown ones give a
            :class:`~ts_shape.errors.DataQualityWarning`.
        hints: ``{uuid: object_type}`` pins a signal's type (and makes it an
            identifier). Types must be in :data:`IDENTIFIER_TYPES` or a group.
        min_samples: Signals with fewer rows are ``insufficient_data``.
        shift_definitions: Shift name -> ``("HH:MM", "HH:MM")``, for the
            operator evidence (changes at shift boundaries). Defaults to
            :data:`ts_shape.utils.shifts.DEFAULT_SHIFTS`.
        time_column: Timestamp column.
        uuid_column: Signal identifier column.

    Returns:
        :class:`ObjectReport` with one row per signal.

    Raises:
        ColumnNotFoundError: When ``uuid_column`` or ``time_column`` is missing.
        ValueError: When a hint names an unknown type.
    """
    require_columns(dataframe, (uuid_column, time_column), where="identify_objects")
    _register_vocabulary()
    hints = {str(k): str(v) for k, v in dict(hints or {}).items()}
    for key, value in hints.items():
        if not _known_target(value):
            raise ValueError(
                f"hint for {key!r}: unknown object type {value!r}; register it "
                "with register_identifier_type"
            )
    shifts = dict(shift_definitions or DEFAULT_SHIFTS)
    frame = dataframe
    if signal_uuids is not None or hints:
        codes, uniques = pd.factorize(frame[uuid_column])
        present = {str(u) for u in uniques}
        if signal_uuids is not None:
            wanted = {str(u) for u in signal_uuids}
            unknown = sorted(wanted - present)
            if unknown and len(frame):
                _warn_unknown("signal_uuids", unknown, present)
            keep = np.array([str(u) in wanted for u in uniques], dtype=bool)
            frame = frame[keep[codes] if len(codes) else np.zeros(0, bool)]
        else:
            wanted = present
        missing_hints = sorted(set(hints) - (present & wanted))
        if missing_hints and len(dataframe):
            _warn_unknown("hints", missing_hints, present)

    signals = [
        _screen(
            key,
            rows,
            time_column=time_column,
            min_samples=min_samples,
            shift_definitions=shifts,
            hinted=key in hints,
        )
        for key, rows in _iter_signal_groups(frame, uuid_column, time_column)
    ]
    for sig in signals:
        if sig.uuid in hints and sig.role != "identifier":
            if sig.metrics.get("distinct_count", 0):
                sig.screen_evidence.append(
                    f"screened as {sig.role}; hint makes it an identifier"
                )
                sig.role = "identifier"
    ids = [s for s in signals if s.role == "identifier"]
    for sig in ids:
        _score_types(sig)
    cross = sorted(
        ids, key=lambda s: (-(0 if math.isnan(s.score) else s.score), s.uuid)
    )
    counters = [s for s in signals if s.role == "counter"]
    _cross_signal(cross[:MAX_CROSS_SIGNALS], counters)
    for sig in ids:
        _score_types(sig)
        if sig.uuid in hints:
            sig.object_type = hints[sig.uuid]
            sig.confidence = 1.0
            ranked = [
                (t, s) for t, s in _ranked(sig.scores) if t != sig.object_type and s > 0
            ]
            sig.runner_up = ranked[0][0] if ranked else None
            sig.runner_up_confidence = (
                _confidence(ranked[0][1], sig.scores.get(sig.object_type, 0.0))
                if ranked
                else math.nan
            )
            sig.evidence.append(f"type pinned by hints: {sig.object_type}")
        else:
            _decide(sig)
        if sig.parent is not None:
            sig.evidence.append(f"nested in {sig.parent} (depth {sig.depth})")

    if not signals:
        return ObjectReport(table=_empty_table(), type_scores={})
    table = pd.DataFrame([_row(s) for s in signals], columns=REPORT_COLUMNS)
    for c in _FLOAT_COLUMNS:
        table[c] = table[c].astype(float)
    for c in ("sample_count", "distinct_count", "run_count"):
        table[c] = table[c].astype("int64")
    for c in (
        "value_column",
        "object_type",
        "runner_up_type",
        "pattern",
        "parent_uuid",
    ):
        table[c] = table[c].astype(object).where(table[c].notna(), None)
    scores = {s.uuid: dict(s.scores) for s in ids}
    return ObjectReport(table=table, type_scores=scores)


def _warn_unknown(name: str, unknown: list[str], present: set[str]) -> None:
    shown = sorted(present)[:10]
    more = f" (+{len(present) - 10} more)" if len(present) > 10 else ""
    warnings.warn(
        f"identify_objects: {name} {unknown} not found in the data; they are "
        f"not in the report. Available uuids: {shown}{more}",
        DataQualityWarning,
        stacklevel=caller_stacklevel(),
    )
