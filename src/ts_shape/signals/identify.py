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

Stage 0, clean-up
-----------------
Readers misread. Before anything is measured, spellings of one value that
differ only in case, blanks or leading zeros are merged; *misreads* -- rare
values that last a sample or two and do not fit the typical values (no-read
tokens, truncations, PLC defaults such as 0 or 65535) -- keep the previous
reading; and a single empty sample inside a hold is a dropped reading, not
"no item". The report counts the misreads in the evidence.

Stage 1, screening
------------------
Every signal is reduced to its run-length encoding and its distinct values
(O(n), one signal in memory at a time). A signal first meets the negative
roles, in this order: ``insufficient_data`` (fewer than ``min_samples`` rows,
or no non-empty reading), ``constant``, ``state`` (booleans),
``measurement`` (non-integral floats, or decimal numbers stored as text),
``timestamp`` (ISO and local date formats, compact dates, epoch numbers, and
any text a date parser reads as the moment it appears), ``counter``,
``measurement`` (a quantized analog value, or a number that changes at nearly
every sample with no format, no holding and no circulating pool), ``state``
(few word-like values, a small dense integer enumeration such as 1/2/3, or a
phase sequence that cycles in a fixed order within seconds), ``free_text``
(sentences, even when they embed ids). What is left gets an
``identifier_score`` from :data:`SCREEN_WEIGHTS`; at
:data:`IDENTIFIER_MIN_SCORE` or more the role is ``identifier``. Shares of a
format or shape are weighed by each value's runs, so a misread cannot
outvote a small pool.

Low-cardinality *codes* are told from *states* by format (letters with
digits, fixed-width numbers, sparse or large integers are codes; words and
dense small enumerations are states) and by behaviour: a code from a pool
that is present only briefly, in no fixed order and changing with no other
id, is a fault or alarm code and becomes a ``state`` after the cross-signal
stage.

Stage 2, identification
-----------------------
Facts about an identifier's values (formats, :data:`FACT_DESCRIPTIONS`) and
its behaviour (cardinality, holds, revisits, recurrence, a circulating
tour, alternation at cycle rate, shift alignment and shift-bounded holds,
the spread of holds) add the weights of :data:`TYPE_RULES` to the types they
point at. Cross-signal structure adds a second round of facts:

* containment -- each presence of A lies inside one presence of B and B's
  changes are matched by A's (a serial in its batch);
* attributes -- A changes only when B changes (a recipe or material with
  its work order, a customer with its order); whether B belongs to the
  production hierarchy (it nests batches or serials) tells production
  attributes from business partners, and a unique id that changes with a
  coarser unique id is a batch under its work order;
* part clocks -- a fixed count of serials (or counter increments) between
  changes is a package (handling unit) or its storage location; one change
  per counted part is a serial.

Change times are compared within :data:`COCHANGE_TOLERANCE_SECONDS` plus the
polling interval of the signals, and only where chance alone would not give
the match. The best type wins when it reaches :data:`TYPE_MIN_SCORE`;
otherwise the type is ``identifier`` and the evidence says what was missing.
When the two best types belong to one group of :data:`IDENTIFIER_TYPES` and
no format cue separates them, the group is reported (``order``,
``business_partner``).

Nesting
-------
On the types the values and the relations point at, each serial, batch and
lot gets a *production parent* (``parent_uuid``): for a serial its batch or
lot, else its order, material or recipe; for a batch or lot its material or
recipe, else its order (:data:`_PARENT_RANK`, :data:`_QUANTITY_PARENT_RANK`;
a farther level only when the data has no id of the nearest one). The
parent must nest the child both ways: the parent keeps its value during the
child's presences, and the parent's changes meet the child's starts, ends
or pauses far more often than chance would give (:data:`NEST_MIN_LIFT`,
:data:`NEST_MAX_CHANCE_P`), so an id on another line or clock is never a
parent. A handling unit (or any id holding a fixed count of parts) that
nests a serial is its *packing parent* (``packing_parent_uuid``), next to
the production parent. Resources and people (tools, carriers, operators,
stations, locations, partners) and messages that name an id are never
part of a nesting.

Attribute signals
-----------------
A signal of any role whose value changes only at an identifier's presence
boundaries, is constant within its presences and comes back with its
values (the customer of an order, the material of a work order, a
description text) gets ``attribute_of``, that identifier's uuid; the
identifier's own parents are not its attributes. See
:func:`_attribute_hosts`. :meth:`ObjectReport.to_object_specs` with
``attributes=True`` turns them into :attr:`ObjectSpec.attributes
<ts_shape.eventlog.ObjectSpec.attributes>`, :meth:`ObjectReport.hierarchy`
the nesting into ``part_of`` edges, and :meth:`ObjectReport.to_event_log`
runs the chain into an event log.

Confidence: ``best / (best + runner_up) * min(1, best / CONFIDENT_SCORE)``,
with ``runner_up`` the best score outside the reported type (or group).

Extend it with :func:`register_identifier_type` (new types and groups, also
registered as OCEL object types) and :func:`register_identifier_pattern`
(site-specific formats).
"""

from __future__ import annotations

import bisect
import functools
import math
import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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
    from ts_shape.eventlog.model import EventLog
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
    "recipe": "setup",  # recipe or product variant
    # orders
    "work_order": None,  # production order
    "sales_order": "order",
    "purchase_order": "order",
    # resources
    "tool": "setup",  # tool, mold or die
    "fixture": "setup",
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

#: Object types that are resources or people: no part of the containment
#: evidence for the types, as container or contained (a long-running tool
#: contains every batch without owning it).
_NOT_NESTED = frozenset(
    {"tool", "fixture", "setup", "carrier", "rfid_tag", "asset", "station", "location"}
    | {"operator", "customer", "supplier", "business_partner"}
)

#: Types that get a production parent in the reported nesting.
_NEST_CHILDREN = frozenset({"serial", "batch", "lot"})
#: Production parents of a serial, preferred in this order (lower first): its
#: batch or lot, else its order, else its material, else its recipe.
_PARENT_RANK: dict[str, int] = {
    "batch": 0,
    "lot": 0,
    "work_order": 1,
    "order": 1,
    "sales_order": 1,
    "purchase_order": 1,
    "material": 2,
    "article": 2,
    "recipe": 3,
    "setup": 3,
}
#: Production parents of a batch or lot. A batch is a quantity of one
#: material made by one recipe (batch numbers are kept per material; ISA-88
#: runs a batch on a recipe): its material or recipe -- the finer of them --
#: comes before the order that produced it. On its values alone a material
#: that was never revisited in the recording is easily taken for a lot; the
#: two-way alignment tells a refilled raw-material lot (its own clock) apart.
_QUANTITY_PARENT_RANK: dict[str, int] = {
    "material": 0,
    "article": 0,
    "recipe": 0,
    "setup": 0,
    "lot": 0,
    "work_order": 1,
    "order": 1,
    "sales_order": 1,
    "purchase_order": 1,
}
#: Packing parents: a package that holds parts, next to their production
#: parent (also any id that holds a fixed count of parts, see
#: ``fixed_count``).
_PACKING_TYPES = frozenset({"handling_unit"})

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
FORMAT_SAMPLE = 400
#: Timestamps in any format: up to CLOCK_SAMPLE values are parsed; a value
#: is a time when it lies within CLOCK_WINDOW_DAYS of its first appearance.
CLOCK_SAMPLE = 48
CLOCK_WINDOW_DAYS = 2.0
#: A one-sample value is a misread when the typical value has at least this
#: many times as many runs (and the value is foreign; see ``_clean_codes``).
MISREAD_RARITY = 4
#: ... and lasts at most this many consecutive samples.
MISREAD_MAX_SAMPLES = 2
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
#: Shuffled: the number in the value goes up at this share of changes
#: (random bins, random lots), so it is not a running number.
SHUFFLED_INCREASE = (0.3, 0.7)
#: Regular recurrence: ``recurrence_cv`` at most this (a pool of carriers
#: circulating); irregular alternation: at least IRREGULAR_RECURRENCE_CV.
REGULAR_RECURRENCE_CV = 0.7
IRREGULAR_RECURRENCE_CV = 0.8
#: Tour coverage: a window of as many runs as there are values shows at
#: least this share of the values (a circulating pool, not a wandering
#: measurement).
TOUR_COVERAGE = 0.8
#: Fixed successor: this share of transitions goes to each value's usual
#: successor (stations visited in order).
FIXED_SUCCESSOR = 0.75
#: A pool of at most this many values is tiny (recipes, tools).
TINY_POOL = 4
#: Cycle rate: the value changes about every machine cycle (median hold at
#: most this many seconds, and values come back).
CYCLE_RATE_SECONDS = 300.0
#: Hold-time spread: CV in this band is the spread of independent arrivals
#: (an order book). For one-sample stamps the spread of the gaps counts.
POISSON_HOLD_CV = (0.75, 1.6)
#: Within a shift: the longest hold is at most this factor times the longest
#: shift (and the 90th percentile at least half a shift).
WITHIN_SHIFT_SLACK = 1.15
#: Shift aligned: at least this share of the changes (and 3) lie at a shift
#: boundary. By chance about 6 % do (three boundaries, +-15 min).
SHIFT_ALIGNED_SHARE = 0.3
#: Sparse presence: a value is present less than this share of the time.
SPARSE_PRESENCE = 0.2
#: A phase sequence (a state): few values in a fixed order (successor share
#: at least PHASE_SUCCESSOR), each held less than PHASE_MAX_HOLD_SECONDS.
PHASE_MAX_HOLD_SECONDS = 15.0
PHASE_SUCCESSOR = 0.9
#: Event-stamped: this share of id runs is one sample cleared by an empty.
EVENT_STAMPED_SHARE = 0.8
#: A change within this many seconds of a shift boundary is aligned.
SHIFT_TOLERANCE_SECONDS = 900.0
#: Two changes within this many seconds happen together.
COCHANGE_TOLERANCE_SECONDS = 5.0
#: Reported nesting (:func:`_nest`): during this share of the child's
#: presences the parent keeps its value (the polling lag of either signal
#: may stick out), and the child has at least NEST_MIN_FANOUT times as many
#: presences ...
NEST_MIN_SHARE = 0.9
NEST_MIN_FANOUT = 1.25
#: ... and the parent's changes meet a change of the child -- a presence
#: starts or ends, nothing is present, or the child pauses (a presence of
#: NEST_PAUSE_HOLDS median holds or more) -- within
#: COCHANGE_TOLERANCE_SECONDS or NEST_ALIGN_HOLD_SHARE of the child's median
#: hold, widened by the polling lag: at least NEST_ALIGN_SHARE of them, a
#: lift ``(matched - chance) / (1 - chance)`` of NEST_MIN_LIFT or more over
#: the share of the time such windows cover, and a binomial tail of at most
#: NEST_MAX_CHANCE_P. Without the chance correction a parent on another
#: line would nest every serial: any moment is near a change of a fast
#: clock.
NEST_ALIGN_SHARE = 0.5
NEST_ALIGN_HOLD_SHARE = 0.1
NEST_MIN_LIFT = 0.4
NEST_MAX_CHANCE_P = 1e-2
NEST_PAUSE_HOLDS = 4.0
#: A preferred parent whose alignment polling hides replaces the verified
#: one when it lies in it with at least this fan-out (1:1, give or take).
NEST_ALIAS_FANOUT = 0.9
#: A signal is polled (seen up to one period late) when this share of its
#: sample intervals lies within 5 % of the median interval.
POLL_REGULAR_SHARE = 0.6
#: Attribute relation: this share of A's changes has a change of B within
#: COCHANGE_TOLERANCE_SECONDS plus the polling lag (capped at
#: MAX_LAG_SECONDS); B's changes would cover at most ATTRIBUTE_MAX_CHANCE of
#: the time by chance. A unique id is *coarser* than another when it changes
#: at most COARSER_RATIO times as often.
ATTRIBUTE_SHARE = 0.7
#: ... and A is a recurring *attribute* (a recipe, a customer) when its
#: revisit ratio is above this; a less recurring A is a coarser unique id
#: (a work order over its batches).
ATTRIBUTE_MIN_REVISIT = 0.2
ATTRIBUTE_MAX_CHANCE = 0.1
MAX_LAG_SECONDS = 900.0
COARSER_RATIO = 0.8
#: Fixed count: the parts counted between changes vary by at most this CV.
FIXED_COUNT_CV = 0.3
#: Containment evidence for the types (:func:`_containers`): share of the
#: child's presences inside one of the container's, fan-out, and share of
#: the container's changes matched by the child's.
CONTAIN_MIN_SHARE = 0.9
CONTAIN_MIN_FANOUT = 1.5
CONTAIN_ALIGN_SHARE = 0.8
#: Cross-signal work runs among at most this many identifiers.
MAX_CROSS_SIGNALS = 64
#: ``attribute_of`` (:func:`_attribute_hosts`): the share of a signal's value
#: changes that meet a presence boundary of the host, and of the host's
#: presences during which the signal keeps one value, and of the host's
#: revisited values that bring back the same value of the signal; at least
#: ATTRIBUTE_OF_MIN_CHANGES value changes.
ATTRIBUTE_OF_SHARE = 0.9
ATTRIBUTE_OF_MIN_CHANGES = 3
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
    "CH": ("batch", "lot"),  # German "Charge": batch or lot
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
    "unique_sequence": "4+ values in a row, none ever comes back (not a pool)",
    "cycle_hold": f"held about one cycle (median <= {CYCLE_MAX_HOLD_SECONDS:.0f} s)",
    "long_hold": f"long holds (median >= {LONG_HOLD_SECONDS:.0f} s)",
    "hours_hold": f"hours-long holds (median >= {HOURS_HOLD_SECONDS:.0f} s)",
    "very_long_hold": f"very long holds (median >= {VERY_LONG_HOLD_SECONDS:.0f} s)",
    "sequential": f"next value is 1..{SEQ_MAX_STEP} above (>= {RATIO_HOLDS:.0%})",
    "shuffled": "numbers go up and down at random (not a running number)",
    "increasing": f"values increase (>= {RATIO_HOLDS:.0%} of changes)",
    "regular_recurrence": (
        f"a fixed pool recurring regularly (recurrence CV <= {REGULAR_RECURRENCE_CV})"
    ),
    "fixed_successor": f"each value has a fixed successor (>= {FIXED_SUCCESSOR:.0%})",
    "event_stamped": "one-sample ids cleared by an empty reading",
    "shift_aligned": f"changes at shift boundaries (>= {SHIFT_ALIGNED_SHARE:.0%})",
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
    "tiny_pool": f"a tiny pool (<= {TINY_POOL} values)",
    "cycle_rate": f"changes about every cycle (median hold <= {CYCLE_RATE_SECONDS:.0f} s)",
    "irregular_alternation": (
        f"few values alternating at cycle rate in no fixed order "
        f"(recurrence CV >= {IRREGULAR_RECURRENCE_CV})"
    ),
    "poisson_holds": (
        f"holds spread like independent arrivals (CV {POISSON_HOLD_CV[0]}.."
        f"{POISSON_HOLD_CV[1]})"
    ),
    "within_shift": "never held longer than a shift",
    "cycle_alternation": (
        f"a tiny pool (<= {TINY_POOL}) taking turns every cycle (parallel resources)"
    ),
    "circulating": (
        f"a circulating pool: a tour of runs shows >= {TOUR_COVERAGE:.0%} of the values"
    ),
    "sparse_presence": f"present less than {SPARSE_PRESENCE:.0%} of the time",
    "attribute_of_unique": "changes only when a non-recurring id changes",
    "attribute_of_event": (
        "changes only when an event-like id outside production (an order) changes"
    ),
    "attribute_of_production": (
        "changes only when an id of the production hierarchy (a work order) changes"
    ),
    "plain_host": "has a recurring attribute but no production structure",
    "has_attribute": "a recurring id changes only when this one changes",
    "coarser_unique": "a finer non-recurring id changes whenever this one does",
    "finer_unique": "changes whenever a coarser non-recurring id changes",
    "unrelated": "changes with no other identifier",
    "fixed_count": f"a fixed count of parts per value (CV <= {FIXED_COUNT_CV})",
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
    ("serial", "finer_unique", 0.15),
    ("serial", "fixed_count", -0.50),  # a package of N serials
    ("serial", "gs1_serial", 1.20),
    ("serial", "long_hold", -0.30),
    ("serial", "often_revisited", -0.40),
    ("serial", "event_stamped", -0.20),
    # batch: contiguous holds of a production rhythm, inside a work order
    ("batch", "never_revisited", 0.20),
    ("batch", "long_hold", 0.30),
    ("batch", "increasing", 0.10),
    ("batch", "sequential", 0.15),
    ("batch", "unrelated", -0.15),
    ("batch", "shuffled", -0.20),
    ("batch", "nested_child", 0.15),
    ("batch", "parent_of_serial", 0.30),
    ("batch", "finer_unique", 0.35),
    ("batch", "date_coded", 0.70),
    ("batch", "gs1_batch", 1.20),
    ("batch", "cycle_hold", -0.40),
    ("batch", "often_revisited", -0.30),
    ("batch", "event_stamped", -0.30),
    ("batch", "fixed_count", -0.30),
    ("batch", "cycle_rate", -0.50),
    ("batch", "poisson_holds", -0.15),
    ("batch", "has_attribute", -0.10),
    ("batch", "coarser_unique", -0.30),
    # lot: long holds, occasionally revisited, independent of the order flow
    ("lot", "rarely_revisited", 0.10),
    ("lot", "occasionally_revisited", 0.25),
    ("lot", "long_hold", 0.25),
    ("lot", "unrelated", 0.30),
    ("lot", "date_coded", 0.60),
    ("lot", "gs1_batch", 1.00),
    ("lot", "cycle_hold", -0.20),
    ("lot", "event_stamped", -0.30),
    ("lot", "fixed_count", -0.30),
    ("lot", "cycle_rate", -0.50),
    ("lot", "poisson_holds", -0.10),
    ("lot", "increasing", -0.10),
    ("lot", "sequential", -0.10),
    # work order: hours-long holds; recipe and material change with it
    ("work_order", "hours_hold", 0.30),
    ("work_order", "occasionally_revisited", 0.15),
    ("work_order", "increasing", 0.10),
    ("work_order", "parent_of_serial", 0.15),
    ("work_order", "parent_of_batch", 0.30),
    ("work_order", "has_attribute", 0.15),
    ("work_order", "plain_host", -0.20),
    ("work_order", "coarser_unique", 0.40),
    ("work_order", "cycle_hold", -0.30),
    ("work_order", "often_revisited", -0.30),
    ("work_order", "event_stamped", -0.30),
    ("work_order", "fixed_count", -0.30),
    ("work_order", "cycle_rate", -0.50),
    ("work_order", "very_long_hold", -0.15),
    ("work_order", "poisson_holds", -0.20),
    ("work_order", "finer_unique", -0.20),
    ("work_order", "shuffled", -0.20),
    # orders: business events -- one record each, irregular arrivals, a
    # partner that changes with them
    ("order", "event_stamped", 0.35),
    ("order", "fixed_count", -0.30),
    ("order", "never_revisited", 0.20),
    ("order", "poisson_holds", 0.35),
    ("order", "has_attribute", 0.20),
    ("order", "plain_host", 0.30),
    ("order", "unrelated", -0.15),
    ("order", "shuffled", -0.20),
    ("order", "parent_of_batch", -0.30),
    ("order", "finer_unique", -0.30),
    ("order", "coarser_unique", -0.30),
    # recipe / material / article: few values, revisited, long holds; they
    # change with the work order
    ("recipe", "medium_cardinality", 0.15),
    ("recipe", "unique_sequence", -0.30),
    ("recipe", "tiny_pool", 0.20),
    ("recipe", "often_revisited", 0.30),
    ("recipe", "long_hold", 0.20),
    ("recipe", "attribute_of_unique", 0.15),
    ("recipe", "attribute_of_production", 0.30),
    ("recipe", "very_long_hold", -0.20),
    ("recipe", "event_stamped", -0.30),
    ("recipe", "shift_aligned", -0.30),
    ("recipe", "attribute_of_event", -0.40),
    ("recipe", "within_shift", -0.20),
    ("recipe", "cycle_rate", -0.40),
    ("recipe", "unrelated", -0.10),
    ("material", "medium_cardinality", 0.15),
    ("material", "unique_sequence", -0.30),
    ("material", "often_revisited", 0.30),
    ("material", "long_hold", 0.20),
    ("material", "attribute_of_unique", 0.15),
    ("material", "attribute_of_production", 0.30),
    ("material", "parent_of_batch", 0.35),
    ("material", "sap_material", 1.20),
    ("material", "tiny_pool", -0.10),
    ("material", "event_stamped", -0.30),
    ("material", "shift_aligned", -0.30),
    ("material", "attribute_of_event", -0.40),
    ("material", "within_shift", -0.20),
    ("material", "cycle_rate", -0.40),
    ("material", "unrelated", -0.10),
    ("article", "medium_cardinality", 0.15),
    ("article", "unique_sequence", -0.30),
    ("article", "often_revisited", 0.30),
    ("article", "long_hold", 0.20),
    ("article", "attribute_of_unique", 0.15),
    ("article", "attribute_of_production", 0.30),
    ("article", "parent_of_serial", 0.25),
    ("article", "gtin", 1.00),
    ("article", "gs1_gtin", 1.20),
    ("article", "tiny_pool", -0.10),
    ("article", "event_stamped", -0.30),
    ("article", "shift_aligned", -0.30),
    ("article", "attribute_of_event", -0.40),
    ("article", "within_shift", -0.20),
    ("article", "cycle_rate", -0.40),
    ("article", "unrelated", -0.10),
    # tool / fixture: very few values, very long holds, changed on their own
    # (a changeover follows the order change, it does not coincide with it)
    ("tool", "very_few_values", 0.15),
    ("tool", "unique_sequence", -0.30),
    ("tool", "tiny_pool", 0.25),
    ("tool", "very_long_hold", 0.40),
    ("tool", "often_revisited", 0.20),
    ("tool", "long_hold", 0.15),
    ("tool", "unrelated", 0.25),
    ("tool", "event_stamped", -0.30),
    ("tool", "shift_aligned", -0.30),
    ("tool", "hierarchical_code", -0.40),
    ("tool", "attribute_of_unique", -0.30),
    ("tool", "cycle_rate", -0.40),
    ("tool", "within_shift", -0.20),
    ("fixture", "very_few_values", 0.14),
    ("fixture", "unique_sequence", -0.30),
    ("fixture", "tiny_pool", 0.22),
    ("fixture", "very_long_hold", 0.36),
    ("fixture", "often_revisited", 0.18),
    ("fixture", "long_hold", 0.14),
    ("fixture", "unrelated", 0.22),
    ("fixture", "event_stamped", -0.30),
    ("fixture", "shift_aligned", -0.30),
    ("fixture", "hierarchical_code", -0.40),
    ("fixture", "attribute_of_unique", -0.30),
    ("fixture", "cycle_rate", -0.40),
    ("fixture", "within_shift", -0.20),
    # carrier: a fixed pool circulating regularly at cycle rate
    ("carrier", "regular_recurrence", 0.45),
    ("carrier", "unique_sequence", -0.30),
    ("carrier", "circulating", 0.30),
    ("carrier", "often_revisited", 0.10),
    ("carrier", "cycle_rate", 0.10),
    ("carrier", "positional_steps", -0.30),
    ("carrier", "very_long_hold", -0.30),
    ("carrier", "long_hold", -0.30),
    ("carrier", "tiny_pool", -0.30),
    ("carrier", "irregular_alternation", -0.30),
    ("carrier", "sparse_presence", -0.30),
    ("carrier", "cycle_alternation", -0.30),
    # rfid tag: EPC format
    ("rfid_tag", "epc_hex", 1.20),
    # ... usually on a carrier that circulates
    ("rfid_tag", "regular_recurrence", 0.30),
    ("rfid_tag", "circulating", 0.20),
    # asset / station / location: positional formats; parallel machines
    # alternate irregularly at cycle rate, stations follow a fixed order
    ("asset", "hierarchical_code", 0.60),
    ("asset", "very_few_values", 0.15),
    ("asset", "very_long_hold", 0.15),
    ("asset", "irregular_alternation", 0.40),
    ("asset", "cycle_alternation", 0.35),
    ("station", "positional_steps", 0.35),
    ("station", "few_values", 0.10),
    ("station", "fixed_successor", 0.30),
    ("station", "often_revisited", 0.10),
    ("station", "irregular_alternation", -0.30),
    ("station", "tiny_pool", -0.20),
    ("station", "cycle_alternation", -0.20),
    ("location", "location_code", 1.20),
    ("location", "fixed_count", 0.40),
    ("location", "shuffled", 0.20),
    ("location", "increasing", -0.30),
    ("location", "occasionally_revisited", 0.20),
    # operator: changes at shift boundaries, never logged in beyond a shift
    ("operator", "shift_aligned", 0.55),
    ("operator", "unique_sequence", -0.30),
    ("operator", "within_shift", 0.45),
    ("operator", "rarely_revisited", -0.30),
    ("operator", "medium_cardinality", 0.10),
    ("operator", "often_revisited", 0.10),
    ("operator", "event_stamped", -0.30),
    ("operator", "cycle_rate", -0.40),
    ("operator", "attribute_of_unique", -0.20),
    # partners: recurring values that change with an order
    ("business_partner", "event_stamped", 0.30),
    ("business_partner", "unique_sequence", -0.30),
    ("business_partner", "often_revisited", 0.25),
    ("business_partner", "attribute_of_unique", 0.20),
    ("business_partner", "attribute_of_event", 0.50),
    ("business_partner", "attribute_of_production", -0.40),
    ("business_partner", "gtin", -0.50),  # a trade item number, not a partner
    ("business_partner", "sap_material", -0.50),
    ("business_partner", "poisson_holds", 0.20),
    ("business_partner", "cycle_rate", -0.40),
    # handling unit: SSCC
    ("handling_unit", "sscc", 1.20),
    ("handling_unit", "gs1_sscc", 1.20),
    ("handling_unit", "event_stamped", 0.10),
    ("handling_unit", "never_revisited", 0.10),
    ("handling_unit", "rarely_revisited", 0.10),
    ("handling_unit", "fixed_count", 0.45),
    ("handling_unit", "increasing", 0.20),
    ("handling_unit", "shuffled", -0.30),
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
_LOCATION_LEVELS = re.compile(
    r"[A-Z]{1,2}\d{1,3}(?:([-./])[A-Z]{1,2}\d{1,3})(?:\1[A-Z]{1,2}\d{1,3}){1,2}"
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
_INT_SENTINELS = frozenset(
    {"0", "-1", "32767", "-32768", "65535", "2147483647", "4294967295"}
)
_DECIMAL_TEXT = re.compile(r"[+-]?\d+[.,]\d+")
_FAMILY_DIGITS = re.compile(r"\d+")
_SEGMENT_RUNS = re.compile(r"[A-Za-z0-9]+")
_FAMILY_SEPARATORS = re.compile(r"[-_./:]")


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
    if len(head) < 4 or (int(head[:2]) - seen.year) % 100 not in (0, 1, 99):
        return False  # cheap pre-check: this, last or next year
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


def _format_facts(
    texts: list[str],
    seen: list[pd.Timestamp],
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Share of ``texts`` that match each format (only shares > 0).

    ``seen`` is when each value first appears (for date codes); ``weights``
    (default 1 each) weigh each value, by its number of runs.
    """
    n = len(texts)
    if n == 0:
        return {}
    w = [1.0] * n if weights is None else [float(x) for x in weights]
    total = sum(w) or 1.0
    counts: defaultdict[str, float] = defaultdict(float)
    gs1_ais: defaultdict[str, float] = defaultdict(float)
    gs1_n = 0.0
    epc_letters = 0.0
    for text, first, wt in zip(texts, seen, w, strict=True):
        ais = _gs1_parse(text) if len(text) >= 16 or text.startswith("(") else None
        if ais:
            gs1_n += wt
            for ai in ais:
                gs1_ais[ai] += wt
            continue
        if text.isdigit():
            # an integer column drops leading zeros; the GS1 check digit does
            # not depend on them
            padded = text.zfill(18) if 15 <= len(text) < 18 else text
            if len(text) == 18 and _SAP.fullmatch(text):
                counts["sap_material"] += wt
            elif _DIGITS18.fullmatch(padded) and _mod10_ok(padded):
                counts["sscc"] += wt
            elif _GTIN.fullmatch(text) and _mod10_ok(text):
                counts["gtin"] += wt
        if _EPC.fullmatch(text):
            counts["epc_hex"] += wt
            if not text.isdigit() or text.startswith(_EPC_HEADERS):
                epc_letters += wt
        if _GUID.fullmatch(text):
            counts["guid"] += wt
        if _LOCATION.fullmatch(text) or _LOCATION_LEVELS.fullmatch(text):
            counts["location_code"] += wt
        if _HIERARCHICAL.fullmatch(text):
            counts["hierarchical_code"] += wt
        if _date_coded(text, first):
            counts["date_coded"] += wt
    if epc_letters < 0.5 * counts.get("epc_hex", 0):
        counts.pop("epc_hex", None)
    shares = {k: v / total for k, v in counts.items()}
    if gs1_n:
        share = gs1_n / total
        for ai, fact in (("00", "gs1_sscc"), ("21", "gs1_serial"), ("10", "gs1_batch")):
            if gs1_ais[ai] >= FORMAT_MIN_SHARE * gs1_n:
                shares[fact] = share
                break
        else:
            if gs1_ais["01"] >= FORMAT_MIN_SHARE * gs1_n:
                shares["gs1_gtin"] = share
    for pattern in _PATTERNS.values():
        hits = sum(wt for text, wt in zip(texts, w) if _safe(pattern.test, text))
        if hits:
            shares[f"pattern:{pattern.name}"] = hits / total
    lead: defaultdict[str, float] = defaultdict(float)
    for t, wt in zip(texts, w):
        found = _LEAD_LETTERS.match(t)
        lead[found.group(1).upper() if found else ""] += wt
    token, hits = max(lead.items(), key=lambda kv: kv[1])
    if token in PREFIX_FAMILIES and token != "":
        rest_ok = sum(wt for t, wt in zip(texts, w) if len(t) > len(token)) / total
        if rest_ok >= FORMAT_MIN_SHARE:
            shares[f"prefix:{token}"] = hits / total
    return shares


def _safe(test: Callable[[str], bool], text: str) -> bool:
    try:
        return bool(test(text))
    except Exception:  # a site pattern must not break the screening
        return False


_FINE_TABLE = str.maketrans(
    "0123456789" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz",
    "9" * 10 + "A" * 26 + "a" * 26,
)
_COARSE_TABLE = str.maketrans(
    "0123456789" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz",
    "X" * 62,
)
_REPEATS = re.compile(r"(.)\1+", re.DOTALL)


def _count_run(match: re.Match[str]) -> str:
    return f"{match.group(1)}{{{len(match.group(0))}}}"


@functools.lru_cache(maxsize=65536)
def _shape(text: str, coarse: bool = False) -> str:
    """Shape signature: ``A`` upper, ``a`` lower, ``9`` digit, other kept;
    a run of one class is written once with its length (``A{2}-9{3}``).

    ``coarse`` maps every ASCII letter and digit to ``X``.
    """
    mapped = text.translate(_COARSE_TABLE if coarse else _FINE_TABLE)
    return _REPEATS.sub(_count_run, mapped)


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
    # value code of each presence
    p_code: _I64 = field(default_factory=lambda: np.empty(0, np.int64))
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
    packing_parent: str | None = None
    # the identifier whose presences this signal's value describes
    attribute_of: str | None = None
    depth: int = 0
    distinct: int = 0
    runs: int = 0
    lag_ns: int = 0
    # sampling period of a polled signal (regular sample times), else 0
    poll_ns: int = 0
    # measured features behind the facts (hold spread, time presence, ...)
    extra: dict[str, Any] = field(default_factory=dict)


def _id_text(value: object) -> str:
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _canonical_key(text: str) -> str:
    """Reading-independent spelling: trimmed, upper case, no leading zeros
    on an all-digit value (``"vr-415"`` is ``"VR-415"``, ``"0000140630"`` is
    ``"140630"``)."""
    key = text.strip().upper()
    if key.isdigit():
        key = key.lstrip("0") or "0"
    return key


def _clean_codes(
    codes: _I64, uniques: pd.Index, *, text: bool
) -> tuple[_I64, list[str], int]:
    """Merge spellings of one value and drop misreads.

    Returns the new codes (``-1`` empty), the text of each code (its most
    frequent spelling) and the number of misread values dropped.

    1. Spellings that differ only in case, surrounding blanks or leading
       zeros (all-digit values) are one value.
    2. A *misread* is a value that never lasts more than
       :data:`MISREAD_MAX_SAMPLES` samples, is rare --
       fewer than 1 / :data:`MISREAD_RARITY` of the runs of the typical
       value (run-weighted median) -- and is foreign to the typical values:
       it has no letter or digit, is a PLC default number (0, -1,
       2**15-1, 2**16-1, ...), has a different shape than every typical
       value, or is a truncation (strict prefix) of one. A misread
       sample keeps the previous reading (a no-read during a hold does not
       change what is there; after an empty it stays empty).
    3. In an integer column, a PLC default (0, -1, 2**15-1, 2**16-1, ...) in
       a shape no typical value has is a misread too, however long it lasts.
    """
    raw_texts = [_id_text(u) for u in uniques]
    if not raw_texts:
        return codes, [], 0
    keys = [_canonical_key(x) for x in raw_texts] if text else raw_texts
    kcodes, kuniq = pd.factorize(pd.Index(keys, dtype=object))
    kcodes = np.asarray(kcodes, dtype=np.int64)
    k = len(kuniq)
    present = codes >= 0
    new = np.full(len(codes), -1, dtype=np.int64)
    new[present] = kcodes[codes[present]]
    if k == len(raw_texts):
        labels = raw_texts
    else:
        counts = np.bincount(codes[present], minlength=len(raw_texts))
        best = np.full(k, -1, dtype=np.int64)
        best_n = np.full(k, -1, dtype=np.int64)
        for i, kc in enumerate(kcodes):
            if counts[i] > best_n[kc]:
                best[kc], best_n[kc] = i, counts[i]
        labels = [raw_texts[i] for i in best]
    # first-pass runs per value
    n = len(new)
    if n == 0:
        return new, labels, 0
    brk = np.ones(n, dtype=bool)
    brk[1:] = new[1:] != new[:-1]
    idx = np.flatnonzero(brk)
    rc = new[idx]
    rlen = np.diff(np.append(idx, n))
    ok = rc >= 0
    runs_per = np.bincount(rc[ok], minlength=k)
    maxlen = np.zeros(k, dtype=np.int64)
    np.maximum.at(maxlen, rc[ok], rlen[ok])
    samples_per = np.bincount(new[new >= 0], minlength=k)
    typical_runs = _weighted_median(runs_per)
    typical_samples = _weighted_median(samples_per)
    if typical_runs is None or typical_samples is None:
        return new, labels, 0
    noise = np.zeros(k, dtype=bool)
    present_codes = np.flatnonzero(runs_per > 0)
    if not text:
        # PLC defaults (0, -1, 2**15-1, 2**16-1, ...) in a shape no other
        # value has
        sentinel = [c for c in present_codes if labels[c] in _INT_SENTINELS]
        if sentinel:
            others = [c for c in present_codes if labels[c] not in _INT_SENTINELS]
            shapes = {_shape(labels[c]) for c in others[:FORMAT_SAMPLE]}
            for c in sentinel:
                noise[c] = bool(shapes) and _shape(labels[c]) not in shapes
    cand = np.flatnonzero(
        (maxlen <= MISREAD_MAX_SAMPLES)
        & (runs_per > 0)
        & ~noise
        & (
            (runs_per * MISREAD_RARITY < typical_runs)
            | (samples_per * MISREAD_RARITY < typical_samples)
        )
    )
    if len(cand):
        is_core = (runs_per * MISREAD_RARITY >= typical_runs) & (
            samples_per * MISREAD_RARITY >= typical_samples
        )
        core = sorted(labels[c].strip().upper() for c in np.flatnonzero(is_core))
        core_shapes = {_shape(x) for x in core}
        for c in cand:
            x = labels[c].strip().upper()
            pos = bisect.bisect_right(core, x)
            noise[c] = (
                not any(ch.isalnum() for ch in x)
                or _canonical_key(x) in _INT_SENTINELS
                or _shape(x) not in core_shapes
                or (pos < len(core) and core[pos].startswith(x))
            )
    if not noise.any():
        return new, labels, 0
    bad = (new >= 0) & noise[new.clip(min=0)]
    keep = np.maximum.accumulate(np.where(bad, -1, np.arange(n)))
    new = np.where(keep >= 0, new[keep.clip(min=0)], -1)
    return new, labels, int(noise.sum())


def _fill_dropouts(codes: _I64) -> _I64:
    """Fill one-sample gaps inside a hold: ``A A _ A`` is a dropped reading,
    not "no item". Only a single empty sample between two runs of the same
    value, at least one of them two samples or longer, is filled -- a
    one-sample id cleared by an empty (``A _ A _``) keeps its empties."""
    n = len(codes)
    if n < 3:
        return codes
    brk = np.ones(n, dtype=bool)
    brk[1:] = codes[1:] != codes[:-1]
    idx = np.flatnonzero(brk)
    rc = codes[idx]
    rlen = np.diff(np.append(idx, n))
    if len(rc) < 3:
        return codes
    gap = (rc[1:-1] == -1) & (rlen[1:-1] == 1) & (rc[:-2] == rc[2:]) & (rc[:-2] >= 0)
    gap &= (rlen[:-2] >= 2) | (rlen[2:] >= 2)
    if not gap.any():
        return codes
    out = codes.copy()
    pos = idx[1:-1][gap]
    out[pos] = codes[pos - 1]
    return out


def _weighted_median(counts: _I64) -> int | None:
    """The count of the value holding the middle unit, values ranked by count
    (a value that has more runs or samples weighs more)."""
    order = np.sort(counts)
    cum = np.cumsum(order)
    if not len(cum) or cum[-1] == 0:
        return None
    return int(order[np.searchsorted(cum, cum[-1] / 2)])


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


def _tour_coverage(v_codes: _I64, distinct: int, windows: int = 50) -> float:
    """Median share of distinct values in a window of ``distinct`` runs.

    A circulating pool (carriers on a loop) shows nearly every value once
    per tour: about 1. A measurement wandering over its levels, or a few
    values alternating, revisits a handful of nearby levels: much less.
    """
    n = len(v_codes)
    if distinct < 3 or n < 2 * distinct:
        return math.nan
    starts = np.unique(np.linspace(0, n - distinct, windows).astype(np.intp))
    shares = [len(np.unique(v_codes[i : i + distinct])) / distinct for i in starts]
    return float(np.median(shares))


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


def _poll_period(t_ns: _I64) -> int:
    """The sampling period when the samples lie on a regular grid (at least
    :data:`POLL_REGULAR_SHARE` of the intervals within 5 % of the median),
    else 0. A polled value is seen up to one period after it changed."""
    if len(t_ns) < 4:
        return 0
    gaps = np.diff(t_ns).astype(np.float64)
    med = float(np.median(gaps))
    if med <= 0:
        return 0
    regular = float((np.abs(gaps - med) <= 0.05 * med).mean())
    return int(med) if regular >= POLL_REGULAR_SHARE else 0


@functools.lru_cache(maxsize=32)
def _shift_bounds(definitions: tuple[tuple[str, str], ...]) -> npt.NDArray[np.float64]:
    """Shift boundaries as seconds after midnight."""
    bounds = []
    for start, end in definitions:
        for clock in (start, end):
            t = pd.to_datetime(clock).time()
            bounds.append(t.hour * 3600 + t.minute * 60 + t.second)
    return np.unique(np.asarray(bounds, dtype=np.float64))


def _shift_alignment(
    changes: _I64, tz: Any, shift_definitions: Mapping[str, tuple[str, str]]
) -> float:
    if len(changes) == 0:
        return math.nan
    idx = pd.DatetimeIndex(changes.view("datetime64[ns]"))
    if tz is not None:
        idx = idx.tz_localize("UTC").tz_convert(tz)
    tod = (idx - idx.normalize()).total_seconds().to_numpy()
    b = _shift_bounds(tuple(shift_definitions.values()))
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
        labels = [_id_text(u) for u in uniques]
        misreads = 0
    else:
        codes, uniques = _id_codes(raw)
        codes, labels, misreads = _clean_codes(codes, uniques, text=not numeric)
        codes = _fill_dropouts(codes)

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
    sig.p_start, sig.p_end, sig.p_code = p_start, p_end, rc[is_id]
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
    if nonempty > 2 and runs < 0.5 * nonempty:
        # polled: a change is seen up to one sampling interval late
        sig.lag_ns = int(np.median(np.diff(ne_t)))
    sig.poll_ns = _poll_period(t)
    one_cleared = (p_len == 1) & followed_empty
    event_stamped = len(p_len) >= 3 and float(one_cleared.mean()) >= EVENT_STAMPED_SHARE
    held = len(p_len) > 0 and float(np.median(p_len)) >= 2

    # sampled texts of the distinct values, in order of first appearance
    sample_pos = _sample_positions(distinct)
    sample_codes = used[sample_pos]
    texts = [labels[c] for c in sample_codes]
    m = len(texts)
    # each value weighs by its runs, so a misread that survived the clean-up
    # cannot outvote a pool of a few real values
    weights = np.bincount(v_codes, minlength=len(labels))[sample_codes].astype(float)
    wsum = float(weights.sum()) or 1.0

    def share(flags: Iterable[bool]) -> float:
        return float(np.dot(np.fromiter(flags, dtype=float, count=m), weights)) / wsum

    def top(keys: Iterable[str]) -> tuple[str, float]:
        tally: defaultdict[str, float] = defaultdict(float)
        for key, wt in zip(keys, weights):
            tally[key] += float(wt)
        key, hits = max(tally.items(), key=lambda kv: kv[1])
        return key, hits / wsum

    pattern, pattern_share = top(_shape(x) for x in texts)
    _, coarse = top(_shape(x, coarse=True) for x in texts)
    _, family = top(_family(x) for x in texts)
    segment, segments = top(_segments(x) for x in texts)
    text_regularity = max(pattern_share, coarse)
    regularity = max(
        text_regularity,
        family,
        segments if _FAMILY_SEPARATORS.search(segment) else 0.0,
    )
    lengths = np.array([len(x) for x in texts], dtype=float)
    length_cv = float(lengths.std() / lengths.mean()) if lengths.mean() > 0 else 0.0
    spaced = share(" " in x.strip() for x in texts)
    word_like = share(_WORD_LIKE.fullmatch(x) is not None for x in texts)
    min_len = 1 if numeric else 2
    token = share((" " not in x) and min_len <= len(x) <= 40 for x in texts)
    decimal_text = share(_DECIMAL_TEXT.fullmatch(x) is not None for x in texts)
    sentences = share(_sentence_like(x) for x in texts)

    pair_pos = _sample_positions(max(runs - 1, 0))
    pair_codes = np.empty(2 * len(pair_pos), dtype=np.int64)
    pair_codes[0::2] = v_codes[pair_pos]
    pair_codes[1::2] = v_codes[pair_pos + 1]
    numbers = [_digits_number(labels[c]) for c in pair_codes]
    seq_ratio, inc_ratio, small_ratio, down_ratio = _pair_ratios(numbers)
    rec_cv, successor = _recurrence(v_codes, v_start)
    coverage = _tour_coverage(v_codes, distinct)
    # the rhythm of an id: how long each value is held, or for one-sample
    # stamps the time between stamps
    rhythm = np.diff(p_start) / 1e9 if event_stamped else holds
    hold_cv = (
        float(rhythm.std() / rhythm.mean())
        if len(rhythm) >= 3 and rhythm.mean() > 0
        else math.nan
    )
    time_presence = (
        float(min(1.0, (p_end - p_start).sum() / 1e9 / span)) if span > 0 else 1.0
    )
    shift_share = _shift_alignment(sig.changes, sig.tz, shift_definitions)
    shift_len = _shift_length(tuple(shift_definitions.values()))
    max_hold = float(holds.max()) if len(holds) else math.nan
    q90_hold = float(np.quantile(holds, 0.9)) if len(holds) else math.nan
    # a resource number alternating every cycle in no fixed order (which of
    # the parallel machines or nests made the part) is not a state
    alternating = (
        not event_stamped
        and median_hold <= CYCLE_RATE_SECONDS
        and revisit >= 0.9
        and not math.isnan(rec_cv)
        and rec_cv >= IRREGULAR_RECURRENCE_CV
        and not successor >= FIXED_SUCCESSOR
    )

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
    sig.extra = {
        "misreads": misreads,
        "hold_cv": hold_cv,
        "successor": successor,
        "regularity": regularity,
        "word_like": word_like,
        "time_presence": time_presence,
        "shift_share": shift_share,
        "max_hold": max_hold,
        "q90_hold": q90_hold,
        "coverage": coverage,
    }
    if misreads:
        sig.screen_evidence.append(
            f"{misreads} misread value(s) dropped (no-reads, truncations, PLC "
            "defaults)"
        )

    seen = [pd.Timestamp(int(x)) for x in first_seen[sample_pos]]
    formats = _format_facts(texts, seen, weights)
    known = sorted(
        k
        for k, v in formats.items()
        if v >= FORMAT_MIN_SHARE and (k in _KNOWN_FORMATS or k.startswith("pattern:"))
    )

    def facts() -> dict[str, float]:
        out = {k: v for k, v in formats.items() if v >= FORMAT_MIN_SHARE}
        often = revisit > OFTEN_REVISITED
        flags = {
            "high_cardinality": distinct >= HIGH_CARDINALITY,
            "medium_cardinality": 2 <= distinct <= MEDIUM_VALUES,
            "few_values": distinct <= FEW_VALUES,
            "very_few_values": distinct <= VERY_FEW_VALUES,
            "tiny_pool": distinct <= TINY_POOL,
            "never_revisited": revisit <= NEVER_REVISITED,
            "rarely_revisited": revisit <= RARELY_REVISITED,
            "occasionally_revisited": NEVER_REVISITED < revisit <= OFTEN_REVISITED,
            "often_revisited": often,
            "unique_sequence": runs >= 4 and revisit <= NEVER_REVISITED,
            "event_stamped": event_stamped,
            "cycle_hold": not event_stamped and median_hold <= CYCLE_MAX_HOLD_SECONDS,
            "cycle_rate": (
                not event_stamped and median_hold <= CYCLE_RATE_SECONDS and often
            ),
            "long_hold": not event_stamped and median_hold >= LONG_HOLD_SECONDS,
            "hours_hold": not event_stamped and median_hold >= HOURS_HOLD_SECONDS,
            "very_long_hold": (
                not event_stamped and median_hold >= VERY_LONG_HOLD_SECONDS
            ),
            "sequential": seq_ratio >= RATIO_HOLDS,
            "increasing": inc_ratio >= RATIO_HOLDS,
            "shuffled": (
                distinct >= 10
                and SHUFFLED_INCREASE[0] <= inc_ratio <= SHUFFLED_INCREASE[1]
            ),
            "regular_recurrence": (
                distinct >= 3
                and often
                and not math.isnan(rec_cv)
                and rec_cv <= REGULAR_RECURRENCE_CV
            ),
            "irregular_alternation": distinct <= VERY_FEW_VALUES and alternating,
            "cycle_alternation": (
                distinct <= TINY_POOL
                and not event_stamped
                and median_hold <= CYCLE_RATE_SECONDS
                and revisit >= 0.9
            ),
            "fixed_successor": (
                runs >= 2 * distinct
                and not math.isnan(successor)
                and successor >= FIXED_SUCCESSOR
            ),
            "positional_steps": 2 <= distinct <= FEW_VALUES and _positional(texts),
            "poisson_holds": (
                not math.isnan(hold_cv)
                and POISSON_HOLD_CV[0] <= hold_cv <= POISSON_HOLD_CV[1]
                and runs >= 8
            ),
            "within_shift": (
                runs >= 4
                and not math.isnan(max_hold)
                and max_hold <= WITHIN_SHIFT_SLACK * shift_len
                and q90_hold >= 0.5 * shift_len
            ),
            "shift_aligned": (
                len(sig.changes) >= 3
                and not math.isnan(shift_share)
                and shift_share >= SHIFT_ALIGNED_SHARE
                and shift_share * len(sig.changes) >= 3
            ),
            "sparse_presence": time_presence < SPARSE_PRESENCE,
            "circulating": (
                distinct > STATE_MAX_CARDINALITY and coverage >= TOUR_COVERAGE
            ),
        }
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
    if decimal_text >= FORMAT_MIN_SHARE and not known:
        return role("measurement", f"{decimal_text:.0%} of values are decimal numbers")
    num_values: npt.NDArray[np.float64] | None = None
    if numeric:
        code_value = np.array([float(x) for x in labels], dtype=np.float64)
        num_values = code_value[ne_codes]
    stamp_share = share(_ISO_TIME.fullmatch(x) is not None for x in texts)
    if stamp_share >= FORMAT_MIN_SHARE:
        return role("timestamp", f"{stamp_share:.0%} of values parse as dates/times")
    compact = _compact_date_share(texts, seen, weights)
    if compact >= FORMAT_MIN_SHARE:
        return role(
            "timestamp", f"{compact:.0%} of values are compact dates of the sample day"
        )
    clock = _clock_share(texts, seen, weights) if not numeric else 0.0
    if clock >= FORMAT_MIN_SHARE:
        return role(
            "timestamp",
            f"{clock:.0%} of values read as a date/time at the moment they appear",
        )
    epoch_values = num_values
    if epoch_values is None and all(x.isdigit() and 9 <= len(x) <= 19 for x in texts):
        epoch_values = np.array(
            [float(labels[c]) for c in ne_codes[:: max(1, nonempty // FORMAT_SAMPLE)]]
        )
        epoch_t = ne_t[:: max(1, nonempty // FORMAT_SAMPLE)]
    else:
        epoch_t = ne_t
    if epoch_values is not None and _is_epoch_like(epoch_values, epoch_t):
        return role("timestamp", "numbers track the sample time (epoch)")
    if num_values is not None:
        series = pd.Series(num_values)
        is_counter, why = _is_counter(series)
        if not is_counter:
            is_counter, why = _counts_and_resets(num_values, revisit)
        if is_counter and not _serial_like(texts, seq_ratio, series):
            inc = np.flatnonzero(np.diff(num_values) > 0) + 1
            sig.increments = ne_t[inc]
            return role("counter", f"counter: {why}")
        if (
            distinct > STATE_MAX_CARDINALITY
            and not coverage >= TOUR_COVERAGE
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
    small_enum = (
        num_values is not None
        and _small_enum(code_value[used])
        and not (_positional(texts) and distinct >= 3)
        and not alternating
    )
    low = distinct <= STATE_MAX_CARDINALITY and not known
    if low and (word_like >= STATE_WORD_SHARE or small_enum):
        what = (
            "word-like values"
            if word_like >= STATE_WORD_SHARE
            else "a small enumeration of integers"
        )
        return role(
            "state", f"{distinct} {what} (state band <= {STATE_MAX_CARDINALITY})"
        )
    if (
        low
        and not event_stamped
        and median_hold < PHASE_MAX_HOLD_SECONDS
        and not math.isnan(successor)
        and successor >= PHASE_SUCCESSOR
    ):
        return role(
            "state",
            f"{distinct} values cycling in a fixed order every "
            f"{median_hold:.0f} s (a phase or step sequence)",
        )
    if sentences >= FREE_TEXT_SPACE_SHARE and not known:
        return role(
            "free_text",
            f"{sentences:.0%} of values are sentences (2+ plain words), even "
            "when they embed ids",
        )
    if (
        spaced >= FREE_TEXT_SPACE_SHARE
        and (
            length_cv >= FREE_TEXT_LENGTH_CV or lengths.mean() >= FREE_TEXT_MEAN_LENGTH
        )
        and text_regularity < FREE_TEXT_MAX_REGULARITY
    ):
        return role(
            "free_text",
            f"{spaced:.0%} of values contain spaces, length CV {length_cv:.2f}, "
            f"shape regularity {text_regularity:.0%}",
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
    if (
        num_values is not None
        and runs >= EVERY_SAMPLE_SHARE * nonempty
        and distinct > STATE_MAX_CARDINALITY
        and not known
        and not seq_ratio >= RATIO_HOLDS
        and not coverage >= TOUR_COVERAGE
    ):
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


def _family(text: str) -> str:
    """Shape family: letter runs kept (upper case), digit runs collapsed, so
    ``M-7`` and ``M-58`` are one family (``M-#``)."""
    return _FAMILY_DIGITS.sub("#", text.upper())


def _sentence_like(text: str) -> bool:
    """Two or more words, at least two of them plain words of three or more
    letters (``Batch B24-008 released``, ``Door open``)."""
    words = text.split()
    if len(words) < 2:
        return False
    plain = sum(
        1
        for w in words
        if len(w.strip(",.:;!?()")) >= 3 and w.strip(",.:;!?()").isalpha()
    )
    return plain >= 2


def _segments(text: str) -> str:
    """Segment structure: every letter/digit run is ``X``, separators kept
    (``RZ_PP_GF30_v5`` and ``RZ_PBT_T20_v9`` are both ``X_X_X_X``)."""
    return _SEGMENT_RUNS.sub("X", text)


def _small_enum(values: npt.NDArray[np.float64]) -> bool:
    """A small, dense enumeration of integers (1/2/3, 0..7): it starts at -1,
    0 or 1, ends at 99 or less, and the values fill at least half of their
    range. Integer *codes* are sparse (17, 23, 5), large (4711) or start
    elsewhere (9..13)."""
    if not len(values):
        return False
    lo, hi = float(values.min()), float(values.max())
    if lo < -1 or lo > 1 or hi > 99:
        return False
    return hi - lo + 1 <= 2 * len(values) + 1


def _compact_date_share(
    texts: Sequence[str],
    seen: Sequence[pd.Timestamp],
    weights: Sequence[float] | None = None,
) -> float:
    """Share of ``texts`` that are compact dates ``YYYYMMDD`` (optionally with
    ``hhmm[ss]``) within :data:`DATE_CODE_MAX_AGE_DAYS` of when they appear,
    or ``YYMMDD`` of the day they appear on."""
    w = [1.0] * len(texts) if weights is None else [float(x) for x in weights]
    hits = 0.0
    for text, first, wt in zip(texts, seen, w, strict=True):
        if not (text.isdigit() and len(text) in (6, 8, 12, 14)):
            continue
        short = len(text) == 6  # YYMMDD: only the sample day itself
        year = 2000 + int(text[:2]) if short else int(text[:4])
        rest = text[2:] if short else text[4:]
        if not 1970 <= year <= 2200:
            continue
        try:
            day = pd.Timestamp(year=year, month=int(rest[:2]), day=int(rest[2:4]))
        except ValueError:
            continue
        if len(text) > 8 and not (int(text[8:10]) < 24 and int(text[10:12]) < 60):
            continue
        age = (first.normalize() - day).days
        if -1 <= age <= (1 if short else DATE_CODE_MAX_AGE_DAYS):
            hits += wt
    return hits / (sum(w) or 1.0) if texts else 0.0


_CLOCK_CANDIDATE = re.compile(r"(?=(?:\D*\d){6})(?=.*[\s.,:/T-])[\d\s.,:/TZ+-]{6,40}")


def _clock_share(
    texts: Sequence[str],
    seen: Sequence[pd.Timestamp],
    weights: Sequence[float] | None = None,
    k: int = CLOCK_SAMPLE,
) -> float:
    """Share of ``texts`` that a date parser reads as a time within
    :data:`CLOCK_WINDOW_DAYS` of the moment the value first appears (or, for
    a date alone, the day it appears up to :data:`DATE_CODE_MAX_AGE_DAYS`
    before). Values that follow the clock are timestamps, whatever their
    format; an id that merely contains a date code does not parse whole.
    Only short texts of digits and date punctuation are tried."""
    w = [1.0] * len(texts) if weights is None else [float(x) for x in weights]
    triples = list(zip(texts, seen, w, strict=True))[:k]
    if not triples:
        return 0.0
    total = sum(wt for *_, wt in triples) or 1.0
    shaped = sum(wt for x, _, wt in triples if _CLOCK_CANDIDATE.fullmatch(x))
    if shaped < FORMAT_MIN_SHARE * total:
        return 0.0
    hits = 0.0
    for text, first, wt in triples:
        if not _CLOCK_CANDIDATE.fullmatch(text):
            continue
        for dayfirst in (False, True):
            # a parse far from the recording (``"T3-0000123"`` reads as the
            # year 123) is out of the nanosecond range: not a date
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    when = pd.Timestamp(pd.to_datetime(text, dayfirst=dayfirst))
                if when.tzinfo is not None:
                    when = when.tz_convert(None)
                ref = first.tz_convert(None) if first.tzinfo is not None else first
                age = (ref - when).total_seconds() / 86400
            except (ValueError, OverflowError, TypeError):
                # OutOfBoundsDatetime / OutOfBoundsTimedelta are ValueErrors
                continue
            if when == when.normalize():
                ok = -1 <= age <= DATE_CODE_MAX_AGE_DAYS
            else:
                ok = abs(age) <= CLOCK_WINDOW_DAYS
            if ok:
                hits += wt
                break
    return hits / total


@functools.lru_cache(maxsize=32)
def _shift_length(definitions: tuple[tuple[str, str], ...]) -> float:
    """The longest shift, in seconds."""
    longest = 0.0
    for start, end in definitions:
        a, b = (pd.to_datetime(c) for c in (start, end))
        length = (b - a).total_seconds() % 86400 or 86400.0
        longest = max(longest, length)
    return longest or 8 * 3600.0


def _positional(texts: Sequence[str]) -> bool:
    nums = [_digits_number(x) for x in texts]
    valid = [x for x in nums if x is not None]
    if len(valid) < FORMAT_MIN_SHARE * len(texts) or not valid:
        return False
    return sum(1 for x in valid if x % 10 == 0 and x > 0) >= FORMAT_MIN_SHARE * len(
        valid
    )


def _counts_and_resets(
    values: npt.NDArray[np.float64], revisit: float
) -> tuple[bool, str]:
    """A count that restarts: the value steps up by 1..SEQ_MAX_STEP at nearly
    every change, falls rarely, and the same numbers come back after each
    restart (a serial never comes back). Catches counters whose resets were
    polled late (140 -> 76) and so fail the strict reset test."""
    steps = np.diff(values)
    steps = steps[steps != 0]
    if len(steps) < 20 or not revisit > OFTEN_REVISITED:
        return False, ""
    up = float((steps > 0).mean())
    small = float(((steps >= 1) & (steps <= SEQ_MAX_STEP)).mean())
    if up >= 0.9 and small >= RATIO_HOLDS:
        return True, f"counts up by small steps ({small:.0%}) and restarts"
    return False, ""


def _serial_like(texts: Sequence[str], seq_ratio: float, values: pd.Series) -> bool:
    """A monotonic integer that is a serial rather than a counter.

    Every value has the same number of digits (5 or more) and the value
    changes by +1..+SEQ_MAX_STEP (or only ever rises, at 20 changes or
    more), without a reset.
    """
    widths = {len(x) for x in texts}
    steps = values.diff().dropna()
    steps = steps[steps != 0]
    return (
        len(widths) == 1
        and min(widths) >= 5
        and (seq_ratio >= RATIO_HOLDS or (len(steps) >= 20 and bool((steps > 0).all())))
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


def _aligned(
    a: _I64,
    b: _I64,
    tol_ns: int,
    lag_ns: int,
    rate: float,
    share: float,
    max_chance: float = ATTRIBUTE_MAX_CHANCE,
) -> bool:
    """Whether ``share`` of the times ``a`` have a time of ``b`` nearby.

    First within ``tol_ns``; then, for polled signals, within ``tol_ns +
    lag_ns`` -- unless ``b`` happens so often (``rate`` per ns) that chance
    alone would put a time of ``b`` near more than ``max_chance`` of all
    times.
    """
    if _matched_share(a, b, tol_ns) >= share:
        return True
    if lag_ns <= 0 or rate * 2 * (tol_ns + lag_ns) > max_chance:
        return False
    return _matched_share(a, b, tol_ns + lag_ns) >= share


def _value_changes(sig: _Signal) -> _I64:
    """When ``sig`` takes a new value (not the same value again after an
    empty reading)."""
    new = np.asarray(sig.p_code[1:] != sig.p_code[:-1], dtype=bool)
    out: _I64 = sig.p_start[1:][new]
    return out


def _containment(child: _Signal, parent: _Signal, slack_ns: int = 0) -> float:
    """Share of the child's presences during which the parent keeps its
    value (no change of the parent's value more than ``slack_ns``, the
    polling lag, inside them), among those that overlap a parent presence
    at all; 0 when fewer than half do. A child presence while the parent is
    empty (not recorded) is no evidence either way."""
    ps, pe = parent.p_start, parent.p_end
    cs, ce = child.p_start, child.p_end
    if len(ps) == 0 or len(cs) == 0:
        return 0.0
    vc = _value_changes(parent)
    lo, hi = cs + slack_ns, ce - slack_ns
    crossed = np.searchsorted(vc, hi, side="left") > np.searchsorted(
        vc, lo, side="right"
    )
    crossed &= hi > lo
    # overlapping: a parent presence starts before the child ends and ends
    # after it starts
    j = np.searchsorted(ps, ce, side="left") - 1
    overlaps = (j >= 0) & (pe[np.clip(j, 0, len(ps) - 1)] > cs)
    k = np.searchsorted(ps, cs, side="right") - 1
    overlaps |= (k >= 0) & (pe[np.clip(k, 0, len(ps) - 1)] > cs)
    if overlaps.sum() < 0.5 * len(cs):
        return 0.0
    return float((overlaps & ~crossed).sum() / overlaps.sum())


def _change_windows(
    sig: _Signal, before_ns: int, after_ns: int, *, pauses_only: bool = False
) -> tuple[_I64, _I64]:
    """Disjoint ``(starts, ends)`` of the moments that meet a change of
    ``sig``: a presence starts or ends from ``before_ns`` before to
    ``after_ns`` after them, or nothing is present (no presence can straddle
    a change then), or a presence lasts far longer than usual (a pause: the
    line stood still). ``pauses_only`` leaves out the starts and ends."""
    # a boundary seen up to ``after_ns`` after the change it meets
    bounds = np.concatenate([sig.p_start[1:], sig.p_end[:-1]])
    if pauses_only:
        bounds = bounds[:0]
    lo, hi = bounds - after_ns, bounds + before_ns
    gap = sig.p_end[:-1] < sig.p_start[1:]
    lo = np.concatenate([lo, sig.p_end[:-1][gap] - after_ns])
    hi = np.concatenate([hi, sig.p_start[1:][gap] + before_ns])
    # a pause: a presence held far longer than usual (the line stood still)
    hold = sig.p_end - sig.p_start
    pause = hold > NEST_PAUSE_HOLDS * float(np.median(hold)) if len(hold) else hold
    lo = np.concatenate([lo, sig.p_start[pause] - after_ns])
    hi = np.concatenate([hi, sig.p_end[pause] + before_ns])
    if len(lo) == 0:
        return lo, hi
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], np.maximum.accumulate(hi[order])
    new = np.ones(len(lo), dtype=bool)
    new[1:] = lo[1:] > hi[:-1]
    group = np.cumsum(new) - 1
    ends = np.zeros(int(group[-1]) + 1, dtype=np.int64)
    np.maximum.at(ends, group, hi)
    return lo[new], ends


def _cochange_lift(
    edges: _I64,
    child: _Signal,
    before_ns: int,
    after_ns: int,
    lo: int,
    hi: int,
    *,
    pauses_only: bool = False,
) -> float:
    """How far above chance the ``edges`` (a parent's changes) meet a change
    of the child (:func:`_change_windows`): ``(matched - chance) / (1 -
    chance)``, with ``chance`` the share of ``[lo, hi]`` the windows cover;
    ``-inf`` when fewer than :data:`NEST_ALIGN_SHARE` of the edges match."""
    if len(edges) == 0 or hi <= lo:
        return -math.inf
    ws, we = _change_windows(child, before_ns, after_ns, pauses_only=pauses_only)
    if len(ws) == 0:
        return -math.inf
    k = np.searchsorted(ws, edges, side="right") - 1
    hit = (k >= 0) & (edges <= we[np.clip(k, 0, len(we) - 1)])
    matched = float(hit.mean())
    if matched < NEST_ALIGN_SHARE:
        return -math.inf
    covered = float((np.minimum(we, hi) - np.maximum(ws, lo)).clip(min=0).sum())
    chance = min(1.0, covered / float(hi - lo))
    if chance >= 1.0:
        return -math.inf
    if _binomial_tail(int(hit.sum()), len(edges), chance) > NEST_MAX_CHANCE_P:
        return -math.inf
    return (matched - chance) / (1.0 - chance)


def _binomial_tail(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``."""
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * lp
        + (n - i) * lq
        for i in range(k, n + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


_WORDY_SHAPE = re.compile(r"(?:^|\s)a\{(?:[3-9]|\d\d)\}|\sAa\{[2-9]\}")


def _wordy(sig: _Signal) -> bool:
    """Whether the values read like a message around an id (``"Batch
    25061207 released"``): blanks between words. Such a signal follows the
    id it names, so it is kept out of the nesting."""
    shape = sig.metrics.get("pattern")
    return isinstance(shape, str) and " " in shape and bool(_WORDY_SHAPE.search(shape))


def _late_ns(sig: _Signal) -> int:
    """How late a change of ``sig`` may be seen: its polling lag."""
    return min(max(sig.lag_ns, sig.poll_ns), int(MAX_LAG_SECONDS * 1e9))


def _align_ns(sig: _Signal) -> int:
    """Tolerance for meeting a change of ``sig``: :data:`NEST_ALIGN_HOLD_SHARE`
    of its median hold, at least :data:`COCHANGE_TOLERANCE_SECONDS`."""
    hold = sig.metrics.get("median_hold_seconds", math.nan)
    hold = 0.0 if math.isnan(hold) else hold
    return int(max(COCHANGE_TOLERANCE_SECONDS, NEST_ALIGN_HOLD_SHARE * hold) * 1e9)


def _inside(child: _Signal, parent: _Signal, fanout: float = NEST_MIN_FANOUT) -> bool:
    """The child's presences lie inside the parent's: at least
    :data:`NEST_MIN_SHARE` of them inside one parent presence, and
    ``fanout`` times as many of them."""
    if parent is child or len(parent.p_start) < 2:
        return False
    if len(child.p_start) < fanout * len(parent.p_start):
        return False
    # a polled reading is late by up to one period (give or take jitter)
    slack = int(1.05 * max(_late_ns(child), _late_ns(parent)))
    slack += int(COCHANGE_TOLERANCE_SECONDS * 1e9)
    return _containment(child, parent, slack) >= NEST_MIN_SHARE


def _cochange(child: _Signal, parent: _Signal) -> float:
    """How far above chance the parent's changes meet a change of the child
    (a presence starts or ends, none is present, or the child pauses): the
    best :func:`_cochange_lift` over the windows tried, ``-inf`` when none
    passes. A parent on another line or clock co-varies with the child only
    by chance. A polled signal is seen up to one sampling period late, so
    the window reaches that far: after the parent's change for a polled
    child, before it for a polled parent. Tried within
    :data:`COCHANGE_TOLERANCE_SECONDS` and within
    :data:`NEST_ALIGN_HOLD_SHARE` of the child's median hold."""
    lo, hi = int(child.p_start[0]), int(child.p_end[-1])
    vc = _value_changes(parent)
    edges = vc[(vc > lo) & (vc < hi)]
    if len(edges) == 0:
        return -math.inf
    lift = -math.inf
    for tol in sorted({int(COCHANGE_TOLERANCE_SECONDS * 1e9), _align_ns(child)}):
        before, after = tol + _late_ns(parent), tol + _late_ns(child)
        lift = max(
            lift,
            _cochange_lift(edges, child, tol, tol, lo, hi),
            _cochange_lift(edges, child, before, after, lo, hi),
            # polled too coarsely for the starts and ends: the pauses tell
            _cochange_lift(edges, child, before, after, lo, hi, pauses_only=True),
        )
    return lift


def _cochanges(child: _Signal, parent: _Signal) -> bool:
    """Whether the parent co-varies with the child (:func:`_cochange`)."""
    return _cochange(child, parent) >= NEST_MIN_LIFT


def _nest(ids: list[_Signal], base: Mapping[str, str | None]) -> None:
    """Choose each id's production parent and packing parent.

    Children are the ids typed :data:`_NEST_CHILDREN` (serials, batches,
    lots); their production parents are the types of :data:`_PARENT_RANK`
    (:data:`_QUANTITY_PARENT_RANK` for a batch or lot), preferred in that
    order; among equals the finer id (:func:`_finer_key`). A parent must
    nest the child both ways: the child's presences lie inside the
    parent's (:func:`_inside`) and the parent's changes meet the child's
    far above chance (:func:`_cochanges`) -- an id on another line or clock
    does not. A farther level is taken only when the data has no id of the
    nearest one (a serial goes under its order only without any batch or
    lot). A handling unit, or any id that holds a fixed count of parts,
    that nests a serial is its packing parent. Messages that name an id
    (:func:`_wordy`) and event-stamped ids are left out.

    Where polling hides when the child changed (a serial read every half
    cycle, a batch polled every few minutes), the co-change test cannot
    decide. A preferred parent the child lies inside then still wins over
    the verified one when it nests in that verified parent itself: both
    belong to the line the verified parent puts the child on.
    """
    held = [
        s
        for s in ids
        if "event_stamped" not in s.facts and len(s.p_start) and not _wordy(s)
    ]
    for child in held:
        kind = base[child.uuid]
        if kind not in _NEST_CHILDREN:
            continue
        ranks = _QUANTITY_PARENT_RANK if kind in ("batch", "lot") else _PARENT_RANK
        verified: list[tuple[int, _Signal]] = []
        unverified: list[tuple[int, _Signal]] = []
        packing: list[_Signal] = []
        for parent in held:
            ptype = base[parent.uuid]
            if ptype is None or ptype == kind:
                continue
            production = ptype in ranks
            packs = kind == "serial" and (
                ptype in _PACKING_TYPES or "fixed_count" in parent.facts
            )
            if not (production or packs) or not _inside(child, parent):
                continue
            if not _cochanges(child, parent):
                if production:
                    unverified.append((ranks[ptype], parent))
                continue
            if production:
                verified.append((ranks[ptype], parent))
            if packs:
                packing.append(parent)
        if kind in ("batch", "lot"):
            # an order that holds another order is no order: it is the
            # material the orders make (a material never revisited in the
            # recording looks like an order on its own values)
            orders = [p for r, p in verified if r == ranks["work_order"]]
            verified = [
                (
                    (0, p)
                    if any(o is not p and _inside(o, p, 1.0) for o in orders)
                    else (r, p)
                )
                for r, p in verified
            ]
        if verified:
            rank, best = min(verified, key=lambda rp: (rp[0], _finer_key(rp[1])))
            for r, cand in sorted(
                unverified, key=lambda rp: (rp[0], _finer_key(rp[1]))
            ):
                if (
                    r <= rank
                    and _inside(cand, best, NEST_ALIAS_FANOUT)
                    and _cochanges(cand, best)
                ):
                    rank, best = r, cand
                    break
            nearest = {t for t, r in ranks.items() if r == 0}
            if rank == 0 or not any(
                base[s.uuid] in nearest for s in held if s is not child
            ):
                child.parent = best.uuid
        packing = [p for p in packing if p.uuid != child.parent]
        if packing:
            child.packing_parent = min(
                packing,
                key=lambda p: (base[p.uuid] not in _PACKING_TYPES, _finer_key(p)),
            ).uuid


def _finer_key(sig: _Signal) -> tuple[bool, int, int, str]:
    """Sort key among equally preferred parents: not a package (a fixed
    count of parts) first, then the finer id -- more distinct values (a
    material under its recipe), then more presences."""
    return ("fixed_count" in sig.facts, -sig.distinct, -len(sig.p_start), sig.uuid)


def _containers(ids: list[_Signal], base: Mapping[str, str | None]) -> dict[str, str]:
    """Type evidence of nesting: each held id's finest container.

    A container holds the child's presences (:data:`CONTAIN_MIN_SHARE`,
    :data:`CONTAIN_MIN_FANOUT` times as many) and its changes are matched by
    the child's (:data:`CONTAIN_ALIGN_SHARE`); of several, the one with the
    most presences. This is looser than the reported nesting
    (:func:`_nest`) -- a serial lies in its handling unit as well as in its
    batch -- and it is what the nesting facts of :data:`TYPE_RULES`
    (``nested_child``, ``parent_of_serial``, ``parent_of_batch``) were
    weighed on.

    Returns:
        ``{child uuid: container uuid}``.
    """
    held = [
        s
        for s in ids
        if "event_stamped" not in s.facts
        and len(s.p_start)
        and base[s.uuid] not in _NOT_NESTED
    ]
    out: dict[str, str] = {}
    for child in held:
        best: _Signal | None = None
        align_ns = _align_ns(child)
        for parent in held:
            if parent is child or base[parent.uuid] == "serial":
                continue
            if len(parent.p_start) < 2:
                continue
            if len(child.p_start) < CONTAIN_MIN_FANOUT * len(parent.p_start):
                continue
            lag = min(max(child.lag_ns, parent.lag_ns), int(MAX_LAG_SECONDS * 1e9))
            if _contained_share(child, parent, lag) < CONTAIN_MIN_SHARE:
                continue
            lo, hi = child.p_start[0], child.p_end[-1]
            edges = parent.changes[(parent.changes > lo) & (parent.changes < hi)]
            if len(edges) == 0:
                continue
            rate = len(child.changes) / max(hi - lo, 1)
            if not _aligned(
                edges, child.changes, align_ns, lag, rate, CONTAIN_ALIGN_SHARE
            ):
                continue
            if best is None or len(parent.p_start) > len(best.p_start):
                best = parent
        if best is not None:
            out[child.uuid] = best.uuid
    return out


def _contained_share(child: _Signal, parent: _Signal, slack_ns: int = 0) -> float:
    """Share of the child's presences inside one presence of the parent
    (each end may stick out by ``slack_ns``, the polling lag)."""
    ps, pe = parent.p_start, parent.p_end
    cs, ce = child.p_start, child.p_end
    if len(ps) == 0 or len(cs) == 0:
        return 0.0
    k = np.searchsorted(ps, cs + slack_ns, side="right") - 1
    ok = k >= 0
    kk = np.clip(k, 0, len(ps) - 1)
    inside = ok & (cs + slack_ns >= ps[kk]) & (ce - slack_ns <= pe[kk])
    return float(inside.mean())


def _depth(uuid: str, parent_of: Mapping[str, str | None]) -> int:
    depth, cur, seen = 0, uuid, {uuid}
    while (nxt := parent_of.get(cur)) is not None and nxt not in seen:
        seen.add(nxt)
        depth += 1
        cur = nxt
    return depth


def _cross_signal(
    ids: list[_Signal],
    counters: list[_Signal],
    hints: Mapping[str, str] | None = None,
) -> None:
    tol = int(COCHANGE_TOLERANCE_SECONDS * 1e9)
    base = {s.uuid: _base_type(s) for s in ids}
    by_uuid = {s.uuid: s for s in ids}
    # containment evidence for the types (the reported nesting follows at
    # the end)
    container = _containers(ids, base)
    for sig in ids:
        depth = _depth(sig.uuid, container)
        if depth >= 1:
            sig.facts["nested_child"] = 1.0
        if depth >= 2:
            sig.facts["nested_depth2"] = 1.0
        if sig.uuid in container:
            holder = by_uuid[container[sig.uuid]]
            if base[sig.uuid] == "serial":
                holder.facts["parent_of_serial"] = 1.0
            if base[sig.uuid] in ("batch", "lot"):
                holder.facts["parent_of_batch"] = 1.0
    # attributes: A changes only when B changes -- a recipe or material with
    # its work order, a customer with its order. The tolerance covers the
    # polling lag of both signals; a relation that chance alone would give
    # (B changing so often that any time is near a change) is not counted.
    attributes: list[tuple[_Signal, _Signal]] = []
    for a in ids:
        if len(a.changes) < 3:
            continue
        a_rev = a.metrics.get("revisit_ratio", math.nan)
        for b in ids:
            if a is b or len(b.changes) < len(a.changes):
                continue
            lag = min(max(a.lag_ns, b.lag_ns), int(MAX_LAG_SECONDS * 1e9))
            span = float(b.changes[-1] - b.changes[0])
            if span <= 0 or len(b.changes) * 2 * tol / span > ATTRIBUTE_MAX_CHANCE:
                continue
            rate = len(b.changes) / span
            if not _aligned(a.changes, b.changes, tol, lag, rate, ATTRIBUTE_SHARE):
                continue
            stamped = "event_stamped" in a.facts or "event_stamped" in b.facts
            b_rev = b.metrics.get("revisit_ratio", math.nan)
            if not b_rev <= RARELY_REVISITED:
                continue
            if a_rev > ATTRIBUTE_MIN_REVISIT:
                a.facts["attribute_of_unique"] = 1.0
                b.facts["has_attribute"] = 1.0
                attributes.append((a, b))
            elif not stamped and len(a.changes) <= COARSER_RATIO * len(b.changes):
                a.facts["coarser_unique"] = 1.0
                b.facts["finer_unique"] = 1.0
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
    # a fixed count of parts per value (a package of N parts): count the
    # changes of a part clock -- a serial-like id or a counter -- between
    # consecutive changes
    # (counters only without a serial-like id: a counter may count strokes
    # or shots, not parts)
    clocks = [s.changes for s in ids if base[s.uuid] == "serial"]
    clocks = clocks or [c.increments for c in counters]
    for sig in ids:
        if len(sig.changes) < 4:
            continue
        for clock in clocks:
            if clock is sig.changes or len(clock) < 4 * len(sig.changes):
                continue
            parts = np.diff(np.searchsorted(clock, sig.changes)).astype(float)
            if parts.mean() >= 2 and parts.std() <= FIXED_COUNT_CV * parts.mean():
                sig.facts["fixed_count"] = 1.0
                break
    # whose attribute: a host inside the production hierarchy (a work order
    # over batches or serials) has production attributes -- recipe,
    # material; a host outside it that arrives like events (an order) has a
    # business partner
    for a, b in attributes:
        if any(f in b.facts for f in _PRODUCTION_FACTS):
            a.facts["attribute_of_production"] = 1.0
        else:
            b.facts["plain_host"] = 1.0
            if "poisson_holds" in b.facts or "event_stamped" in b.facts:
                a.facts["attribute_of_event"] = 1.0
    holders = set(container.values())
    for sig in ids:
        alone = sig.uuid not in container and sig.uuid not in holders
        if len(sig.changes) >= 3 and alone:
            if not any(f in sig.facts for f in _RELATION_FACTS):
                sig.facts["unrelated"] = 1.0
    # the reported nesting, on the types the values and the relations to
    # other ids point at (a work order that looked like a batch on its own
    # values is a coarser unique id by now), without the containment
    # evidence it is to decide afresh
    # (a hinted type is taken as given)
    pinned = dict(hints or {})
    _nest(
        ids,
        {
            s.uuid: pinned.get(s.uuid) or _base_type_without(s, _CONTAINMENT_FACTS)
            for s in ids
        },
    )
    parent_of = {s.uuid: s.parent for s in ids}
    for sig in ids:
        sig.depth = _depth(sig.uuid, parent_of)


#: Facts that come from the containment evidence (:func:`_containers`).
_CONTAINMENT_FACTS = frozenset(
    {
        "nested_child",
        "nested_depth2",
        "parent_of_serial",
        "parent_of_batch",
        "attribute_of_production",
        "plain_host",
        "attribute_of_event",
        "unrelated",
    }
)


def _base_type_without(sig: _Signal, facts: frozenset[str]) -> str | None:
    """The best type on the facts other than ``facts`` (scores are restored)."""
    kept = sig.facts
    scores, contributions = sig.scores, sig.contributions
    sig.facts = {k: v for k, v in kept.items() if k not in facts}
    try:
        _score_types(sig)
        return _base_type(sig)
    finally:
        sig.facts = kept
        sig.scores, sig.contributions = scores, contributions


#: Facts that put an id inside the production hierarchy.
_PRODUCTION_FACTS = (
    "coarser_unique",
    "finer_unique",
    "parent_of_batch",
    "parent_of_serial",
)

#: Cross-signal facts; a signal with none of them is ``unrelated``.
_RELATION_FACTS = (
    "attribute_of_unique",
    "has_attribute",
    "coarser_unique",
    "finer_unique",
    "counter_aligned",
)


def _is_fault_code(sig: _Signal) -> str | None:
    """Why a screened identifier is a fault/alarm code (a state), or None.

    A code from a recurring pool that is present only briefly -- held for a
    few readings at a stop, or stamped once -- and changes with no other id,
    in no fixed order and with no known format: what a fault or alarm code
    does. An order's partner changes with the order; a circulating carrier
    keeps a fixed order.
    """
    f = sig.facts
    if not ("sparse_presence" in f and "often_revisited" in f):
        return None
    if any(k in _KNOWN_FORMATS or k.startswith("pattern:") for k in f):
        return None
    if "fixed_successor" in f or "circulating" in f:
        return None
    if any(k in f for k in _RELATION_FACTS) or "nested_child" in f:
        return None
    share = sig.extra.get("time_presence", math.nan)
    return (
        f"a recurring code present {share:.0%} of the time, in no fixed order, "
        "changing with no other id: a fault or alarm code (state)"
    )


def _ancestors(sig: _Signal, by_uuid: Mapping[str, _Signal]) -> set[str]:
    """The production and packing parents of ``sig``, and theirs."""
    out: set[str] = set()
    todo = [sig]
    while todo:
        cur = todo.pop()
        for up in (cur.parent, cur.packing_parent):
            if up is not None and up not in out:
                out.add(up)
                if up in by_uuid:
                    todo.append(by_uuid[up])
    return out


def _one_value_per_host_value(attr: _Signal, host: _Signal) -> bool:
    """Whether a revisited host value brings back the same attribute value:
    the attribute is a function of the host's value (an order keeps its
    customer). Read at the middle of each host presence; true when the host
    revisits no value."""
    if len(attr.p_start) == 0:
        return False
    mid = host.p_start + (host.p_end - host.p_start) // 2
    k = np.searchsorted(attr.p_start, mid, side="right") - 1
    kk = np.clip(k, 0, len(attr.p_start) - 1)
    present = (k >= 0) & (mid < attr.p_end[kk])
    if not present.any():
        return False
    pairs = pd.DataFrame({"h": host.p_code[present], "a": attr.p_code[kk][present]})
    sizes = pairs.groupby("h")["a"].transform("size")
    pairs = pairs[sizes.to_numpy() >= 2]
    if pairs.empty:
        return True
    top = pairs.groupby("h")["a"].agg(lambda s: int(s.value_counts().iloc[0]))
    return float(top.sum()) / len(pairs) >= ATTRIBUTE_OF_SHARE


def _bounds(sig: _Signal) -> _I64:
    """The presence boundaries of ``sig`` (starts and ends), sorted."""
    out: _I64 = np.unique(np.concatenate([sig.p_start, sig.p_end]))
    return out


def _describes(attr: _Signal, host: _Signal, changes: _I64, bounds: _I64) -> bool:
    """Whether ``attr``'s value ``changes`` happen only at ``host``'s
    presence ``bounds`` (:data:`ATTRIBUTE_OF_SHARE` of them, far more often
    than chance) -- the one test of :func:`_attribute_hosts` that is used
    both ways, to find aliases."""
    if len(changes) < ATTRIBUTE_OF_MIN_CHANGES or len(host.p_start) < 2:
        return False
    if len(bounds) < len(changes):
        return False
    tol = int(COCHANGE_TOLERANCE_SECONDS * 1e9) + max(_late_ns(attr), _late_ns(host))
    near = _matched_share(changes, bounds, tol)
    if near < ATTRIBUTE_OF_SHARE:
        return False
    span = float(host.p_end[-1] - host.p_start[0])
    if span <= 0:
        return False
    chance = min(1.0, len(bounds) * 2.0 * tol / span)
    if chance > ATTRIBUTE_MAX_CHANCE:
        return False
    hits = round(near * len(changes))
    return _binomial_tail(hits, len(changes), chance) <= NEST_MAX_CHANCE_P


def _attribute_hosts(signals: list[_Signal], hosts: list[_Signal]) -> None:
    """Set ``attribute_of``: the identifier whose presences a signal describes.

    A signal (of any role but constant or insufficient data) describes an
    identifier -- the customer of an order, the material of a batch, a
    description text -- when its value changes only at the identifier's
    presence boundaries (:func:`_describes`), it keeps one value during
    :data:`ATTRIBUTE_OF_SHARE` of the identifier's presences, and a
    revisited identifier value brings back the same value
    (:func:`_one_value_per_host_value`). Not an attribute: the identifier's
    production or packing parent (or theirs), and a non-recurring identifier
    holding several of the host's presences (a container the nesting did
    not report). Two ids that change together one to one are aliases: the
    one with fewer distinct values is the attribute, a non-identifier before
    an identifier, otherwise neither. Of several hosts the coarsest (fewest
    presences) wins.
    """
    hosts = sorted(hosts, key=lambda h: h.uuid)
    by_uuid = {h.uuid: h for h in hosts}
    ancestors = {h.uuid: _ancestors(h, by_uuid) for h in hosts}
    bounds = {h.uuid: _bounds(h) for h in hosts}
    host_changes = {h.uuid: _value_changes(h) for h in hosts}
    for attr in signals:
        if attr.role in ("constant", "insufficient_data") or attr.distinct < 2:
            continue
        changes = host_changes.get(attr.uuid)
        if changes is None:
            changes = _value_changes(attr)
        if len(changes) < ATTRIBUTE_OF_MIN_CHANGES:
            continue
        attr_bounds: _I64 | None = None
        is_id = attr.role == "identifier"
        revisit = attr.metrics.get("revisit_ratio", math.nan)
        best: _Signal | None = None
        for host in hosts:
            if host is attr or attr.uuid in ancestors[host.uuid]:
                continue
            if best is not None and len(host.p_start) >= len(best.p_start):
                continue
            if (
                is_id
                and not revisit > ATTRIBUTE_MIN_REVISIT
                and len(host.p_start) >= NEST_MIN_FANOUT * len(attr.p_start)
            ):
                continue
            if not _describes(attr, host, changes, bounds[host.uuid]):
                continue
            slack = int(1.05 * max(_late_ns(attr), _late_ns(host)))
            slack += int(COCHANGE_TOLERANCE_SECONDS * 1e9)
            if _containment(host, attr, slack) < ATTRIBUTE_OF_SHARE:
                continue
            if not _one_value_per_host_value(attr, host):
                continue
            if attr_bounds is None:
                attr_bounds = bounds.get(attr.uuid)
                if attr_bounds is None:
                    attr_bounds = _bounds(attr)
            if _describes(host, attr, host_changes[host.uuid], attr_bounds) and not (
                attr.distinct < host.distinct
                or (not is_id and attr.distinct <= host.distinct)
            ):
                continue
            best = host
        if best is not None:
            attr.attribute_of = best.uuid


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
    "packing_parent_uuid",
    "evidence",
    "attribute_of",
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


@dataclass(frozen=True, eq=False)
class ObjectReport(Mapping[str, "str | None"]):
    """Result of :func:`identify_objects`.

    The report is also a read-only mapping ``{uuid: object_type}`` over every
    signal, with ``None`` for the signals that are not identifiers:
    ``report["line:a"]``, ``dict(report)``.

    Attributes:
        table: One row per signal (columns :data:`REPORT_COLUMNS`), sorted by
            uuid. ``parent_uuid`` is an identifier's production parent (a
            serial's batch, a batch's material), ``packing_parent_uuid`` the
            handling unit that packs it; both ``None`` when there is none.
            ``attribute_of`` (last column) is the identifier whose objects a
            signal describes (the customer of an order), else ``None``.
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

    def _types(self) -> dict[str, str | None]:
        roles = self.table["role"].to_numpy()
        types = self.table["object_type"].to_numpy()
        return {
            str(u): (str(t) if r == "identifier" and t is not None else None)
            for u, r, t in zip(self.table["uuid"].to_numpy(), roles, types)
        }

    def __getitem__(self, signal_uuid: str) -> str | None:
        types = self._types()
        if signal_uuid not in types:
            raise KeyError(signal_uuid)
        return types[signal_uuid]

    def __iter__(self) -> Iterator[str]:
        return iter(self._types())

    def __len__(self) -> int:
        return len(self.table)

    def to_frame(self) -> pd.DataFrame:
        """The per-signal table (a copy)."""
        return self.table.copy()

    def identifiers(self) -> pd.DataFrame:
        """The rows whose role is ``identifier``."""
        out = self.table[self.table["role"] == "identifier"]
        return out.reset_index(drop=True)

    def to_object_specs(
        self,
        min_confidence: float = 0.6,
        *,
        include_generic: bool = False,
        namespace: bool = False,
        attributes: bool = False,
    ) -> list[ObjectSpec]:
        """Object specs for :func:`~ts_shape.eventlog.detect_objects`.

        Args:
            min_confidence: Keep identifiers whose ``type_confidence`` is at
                least this.
            include_generic: Also keep identifiers typed ``identifier`` (their
                confidence is the identifier score).
            namespace: Render object ids as ``"{type}:{value}"``
                (:attr:`~ts_shape.eventlog.ObjectSpec.id_template`), so the
                same value on two types (an order and a batch both numbered
                ``4711``) gives two object ids.
            attributes: Fill :attr:`~ts_shape.eventlog.ObjectSpec.attributes`
                with the signals whose ``attribute_of`` is the spec's signal
                (the customer of an order, the material of a work order). An
                attribute is named by its type when it is an identifier typed
                with at least ``min_confidence``, else by its uuid. Such an
                identifier is still its own spec when it is kept.

        Returns:
            One :class:`~ts_shape.eventlog.ObjectSpec` per kept identifier,
            in uuid order.
        """
        from ts_shape.eventlog.objects import _RESERVED, ObjectSpec

        _register_vocabulary()
        described: dict[str, dict[str, str]] = {}
        if attributes and "attribute_of" in self.table.columns:
            hosts = self.table["attribute_of"].to_numpy()
            names: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
            for row, host in zip(self.table.itertuples(index=False), hosts):
                if host is None or (isinstance(host, float) and math.isnan(host)):
                    continue
                typed = (
                    row.role == "identifier"
                    and row.object_type not in (None, "identifier")
                    and row.type_confidence >= min_confidence
                )
                name = str(row.object_type) if typed else str(row.uuid)
                names[str(host)].append((name, str(row.uuid)))
            for host, pairs in names.items():
                counts: defaultdict[str, int] = defaultdict(int)
                for name, _ in pairs:
                    counts[name] += 1
                described[host] = {
                    (
                        name if counts[name] == 1 and name not in _RESERVED else uuid
                    ): uuid
                    for name, uuid in pairs
                }
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
                    id_template="{type}:{value}" if namespace else "{value}",
                    attributes=described.get(str(row.uuid), {}),
                )
            )
        return specs

    def hierarchy(
        self, min_confidence: float = 0.6, *, include_generic: bool = False
    ) -> dict[str, list[str]]:
        """Signal-level ``part_of`` edges for
        :func:`~ts_shape.eventlog.detect_objects`.

        ``{child_uuid: [parent_uuid, packing_parent_uuid]}`` from the nesting
        columns, for the signals :meth:`to_object_specs` keeps with the same
        arguments (a parent it does not keep is left out, and so is a child
        left without parents). Pass it as
        ``detect_objects(frame, specs, hierarchy=report.hierarchy())``.

        Args:
            min_confidence: As for :meth:`to_object_specs`.
            include_generic: As for :meth:`to_object_specs`.

        Returns:
            Child signal uuid -> its production parent, then its packing
            parent, in uuid order of the children.
        """
        kept = {
            s.uuid
            for s in self.to_object_specs(
                min_confidence, include_generic=include_generic
            )
        }
        out: dict[str, list[str]] = {}
        for row in self.identifiers().itertuples(index=False):
            if str(row.uuid) not in kept:
                continue
            parents = [
                str(p)
                for p in (row.parent_uuid, row.packing_parent_uuid)
                if p is not None and str(p) in kept
            ]
            if parents:
                out[str(row.uuid)] = parents
        return out

    def to_event_log(
        self,
        dataframe: pd.DataFrame,
        events: object = None,
        *,
        min_confidence: float = 0.6,
        scope: Mapping[str, object] | None = None,
        namespace: bool = True,
        attributes: bool = True,
        include_generic: bool = False,
        detector: str | None = None,
        **kwargs: Any,
    ) -> EventLog:
        """The objects of the report as an OCEL 2.0 event log, in one call.

        Runs :meth:`to_object_specs` and :meth:`hierarchy`, then
        :func:`~ts_shape.eventlog.detect_objects` (without ``events``) or
        :func:`~ts_shape.eventlog.attach_objects` (with ``events``) on
        ``dataframe``.

        Args:
            dataframe: The long-format frame the report was made from (the id
                and attribute signals).
            events: The events to link to the objects: an
                :class:`~ts_shape.eventlog.EventLog`, a detector frame (with
                ``detector=``), ``{detector: frame}``, or a list of these.
                ``None`` gives an object-only log (``objects``, ``o2o``,
                ``object_changes``; no events or relations).
            min_confidence: As for :meth:`to_object_specs`.
            scope: ``{signal_uuid: label}`` for
                :func:`~ts_shape.eventlog.attach_objects`: an event links only
                to objects of its own line or station. Needs ``events``.
            namespace: Object ids ``"{type}:{value}"`` (default ``True``).
            attributes: Capture the ``attribute_of`` signals as object
                attributes (default ``True``).
            include_generic: As for :meth:`to_object_specs`.
            detector: ``"ClassName.method_name"`` of the detector frames in
                ``events``, for :func:`~ts_shape.eventlog.to_event_log`.
            **kwargs: Passed to ``detect_objects`` / ``attach_objects``
                (``hierarchy`` replaces :meth:`hierarchy`; ``qualifiers``,
                ``fanout_threshold``, ``infer_o2o``, ``uuid_column``,
                ``time_column``, ``validate``).

        Returns:
            :class:`~ts_shape.eventlog.EventLog`.

        Raises:
            ValueError: When ``scope`` is given without ``events``, or a
                detector frame without ``detector``.
            TypeError: For an ``events`` value of another kind.
        """
        from ts_shape.eventlog.objects import attach_objects, detect_objects

        specs = self.to_object_specs(
            min_confidence,
            include_generic=include_generic,
            namespace=namespace,
            attributes=attributes,
        )
        kwargs.setdefault(
            "hierarchy",
            self.hierarchy(min_confidence, include_generic=include_generic),
        )
        if events is None:
            if scope is not None:
                raise ValueError(
                    "to_event_log: scope= links events to objects; pass events="
                )
            return detect_objects(dataframe, specs, **kwargs)
        log = _events_log(events, detector)
        return attach_objects(log, dataframe, specs, scope=scope, **kwargs)

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
            if row.packing_parent_uuid is not None:
                parent += f", packed in {row.packing_parent_uuid}"
            lines.append(
                f"  {row.uuid}: {row.object_type} "
                f"({row.type_confidence:.2f}){parent}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _events_log(events: object, detector: str | None) -> EventLog:
    """One event log from an EventLog, a detector frame, ``{detector:
    frame}`` or a list of these."""
    from ts_shape.eventlog.concat import concat
    from ts_shape.eventlog.model import EventLog
    from ts_shape.eventlog.normalizer import to_event_log

    def one(item: object) -> list[EventLog]:
        if isinstance(item, EventLog):
            return [item]
        if isinstance(item, pd.DataFrame):
            if detector is None:
                raise ValueError(
                    "to_event_log: events= is a detector frame; pass "
                    "detector='ClassName.method_name' (or an EventLog, or "
                    "{detector: frame})"
                )
            return [to_event_log(item, detector=detector)]
        if isinstance(item, Mapping):
            return [to_event_log(f, detector=str(d)) for d, f in item.items()]
        raise TypeError(
            "to_event_log: events= takes an EventLog, a detector frame, "
            "{detector: frame} or a list of these; got " + type(item).__name__
        )

    if isinstance(events, list | tuple):
        logs = [log for item in events for log in one(item)]
    else:
        logs = one(events)
    return logs[0] if len(logs) == 1 else concat(*logs)


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
        "packing_parent_uuid": sig.packing_parent if is_id else None,
        "evidence": tuple(sig.screen_evidence + sig.evidence),
        "attribute_of": sig.attribute_of,
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
    min_samples: int = 3,
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
        :class:`ObjectReport` with one row per signal; ``parent_uuid`` and
        ``packing_parent_uuid`` give each identifier's production and
        packing parent (see *Nesting* in the module docstring).

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
    _cross_signal(cross[:MAX_CROSS_SIGNALS], counters, hints)
    for sig in list(ids):
        why = None if sig.uuid in hints else _is_fault_code(sig)
        if why is not None:
            sig.role = "state"
            sig.screen_evidence.append(why)
            ids.remove(sig)
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
        if sig.packing_parent is not None:
            sig.evidence.append(f"packed in {sig.packing_parent}")
    cross_ids = {s.uuid for s in cross[:MAX_CROSS_SIGNALS]}
    _attribute_hosts(signals, [s for s in ids if s.uuid in cross_ids])

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
        "packing_parent_uuid",
        "attribute_of",
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
