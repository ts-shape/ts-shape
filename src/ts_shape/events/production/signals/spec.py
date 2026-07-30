"""Declarative signal mapping for production reporting.

Reporting across hundreds of machines fails on configuration, not on maths. The
analytics for a press, a CNC cell and a filling line are identical; what differs
is *which tag means what* and *how it is encoded*. Passing those differences as
keyword arguments does not scale past a handful of assets, so this module makes
the mapping **data**: a :class:`MachineSpec` per asset, collected in a
:class:`FleetSpec`, both round-trippable through plain dicts (and therefore
JSON/YAML, a config table, or a row per machine in a database).

The vocabulary is deliberately small:

* :class:`CounterSpec` -- one part-counting signal, with the encoding it uses
  (:class:`CounterMode`) and what it counts (:class:`CounterKind`).
* :class:`StatusSpec` -- how a machine reports its state
  (:class:`StatusMode`), mapped onto the canonical :class:`MachineState`.
* :class:`CycleSpec` -- how cycle time can be derived (:class:`CycleMode`).
* :class:`ContextSpec` -- one production-context dimension (order, batch, part,
  serial, customer part number, ...) and how it binds to events
  (:class:`ContextBinding`).

Specs are frozen dataclasses and validate themselves on construction, so a
typo in a fleet config surfaces where it was written rather than as an empty
report three modules later.

Example -- two machines, different encodings, same report::

    from ts_shape.events.production.signals import (
        CounterKind, CounterMode, ContextSpec, CounterSpec,
        FleetSpec, MachineSpec, StatusMode, StatusSpec,
    )

    press = MachineSpec(
        machine_id="PRESS-01",
        attributes={"plant": "Ingolstadt", "line": "L1"},
        counters=(
            CounterSpec(uuid="press01.total", kind=CounterKind.TOTAL,
                        mode=CounterMode.CUMULATIVE, rollover_at=65536),
            CounterSpec(uuid="press01.nok", kind=CounterKind.NOK,
                        mode=CounterMode.CUMULATIVE, rollover_at=65536),
        ),
        status=StatusSpec(mode=StatusMode.PACKML, uuid="press01.packml"),
        context=(ContextSpec(name="order_number", uuid="press01.order"),),
        ideal_cycle_seconds=12.0,
    )
    cell = MachineSpec(
        machine_id="CELL-07",
        attributes={"plant": "Ingolstadt", "line": "L2"},
        counters=(
            CounterSpec(uuid="cell07.part_done", kind=CounterKind.OK,
                        mode=CounterMode.PULSE, parts_per_count=2),
        ),
        status=StatusSpec(mode=StatusMode.BOOLEAN, uuid="cell07.running"),
    )
    fleet = FleetSpec(machines=(press, cell))
    config = fleet.to_dict()      # JSON-serialisable, diffable, reviewable
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from typing import Any, TypeVar

__all__ = [
    "CONTEXT_DIMENSIONS",
    "PACKML_STATE_MAP",
    "STACKLIGHT_STATE_MAP",
    "ContextBinding",
    "ContextSpec",
    "CounterKind",
    "CounterMode",
    "CounterSpec",
    "CycleMode",
    "CycleSpec",
    "FleetSpec",
    "MachineSpec",
    "MachineState",
    "ResetPolicy",
    "Rollover",
    "StatusMode",
    "StatusSpec",
    "TimeUnit",
]

_E = TypeVar("_E", bound=Enum)


def _as_enum(enum_cls: type[_E], value: Any, *, field_name: str) -> _E:
    """Coerce a string/enum member to ``enum_cls``, with an actionable error."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(repr(m.value) for m in enum_cls)
        raise ValueError(
            f"{field_name}={value!r} is not a valid {enum_cls.__name__}. "
            f"Expected one of: {allowed}."
        ) from None


# ---------------------------------------------------------------------------
# Canonical vocabularies
# ---------------------------------------------------------------------------
class MachineState(StrEnum):
    """Canonical machine states every status encoding is mapped onto.

    Deliberately coarse: nine states are enough to compute availability and to
    compare a press against a filling line, while any plant-specific nuance
    survives in the ``raw_value`` column next to the canonical state.

    The :attr:`availability_class` grouping follows ISO 22400 time accounting --
    ``productive`` time plus ``loss`` time make up planned busy time, ``planned``
    time sits outside it, and ``unknown`` marks intervals with no data (never
    silently folded into either, since that is how availability reports start
    lying).
    """

    PRODUCING = "producing"
    IDLE = "idle"
    SETUP = "setup"
    DOWN = "down"
    BLOCKED = "blocked"
    STARVED = "starved"
    PLANNED_STOP = "planned_stop"
    OFF = "off"
    UNKNOWN = "unknown"

    @property
    def availability_class(self) -> str:
        """One of ``"productive"``, ``"loss"``, ``"planned"``, ``"unknown"``."""
        if self is MachineState.PRODUCING:
            return "productive"
        if self in (MachineState.PLANNED_STOP, MachineState.OFF):
            return "planned"
        if self is MachineState.UNKNOWN:
            return "unknown"
        return "loss"

    @property
    def is_productive(self) -> bool:
        """True only for :attr:`PRODUCING`."""
        return self is MachineState.PRODUCING

    @property
    def is_downtime(self) -> bool:
        """True for states that consume planned busy time without producing."""
        return self.availability_class == "loss"


class CounterKind(StrEnum):
    """What a counting signal counts.

    ``TOTAL`` is the sum of good and bad parts. Only two of ``OK`` / ``NOK`` /
    ``TOTAL`` are needed -- the third is derived (see
    :class:`~ts_shape.events.production.signals.counters.ProductionCounterNormalizer`).
    ``REWORK`` and ``SCRAP`` are reported alongside but never folded into
    ``TOTAL``, because plants disagree on whether a reworked part was produced
    once or twice.
    """

    OK = "ok"
    NOK = "nok"
    TOTAL = "total"
    REWORK = "rework"
    SCRAP = "scrap"


class CounterMode(StrEnum):
    """How a counting signal encodes its quantity.

    ``CUMULATIVE``
        A totalizer that counts up. Quantity is the difference between
        consecutive readings, with wrap-around and reset handling.
    ``INCREMENTAL``
        Each reading already *is* a quantity (``+1``, ``+8``) and is summed.
    ``PULSE``
        A trigger: every rising edge (or level change) is
        ``parts_per_count`` parts. Covers part-done flags and light barriers.
    ``QUALITY_FLAG``
        A boolean verdict per part: true means OK unless ``invert`` is set.
        Set ``trigger_uuid`` when the verdict is level-held and a separate
        signal marks the part.
    ``DEFECT_CODE``
        A code per part: values in ``ok_values`` are good, everything else is a
        NOK carrying the code as its defect reason.
    ``BATCH_TOTAL``
        An absolute quantity written once per batch/order rather than counted
        up; each *change* of the value contributes that value.
    """

    CUMULATIVE = "cumulative"
    INCREMENTAL = "incremental"
    PULSE = "pulse"
    QUALITY_FLAG = "quality_flag"
    DEFECT_CODE = "defect_code"
    BATCH_TOTAL = "batch_total"


class ResetPolicy(StrEnum):
    """What to do when a cumulative counter jumps backwards.

    A drop that is not explained by ``rollover_at`` means the counter was reset
    -- at a shift change, an order change, or a controller restart.

    ``COUNT_NEW``
        Treat the post-reset reading as production since the reset. This is
        right for the overwhelmingly common "reset to 0, then count up" case,
        and never loses parts.
    ``IGNORE``
        Contribute zero for that step. Use when resets are known to happen at
        arbitrary non-zero values.
    ``RAISE``
        Fail loudly -- for commissioning a new line, where an unexpected reset
        means the tag mapping is wrong.
    """

    COUNT_NEW = "count_new"
    IGNORE = "ignore"
    RAISE = "raise"


class Rollover(float, Enum):
    """Wrap-around moduli of the integer widths PLC counters actually use.

    Pass to ``CounterSpec(rollover_at=...)``; a plain number works too::

        CounterSpec(uuid="c", rollover_at=Rollover.UINT16)   # 65536
    """

    UINT8 = 256.0
    INT8 = 128.0
    UINT16 = 65536.0
    INT16 = 32768.0
    UINT32 = 4294967296.0
    INT32 = 2147483648.0


class StatusMode(StrEnum):
    """How a machine reports its state.

    ``BOOLEAN``
        One flag: true -> ``true_state``, false -> ``false_state``.
    ``CODE_MAP``
        A numeric or string state code mapped through ``state_map``. Covers
        bespoke PLC state machines, which is most of the installed base.
    ``ONE_HOT``
        Several boolean tags (``running``, ``fault``, ``setup``, ...), each
        mapped to a state in ``state_map`` keyed by **uuid**. Overlaps are
        resolved by ``priority``.
    ``STACKLIGHT``
        A single signal carrying a lamp colour, mapped through
        :data:`STACKLIGHT_STATE_MAP` by default. For one lamp per tag use
        ``ONE_HOT``.
    ``PACKML``
        The ISA-88 / OMAC PackML state number, mapped through
        :data:`PACKML_STATE_MAP` by default.
    """

    BOOLEAN = "boolean"
    CODE_MAP = "code_map"
    ONE_HOT = "one_hot"
    STACKLIGHT = "stacklight"
    PACKML = "packml"


class CycleMode(StrEnum):
    """How cycle time is obtained.

    ``EXPLICIT``
        The machine publishes the measured cycle time; ``unit`` says in what.
    ``COUNTER_DELTA``
        Time between consecutive counter increments, divided by the parts in
        that increment. Works with any counter, which makes it the fallback
        that always exists.
    ``TRIGGER_EDGE``
        Time between consecutive edges of one trigger signal.
    ``TRIGGER_PAIR``
        Start signal to end signal, for machines that expose both.
    ``PART_PRESENT``
        Duration a part-present flag stays true (the in-process time).
    ``STATE_INTERVAL``
        Producing time from the status signal divided by parts produced -- the
        last resort when no cycle-level signal exists at all.
    """

    EXPLICIT = "explicit"
    COUNTER_DELTA = "counter_delta"
    TRIGGER_EDGE = "trigger_edge"
    TRIGGER_PAIR = "trigger_pair"
    PART_PRESENT = "part_present"
    STATE_INTERVAL = "state_interval"


class TimeUnit(StrEnum):
    """Unit of an explicit cycle-time signal.

    ``AUTO`` infers from magnitude against the plausibility bounds on the spec
    -- useful across a fleet where some controllers publish milliseconds and
    others seconds, but state the unit when you know it.
    """

    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    MINUTES = "min"
    HOURS = "h"
    AUTO = "auto"

    @property
    def seconds_factor(self) -> float:
        """Multiplier converting this unit to seconds.

        Raises:
            ValueError: For :attr:`AUTO`, which must be resolved from data
                first.
        """
        factors = {
            TimeUnit.SECONDS: 1.0,
            TimeUnit.MILLISECONDS: 1e-3,
            TimeUnit.MICROSECONDS: 1e-6,
            TimeUnit.MINUTES: 60.0,
            TimeUnit.HOURS: 3600.0,
        }
        if self is TimeUnit.AUTO:
            raise ValueError(
                "TimeUnit.AUTO has no fixed factor; it is resolved from the data."
            )
        return factors[self]


class ContextBinding(StrEnum):
    """How a context value is attached to an event in time.

    ``LAST_KNOWN``
        The value in force at the event -- the default, correct for identifiers
        written before production starts (order, part, batch).
    ``NEXT_KNOWN``
        The next value written at or after the event. Necessary for
        late-bound identifiers: a serial number or measured result the machine
        publishes *after* finishing the part it belongs to.
    ``NEAREST``
        Whichever is closer in time. Convenient for context sampled on a slow
        cycle, at the cost of being ambiguous at changeovers.
    """

    LAST_KNOWN = "last_known"
    NEXT_KNOWN = "next_known"
    NEAREST = "nearest"


#: Default mapping for :attr:`StatusMode.PACKML` -- PackML state numbers
#: (ISA-88 / OMAC) onto canonical states.
PACKML_STATE_MAP: Mapping[str, MachineState] = {
    "1": MachineState.SETUP,  # Clearing
    "2": MachineState.PLANNED_STOP,  # Stopped
    "3": MachineState.SETUP,  # Starting
    "4": MachineState.IDLE,  # Idle
    "5": MachineState.STARVED,  # Suspended (external cause)
    "6": MachineState.PRODUCING,  # Execute
    "7": MachineState.PLANNED_STOP,  # Stopping
    "8": MachineState.DOWN,  # Aborting
    "9": MachineState.DOWN,  # Aborted
    "10": MachineState.BLOCKED,  # Holding (internal cause)
    "11": MachineState.BLOCKED,  # Held
    "12": MachineState.SETUP,  # Unholding
    "13": MachineState.STARVED,  # Suspending
    "14": MachineState.SETUP,  # Unsuspending
    "15": MachineState.SETUP,  # Resetting
    "16": MachineState.IDLE,  # Completing
    "17": MachineState.IDLE,  # Complete
}

#: Default mapping for :attr:`StatusMode.STACKLIGHT` -- lamp colour onto
#: canonical state, following the common convention (green running, yellow
#: attention/idle, red fault, blue material request, white planned stop).
STACKLIGHT_STATE_MAP: Mapping[str, MachineState] = {
    "green": MachineState.PRODUCING,
    "yellow": MachineState.IDLE,
    "amber": MachineState.IDLE,
    "orange": MachineState.IDLE,
    "red": MachineState.DOWN,
    "red_flashing": MachineState.DOWN,
    "blue": MachineState.STARVED,
    "white": MachineState.PLANNED_STOP,
    "off": MachineState.OFF,
    "none": MachineState.OFF,
}

#: Context dimension names ts-shape treats as well-known. Nothing enforces this
#: list -- :class:`ContextSpec` accepts any name -- but sticking to it keeps
#: column names consistent across plants, which is what makes a fleet-wide
#: ``GROUP BY`` possible.
CONTEXT_DIMENSIONS: tuple[str, ...] = (
    "order_number",
    "batch_number",
    "part_number",
    "serial_number",
    "customer_part_number",
    "variant",
    "program_number",
    "recipe",
    "tool_id",
    "fixture_id",
    "nest_id",
    "operator_id",
    "shift",
    "material_lot",
    "supplier_lot",
    "work_center",
    "campaign",
    "pallet_id",
    "carrier_id",
    "target_quantity",
)


# ---------------------------------------------------------------------------
# Signal specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CounterSpec:
    """One part-counting signal and how to read it.

    Args:
        uuid: Signal identifier of the counter/trigger/flag.
        kind: What is counted -- see :class:`CounterKind`.
        mode: How the quantity is encoded -- see :class:`CounterMode`.
        value_column: Value column to read; auto-detected when ``None``.
        parts_per_count: Parts represented by one count. Use it for
            multi-cavity moulds and multi-lane machines (a pulse that means
            "8 parts"), or a fractional value for a counter that ticks more
            than once per part.
        rollover_at: Modulus of the counter register (e.g.
            :attr:`Rollover.UINT16`). Set it and a 16-bit counter wrapping
            65530 -> 4 correctly yields 10 parts instead of a phantom reset.
        rollover_tolerance: Largest share of the register a single wrap may
            imply, as a fraction in ``(0, 1]``. This is what separates a real
            wrap from a reset: ``65534 -> 3`` implies 5 parts (plausible),
            while ``9 -> 0`` would imply 65527 parts -- almost the entire
            register -- and is therefore treated as a reset instead. Set
            ``max_plausible_delta`` for exact control when the process rate is
            known.
        reset_policy: How to treat backwards jumps -- see :class:`ResetPolicy`.
        max_plausible_delta: Reject a single step larger than this as a glitch
            (garbage reads and re-initialised registers produce enormous
            jumps). Rejected steps are reported by ``anomalies()``, never
            counted.
        allow_negative: Keep negative steps as negative quantities, for
            correction counters that are decremented deliberately.
        trigger_uuid: For ``QUALITY_FLAG`` / ``DEFECT_CODE``, the signal whose
            edges mark a part; the verdict is then read as of that instant.
            Required when the verdict is level-held rather than pulsed.
        invert: For ``QUALITY_FLAG``, true means NOK instead of OK.
        ok_values: For ``DEFECT_CODE``, the code values that mean "good".
            Compared case-insensitively as strings; defaults to ``("0",)``.
        duplicate_policy: How to collapse two readings sharing a timestamp --
            see :func:`~ts_shape.events.production.signals.values.signal_frame`.
        label: Optional human-readable name carried into outputs.
    """

    uuid: str
    kind: CounterKind = CounterKind.TOTAL
    mode: CounterMode = CounterMode.CUMULATIVE
    value_column: str | None = None
    parts_per_count: float = 1.0
    rollover_at: float | None = None
    rollover_tolerance: float = 0.5
    reset_policy: ResetPolicy = ResetPolicy.COUNT_NEW
    max_plausible_delta: float | None = None
    allow_negative: bool = False
    trigger_uuid: str | None = None
    invert: bool = False
    ok_values: tuple[str, ...] = ()
    edge: str = "rising"
    duplicate_policy: str = "last"
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _as_enum(CounterKind, self.kind, field_name="kind")
        )
        object.__setattr__(
            self, "mode", _as_enum(CounterMode, self.mode, field_name="mode")
        )
        object.__setattr__(
            self,
            "reset_policy",
            _as_enum(ResetPolicy, self.reset_policy, field_name="reset_policy"),
        )
        if not self.uuid:
            raise ValueError("CounterSpec.uuid must be a non-empty signal identifier.")
        if self.parts_per_count <= 0:
            raise ValueError(
                f"CounterSpec(uuid={self.uuid!r}).parts_per_count must be > 0, "
                f"got {self.parts_per_count}."
            )
        if self.rollover_at is not None:
            object.__setattr__(self, "rollover_at", float(self.rollover_at))
            if self.rollover_at <= 0:
                raise ValueError(
                    f"CounterSpec(uuid={self.uuid!r}).rollover_at must be > 0, "
                    f"got {self.rollover_at}."
                )
            if self.mode is not CounterMode.CUMULATIVE:
                raise ValueError(
                    f"CounterSpec(uuid={self.uuid!r}) sets rollover_at but mode is "
                    f"{self.mode.value!r}; wrap-around only applies to "
                    "CounterMode.CUMULATIVE."
                )
        if not 0.0 < self.rollover_tolerance <= 1.0:
            raise ValueError(
                f"CounterSpec(uuid={self.uuid!r}).rollover_tolerance must be in "
                f"(0, 1], got {self.rollover_tolerance}."
            )
        if self.max_plausible_delta is not None and self.max_plausible_delta <= 0:
            raise ValueError(
                f"CounterSpec(uuid={self.uuid!r}).max_plausible_delta must be > 0, "
                f"got {self.max_plausible_delta}."
            )
        if self.mode is CounterMode.DEFECT_CODE and not self.ok_values:
            object.__setattr__(self, "ok_values", ("0",))
        object.__setattr__(self, "ok_values", tuple(str(v) for v in self.ok_values))
        if self.edge not in {"rising", "falling", "both", "any"}:
            raise ValueError(
                f"CounterSpec(uuid={self.uuid!r}).edge={self.edge!r} is invalid; "
                "expected 'rising' (count when the trigger goes true), 'falling' "
                "(when it goes false), 'both', or 'any' (every reported true value "
                "is a part -- for historians that log each pulse without the "
                "intervening false)."
            )
        if self.mode is CounterMode.DEFECT_CODE and self.kind not in (
            CounterKind.TOTAL,
            CounterKind.NOK,
        ):
            raise ValueError(
                f"CounterSpec(uuid={self.uuid!r}) uses CounterMode.DEFECT_CODE, which "
                "classifies every part as OK or NOK; kind must be CounterKind.TOTAL "
                f"(or NOK to count only rejects), got {self.kind.value!r}."
            )

    @property
    def signal_uuids(self) -> tuple[str, ...]:
        """Every signal this spec reads, for coverage checks."""
        return (
            (self.uuid,)
            if self.trigger_uuid is None
            else (self.uuid, self.trigger_uuid)
        )

    @property
    def classifies_quality(self) -> bool:
        """True when this single spec yields both OK and NOK quantities."""
        return self.mode in (CounterMode.QUALITY_FLAG, CounterMode.DEFECT_CODE)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (inverse of :meth:`from_dict`)."""
        return {
            "uuid": self.uuid,
            "kind": self.kind.value,
            "mode": self.mode.value,
            "value_column": self.value_column,
            "parts_per_count": self.parts_per_count,
            "rollover_at": self.rollover_at,
            "rollover_tolerance": self.rollover_tolerance,
            "reset_policy": self.reset_policy.value,
            "max_plausible_delta": self.max_plausible_delta,
            "allow_negative": self.allow_negative,
            "trigger_uuid": self.trigger_uuid,
            "invert": self.invert,
            "ok_values": list(self.ok_values),
            "edge": self.edge,
            "duplicate_policy": self.duplicate_policy,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CounterSpec:
        """Build a spec from a mapping, ignoring unknown keys."""
        return cls(**_known_kwargs(cls, data))


@dataclass(frozen=True)
class StatusSpec:
    """How a machine reports its state, and how that maps to :class:`MachineState`.

    Args:
        mode: Encoding of the status signal -- see :class:`StatusMode`.
        uuid: The status signal, for every mode except ``ONE_HOT``.
        state_map: For ``CODE_MAP`` / ``STACKLIGHT`` / ``PACKML``, raw value ->
            state. For ``ONE_HOT``, **signal uuid** -> the state that signal
            being true means. Keys are compared as lower-cased strings, so
            ``4``, ``"4"`` and ``" 4"`` are the same code.
        default_state: State for values absent from ``state_map``. Defaults to
            :attr:`MachineState.UNKNOWN`, so unmapped codes are visible in
            ``unmapped_values()`` instead of being quietly counted as running.
        true_state: ``BOOLEAN`` mode -- state when the flag is true.
        false_state: ``BOOLEAN`` mode -- state when the flag is false.
        priority: For ``ONE_HOT``, the states to prefer when several flags are
            true at once, most important first. Defaults to
            fault-before-setup-before-production, because a machine that
            reports both ``running`` and ``fault`` is not producing.
        value_column: Value column to read; auto-detected when ``None``.
        min_duration: Debounce -- intervals shorter than this are absorbed
            into their neighbour, so a flickering bit does not turn into
            thousands of micro-intervals. Pandas offset string, e.g. ``"2s"``.
        stale_after: If no update arrives within this span, the machine's state
            becomes :attr:`MachineState.UNKNOWN` until it reports again.
            This is how a dropped connection stops being counted as "still
            producing". Pandas offset string, e.g. ``"15min"``.
        planned_stop_values: Raw values that mean a *planned* stop even though
            ``state_map`` maps them elsewhere -- a convenience for reason-coded
            stop signals.
    """

    mode: StatusMode = StatusMode.BOOLEAN
    uuid: str | None = None
    state_map: Mapping[str, MachineState] = field(default_factory=dict)
    default_state: MachineState = MachineState.UNKNOWN
    true_state: MachineState = MachineState.PRODUCING
    false_state: MachineState = MachineState.IDLE
    priority: tuple[MachineState, ...] = (
        MachineState.DOWN,
        MachineState.BLOCKED,
        MachineState.STARVED,
        MachineState.SETUP,
        MachineState.PLANNED_STOP,
        MachineState.OFF,
        MachineState.PRODUCING,
        MachineState.IDLE,
        MachineState.UNKNOWN,
    )
    value_column: str | None = None
    min_duration: str | None = None
    stale_after: str | None = None
    planned_stop_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _as_enum(StatusMode, self.mode, field_name="mode")
        )
        for name in ("default_state", "true_state", "false_state"):
            object.__setattr__(
                self, name, _as_enum(MachineState, getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "priority",
            tuple(
                _as_enum(MachineState, s, field_name="priority") for s in self.priority
            ),
        )
        object.__setattr__(
            self,
            "planned_stop_values",
            tuple(str(v) for v in self.planned_stop_values),
        )

        resolved = {
            str(k).strip().lower(): _as_enum(MachineState, v, field_name="state_map")
            for k, v in self.state_map.items()
        }
        if not resolved:
            if self.mode is StatusMode.PACKML:
                resolved = {k: v for k, v in PACKML_STATE_MAP.items()}
            elif self.mode is StatusMode.STACKLIGHT:
                resolved = {k: v for k, v in STACKLIGHT_STATE_MAP.items()}
        object.__setattr__(self, "state_map", resolved)

        if self.mode is StatusMode.ONE_HOT:
            if not self.state_map:
                raise ValueError(
                    "StatusSpec(mode=ONE_HOT) requires state_map mapping each signal "
                    "uuid to the MachineState it represents, e.g. "
                    "{'m1.running': MachineState.PRODUCING, "
                    "'m1.fault': MachineState.DOWN}."
                )
            if self.uuid is not None:
                raise ValueError(
                    "StatusSpec(mode=ONE_HOT) reads several signals from state_map; "
                    "leave uuid unset."
                )
        else:
            if not self.uuid:
                raise ValueError(
                    f"StatusSpec(mode={self.mode.value!r}) requires uuid to name the "
                    "status signal."
                )
            if self.mode is StatusMode.CODE_MAP and not self.state_map:
                raise ValueError(
                    "StatusSpec(mode=CODE_MAP) requires state_map mapping each raw "
                    "state code to a MachineState. Use "
                    "MachineStatusNormalizer.unmapped_values() on a sample to "
                    "discover the codes a controller actually emits."
                )

    @property
    def signal_uuids(self) -> tuple[str, ...]:
        """Every signal this spec reads, for coverage checks."""
        if self.mode is StatusMode.ONE_HOT:
            return tuple(self.state_map.keys())
        return (self.uuid,) if self.uuid else ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (inverse of :meth:`from_dict`)."""
        return {
            "mode": self.mode.value,
            "uuid": self.uuid,
            "state_map": {k: v.value for k, v in self.state_map.items()},
            "default_state": self.default_state.value,
            "true_state": self.true_state.value,
            "false_state": self.false_state.value,
            "priority": [s.value for s in self.priority],
            "value_column": self.value_column,
            "min_duration": self.min_duration,
            "stale_after": self.stale_after,
            "planned_stop_values": list(self.planned_stop_values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StatusSpec:
        """Build a spec from a mapping, ignoring unknown keys."""
        kwargs = _known_kwargs(cls, data)
        if "priority" in kwargs and kwargs["priority"] is not None:
            kwargs["priority"] = tuple(kwargs["priority"])
        if (
            "planned_stop_values" in kwargs
            and kwargs["planned_stop_values"] is not None
        ):
            kwargs["planned_stop_values"] = tuple(kwargs["planned_stop_values"])
        return cls(**kwargs)


@dataclass(frozen=True)
class CycleSpec:
    """How to obtain cycle time for a machine.

    Args:
        mode: Derivation strategy -- see :class:`CycleMode`.
        uuid: Primary signal: the cycle-time value (``EXPLICIT``), the counter
            (``COUNTER_DELTA``), the trigger (``TRIGGER_EDGE``,
            ``TRIGGER_PAIR`` start), or the part-present flag
            (``PART_PRESENT``). For ``STATE_INTERVAL`` it names the counter to
            divide producing time by, and may be omitted when canonical count
            events are supplied to the normalizer directly.
        end_uuid: ``TRIGGER_PAIR`` -- the cycle-end signal.
        value_column: Value column to read; auto-detected when ``None``.
        unit: Unit of an ``EXPLICIT`` signal -- see :class:`TimeUnit`.
        cavities: Parts produced per cycle. Cycle time is divided by it to give
            a comparable per-part time, and the raw figure is kept as
            ``cycle_time_raw_seconds``.
        min_seconds: Discard cycles shorter than this (sensor bounce).
        max_seconds: Discard cycles longer than this. Essential for
            edge-derived cycles: without it, the gap across a lunch break
            becomes a 45-minute "cycle" that wrecks every average.
        gate_states: Discard cycles overlapping these machine states. The usual
            choice is the downtime states, so cycle statistics describe
            production rather than interruptions.
        edge: Which transition of a boolean trigger marks a cycle --
            ``"rising"``, ``"falling"``, ``"both"``, or ``"any"`` to treat every
            reported true value as an event. Use ``"any"`` when the historian
            logs one row per pulse and never the intervening false, where edge
            detection would see a single unending pulse.
    """

    mode: CycleMode = CycleMode.COUNTER_DELTA
    uuid: str | None = None
    end_uuid: str | None = None
    value_column: str | None = None
    unit: TimeUnit = TimeUnit.SECONDS
    cavities: float = 1.0
    min_seconds: float | None = None
    max_seconds: float | None = None
    gate_states: tuple[MachineState, ...] = ()
    edge: str = "rising"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode", _as_enum(CycleMode, self.mode, field_name="mode")
        )
        object.__setattr__(
            self, "unit", _as_enum(TimeUnit, self.unit, field_name="unit")
        )
        object.__setattr__(
            self,
            "gate_states",
            tuple(
                _as_enum(MachineState, s, field_name="gate_states")
                for s in self.gate_states
            ),
        )
        if self.edge not in {"rising", "falling", "both", "any"}:
            raise ValueError(
                f"CycleSpec.edge={self.edge!r} is invalid; expected 'rising', "
                "'falling', 'both', or 'any' (treat every reported true value as an "
                "event, for pulse-logged triggers that never report the false)."
            )
        if self.cavities <= 0:
            raise ValueError(f"CycleSpec.cavities must be > 0, got {self.cavities}.")
        if self.mode is not CycleMode.STATE_INTERVAL and not self.uuid:
            raise ValueError(
                f"CycleSpec(mode={self.mode.value!r}) requires uuid to name the "
                "source signal."
            )
        if self.mode is CycleMode.TRIGGER_PAIR and not self.end_uuid:
            raise ValueError(
                "CycleSpec(mode=TRIGGER_PAIR) requires end_uuid for the cycle-end "
                "signal (uuid is the cycle-start signal)."
            )
        if self.unit is TimeUnit.AUTO and self.mode is not CycleMode.EXPLICIT:
            raise ValueError(
                "TimeUnit.AUTO only applies to CycleMode.EXPLICIT; derived cycle "
                "times are already in seconds."
            )
        if (
            self.min_seconds is not None
            and self.max_seconds is not None
            and self.min_seconds >= self.max_seconds
        ):
            raise ValueError(
                f"CycleSpec.min_seconds ({self.min_seconds}) must be below "
                f"max_seconds ({self.max_seconds})."
            )

    @property
    def signal_uuids(self) -> tuple[str, ...]:
        """Every signal this spec reads, for coverage checks."""
        return tuple(u for u in (self.uuid, self.end_uuid) if u)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (inverse of :meth:`from_dict`)."""
        return {
            "mode": self.mode.value,
            "uuid": self.uuid,
            "end_uuid": self.end_uuid,
            "value_column": self.value_column,
            "unit": self.unit.value,
            "cavities": self.cavities,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "gate_states": [s.value for s in self.gate_states],
            "edge": self.edge,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CycleSpec:
        """Build a spec from a mapping, ignoring unknown keys."""
        kwargs = _known_kwargs(cls, data)
        if "gate_states" in kwargs and kwargs["gate_states"] is not None:
            kwargs["gate_states"] = tuple(kwargs["gate_states"])
        return cls(**kwargs)


@dataclass(frozen=True)
class ContextSpec:
    """One production-context dimension and how it binds to events.

    Args:
        name: Output column name. Prefer a value from
            :data:`CONTEXT_DIMENSIONS` so fleet-wide grouping works.
        uuid: Signal carrying the identifier.
        value_column: Value column to read; auto-detected when ``None``.
        binding: Temporal binding -- see :class:`ContextBinding`.
        valid_for: How long a value stays valid, as a pandas offset string.
            Beyond it, context is reported as missing rather than stale. Vital
            for signals that are only written at order start: without a limit,
            last month's order number is attached to today's parts.
        sentinels: Placeholder values treated as "no identifier", on top of
            :data:`~ts_shape.events.production.signals.values.DEFAULT_SENTINELS`.
        upper: Upper-case values before comparison.
        strip_leading_zeros: Treat ``"000123"`` and ``"123"`` as one value.
        zero_is_sentinel: Treat ``0`` as "no identifier" -- right for serials
            and orders, wrong where ``0`` is a real part number.
        clear_on_sentinel: Whether a sentinel value *ends* the context. With
            ``True`` (the default) a machine that writes ``""`` when an order
            finishes leaves subsequent parts with no order, which is the honest
            answer. Set ``False`` when the signal drops to a placeholder between
            parts while the order is still running, so the last real value is
            carried instead.
        numeric: Keep the value numeric (target quantity, nominal weight)
            rather than normalising it to a label.
    """

    name: str
    uuid: str
    value_column: str | None = None
    binding: ContextBinding = ContextBinding.LAST_KNOWN
    valid_for: str | None = None
    sentinels: tuple[str, ...] = ()
    upper: bool = False
    strip_leading_zeros: bool = False
    zero_is_sentinel: bool = False
    clear_on_sentinel: bool = True
    numeric: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding",
            _as_enum(ContextBinding, self.binding, field_name="binding"),
        )
        object.__setattr__(self, "sentinels", tuple(str(s) for s in self.sentinels))
        if not self.name:
            raise ValueError("ContextSpec.name must be a non-empty column name.")
        if not self.uuid:
            raise ValueError(
                f"ContextSpec(name={self.name!r}).uuid must be a non-empty signal "
                "identifier."
            )
        if self.numeric and (self.upper or self.strip_leading_zeros):
            raise ValueError(
                f"ContextSpec(name={self.name!r}) sets numeric=True together with a "
                "label option (upper/strip_leading_zeros); those only apply to "
                "label context."
            )

    @property
    def signal_uuids(self) -> tuple[str, ...]:
        """Every signal this spec reads, for coverage checks."""
        return (self.uuid,)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (inverse of :meth:`from_dict`)."""
        return {
            "name": self.name,
            "uuid": self.uuid,
            "value_column": self.value_column,
            "binding": self.binding.value,
            "valid_for": self.valid_for,
            "sentinels": list(self.sentinels),
            "upper": self.upper,
            "strip_leading_zeros": self.strip_leading_zeros,
            "zero_is_sentinel": self.zero_is_sentinel,
            "clear_on_sentinel": self.clear_on_sentinel,
            "numeric": self.numeric,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextSpec:
        """Build a spec from a mapping, ignoring unknown keys."""
        kwargs = _known_kwargs(cls, data)
        if "sentinels" in kwargs and kwargs["sentinels"] is not None:
            kwargs["sentinels"] = tuple(kwargs["sentinels"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Machine and fleet
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MachineSpec:
    """Everything needed to report on one machine.

    Args:
        machine_id: Stable asset identifier, used as the reporting key.
        counters: Part-counting signals. Any combination of kinds is allowed;
            missing kinds are derived where arithmetic permits.
        status: How the machine reports its state, if it does.
        cycle: How to obtain cycle time, if available.
        context: Production-context dimensions to attach to events.
        ideal_cycle_seconds: Nameplate cycle time per part, for the performance
            factor of OEE and for cycle-deviation reporting.
        attributes: Free-form asset attributes broadcast onto every output row
            -- ``plant``, ``line``, ``area``, ``cell``, ``cost_center``,
            ``machine_type``. These are what fleet-level roll-ups group by.
    """

    machine_id: str
    counters: tuple[CounterSpec, ...] = ()
    status: StatusSpec | None = None
    cycle: CycleSpec | None = None
    context: tuple[ContextSpec, ...] = ()
    ideal_cycle_seconds: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.machine_id:
            raise ValueError("MachineSpec.machine_id must be a non-empty identifier.")
        object.__setattr__(self, "counters", tuple(self.counters))
        object.__setattr__(self, "context", tuple(self.context))
        object.__setattr__(self, "attributes", dict(self.attributes))
        if self.ideal_cycle_seconds is not None and self.ideal_cycle_seconds <= 0:
            raise ValueError(
                f"MachineSpec(machine_id={self.machine_id!r}).ideal_cycle_seconds "
                f"must be > 0, got {self.ideal_cycle_seconds}."
            )
        names = [c.name for c in self.context]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"MachineSpec(machine_id={self.machine_id!r}) has duplicate context "
                f"dimension name(s) {duplicates}; each name becomes one output "
                "column, so they must be unique."
            )
        uuids = [c.uuid for c in self.counters]
        repeated = sorted({u for u in uuids if uuids.count(u) > 1})
        if repeated:
            raise ValueError(
                f"MachineSpec(machine_id={self.machine_id!r}) lists counter signal(s) "
                f"{repeated} more than once; that would count the same parts twice. "
                "Several counters of the *same kind* on different signals are fine "
                "(multi-lane machines) and are summed."
            )

    # -- introspection ------------------------------------------------------
    @property
    def signal_uuids(self) -> tuple[str, ...]:
        """Every signal this machine's specs read, de-duplicated, in order."""
        out: list[str] = []
        for counter in self.counters:
            out.extend(counter.signal_uuids)
        for dimension in self.context:
            out.extend(dimension.signal_uuids)
        if self.status is not None:
            out.extend(self.status.signal_uuids)
        if self.cycle is not None:
            out.extend(self.cycle.signal_uuids)
        return tuple(dict.fromkeys(out))

    @property
    def counter_kinds(self) -> tuple[CounterKind, ...]:
        """Distinct counter kinds configured, in declaration order."""
        return tuple(dict.fromkeys(c.kind for c in self.counters))

    @property
    def context_names(self) -> tuple[str, ...]:
        """Configured context dimension names."""
        return tuple(c.name for c in self.context)

    def counters_of(self, kind: CounterKind | str) -> tuple[CounterSpec, ...]:
        """Return the counter specs of one kind."""
        wanted = _as_enum(CounterKind, kind, field_name="kind")
        return tuple(c for c in self.counters if c.kind is wanted)

    def with_context(self, *specs: ContextSpec) -> MachineSpec:
        """Return a copy with extra context dimensions appended."""
        return replace(self, context=(*self.context, *specs))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (inverse of :meth:`from_dict`)."""
        return {
            "machine_id": self.machine_id,
            "counters": [c.to_dict() for c in self.counters],
            "status": self.status.to_dict() if self.status is not None else None,
            "cycle": self.cycle.to_dict() if self.cycle is not None else None,
            "context": [c.to_dict() for c in self.context],
            "ideal_cycle_seconds": self.ideal_cycle_seconds,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MachineSpec:
        """Build a machine spec from a nested mapping (e.g. parsed JSON/YAML)."""
        kwargs = _known_kwargs(cls, data)
        kwargs["counters"] = tuple(
            c if isinstance(c, CounterSpec) else CounterSpec.from_dict(c)
            for c in (kwargs.get("counters") or ())
        )
        kwargs["context"] = tuple(
            c if isinstance(c, ContextSpec) else ContextSpec.from_dict(c)
            for c in (kwargs.get("context") or ())
        )
        status = kwargs.get("status")
        if status is not None and not isinstance(status, StatusSpec):
            kwargs["status"] = StatusSpec.from_dict(status)
        cycle = kwargs.get("cycle")
        if cycle is not None and not isinstance(cycle, CycleSpec):
            kwargs["cycle"] = CycleSpec.from_dict(cycle)
        return cls(**kwargs)


@dataclass(frozen=True)
class FleetSpec:
    """A collection of :class:`MachineSpec` -- the configuration of a report.

    Holding the whole fleet in one object is what lets a report be written once
    and run over 5 or 500 machines. It also provides the two things that matter
    operationally at that size: a fast ``uuid -> machine`` index for slicing one
    big frame, and the ability to review the configuration as data.

    Args:
        machines: The machines to report on; ``machine_id`` must be unique.
    """

    machines: tuple[MachineSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "machines", tuple(self.machines))
        ids = [m.machine_id for m in self.machines]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(
                f"FleetSpec has duplicate machine_id(s) {duplicates}; reporting keys "
                "must be unique."
            )

    def __len__(self) -> int:
        return len(self.machines)

    def __iter__(self) -> Iterator[MachineSpec]:
        return iter(self.machines)

    @property
    def machine_ids(self) -> tuple[str, ...]:
        """Configured machine ids, in declaration order."""
        return tuple(m.machine_id for m in self.machines)

    def machine(self, machine_id: str) -> MachineSpec:
        """Return one machine spec by id.

        Raises:
            KeyError: If no machine with that id is configured.
        """
        for m in self.machines:
            if m.machine_id == machine_id:
                return m
        raise KeyError(
            f"No machine {machine_id!r} in this FleetSpec. "
            f"Configured: {list(self.machine_ids)}"
        )

    def uuid_index(self) -> dict[str, str]:
        """Map every configured signal uuid to its ``machine_id``.

        This is the key to fleet-scale performance: tag one big frame with
        ``df["machine_id"] = df["uuid"].map(fleet.uuid_index())`` and group,
        instead of filtering the frame once per machine.

        Raises:
            ValueError: If a uuid is claimed by two machines, which would make
                that mapping ambiguous and silently mis-attribute production.
        """
        index: dict[str, str] = {}
        for machine in self.machines:
            for uuid in machine.signal_uuids:
                owner = index.get(uuid)
                if owner is not None and owner != machine.machine_id:
                    raise ValueError(
                        f"Signal {uuid!r} is configured on both {owner!r} and "
                        f"{machine.machine_id!r}. A signal must belong to exactly one "
                        "machine, otherwise its production is counted twice."
                    )
                index[uuid] = machine.machine_id
        return index

    @property
    def signal_uuids(self) -> tuple[str, ...]:
        """Every signal the fleet reads -- use it to scope a loader query."""
        out: list[str] = []
        for machine in self.machines:
            out.extend(machine.signal_uuids)
        return tuple(dict.fromkeys(out))

    @property
    def attribute_keys(self) -> tuple[str, ...]:
        """Union of attribute keys across machines, sorted -- the group-by menu."""
        keys: set[str] = set()
        for machine in self.machines:
            keys.update(machine.attributes.keys())
        return tuple(sorted(keys))

    def filter(self, **attributes: Any) -> FleetSpec:
        """Return a sub-fleet whose machines match all given attribute values.

        Example::

            fleet.filter(plant="Ingolstadt", line="L1")
        """
        selected = tuple(
            m
            for m in self.machines
            if all(m.attributes.get(k) == v for k, v in attributes.items())
        )
        return FleetSpec(machines=selected)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (inverse of :meth:`from_dict`)."""
        return {"machines": [m.to_dict() for m in self.machines]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | Sequence[Any]) -> FleetSpec:
        """Build a fleet from a mapping with a ``machines`` key, or a bare list."""
        raw: Iterable[Any]
        if isinstance(data, Mapping):
            raw = data.get("machines") or ()
        else:
            raw = data
        return cls(
            machines=tuple(
                m if isinstance(m, MachineSpec) else MachineSpec.from_dict(m)
                for m in raw
            )
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        machine_column: str = "machine_id",
        role_column: str = "role",
        uuid_column: str = "uuid",
    ) -> FleetSpec:
        """Build a fleet from a flat table of ``(machine, role, uuid, ...)`` rows.

        A tidy table is how signal mappings actually exist in a plant -- an
        export from the historian's tag list, a maintained spreadsheet, or a
        config table. One row per signal keeps it reviewable by the people who
        know the machines.

        Recognised ``role`` values:

        * ``counter_ok`` / ``counter_nok`` / ``counter_total`` /
          ``counter_rework`` / ``counter_scrap`` -- a :class:`CounterSpec`.
          Extra columns ``mode``, ``parts_per_count``, ``rollover_at``,
          ``reset_policy``, ``max_plausible_delta``, ``trigger_uuid``,
          ``invert``, ``ok_values``, ``value_column`` are passed through.
        * ``status`` -- a :class:`StatusSpec`; extra columns ``mode``,
          ``state_map``, ``true_state``, ``false_state``, ``min_duration``,
          ``stale_after``, ``value_column``.
        * ``cycle`` -- a :class:`CycleSpec`; extra columns ``mode``, ``unit``,
          ``cavities``, ``min_seconds``, ``max_seconds``, ``end_uuid``,
          ``edge``, ``value_column``.
        * ``context`` -- a :class:`ContextSpec`; the dimension name comes from
          a ``name`` (or ``dimension``) column, plus ``binding``, ``valid_for``,
          ``upper``, ``numeric``, ``zero_is_sentinel``, ``value_column``.
        * ``attribute`` -- an asset attribute; ``name`` is the key and ``value``
          the value. Any other column prefixed ``attr_`` on any row of the
          machine is also collected as an attribute.
        * ``ideal_cycle_seconds`` may be given on any row of a machine.

        Args:
            records: Row mappings -- e.g. ``df.to_dict("records")``.
            machine_column: Column naming the machine.
            role_column: Column naming the signal's role.
            uuid_column: Column holding the signal uuid.

        Returns:
            The assembled :class:`FleetSpec`.

        Raises:
            ValueError: If a row lacks a machine or an unknown role is used.
        """
        order: list[str] = []
        counters: dict[str, list[CounterSpec]] = {}
        contexts: dict[str, list[ContextSpec]] = {}
        statuses: dict[str, StatusSpec] = {}
        cycles: dict[str, CycleSpec] = {}
        attributes: dict[str, dict[str, Any]] = {}
        ideal: dict[str, float] = {}

        for row in records:
            data = {k: v for k, v in dict(row).items() if not _is_missing(v)}
            machine_id = data.get(machine_column)
            if not machine_id:
                raise ValueError(
                    f"Row {dict(row)!r} has no {machine_column!r}; every signal must "
                    "be assigned to a machine."
                )
            machine_id = str(machine_id)
            if machine_id not in order:
                order.append(machine_id)
                attributes[machine_id] = {}
            role = str(data.get(role_column, "")).strip().lower()
            uuid = data.get(uuid_column)

            for key, value in data.items():
                if key.startswith("attr_"):
                    attributes[machine_id][key[len("attr_") :]] = value
            if "ideal_cycle_seconds" in data:
                ideal[machine_id] = float(data["ideal_cycle_seconds"])

            if not role:
                continue
            if role.startswith("counter"):
                kind = role.split("_", 1)[1] if "_" in role else "total"
                counters.setdefault(machine_id, []).append(
                    CounterSpec.from_dict({**data, "uuid": uuid, "kind": kind})
                )
            elif role == "status":
                statuses[machine_id] = StatusSpec.from_dict({**data, "uuid": uuid})
            elif role == "cycle":
                cycles[machine_id] = CycleSpec.from_dict({**data, "uuid": uuid})
            elif role == "context":
                name = data.get("name") or data.get("dimension")
                if not name:
                    raise ValueError(
                        f"Context row for machine {machine_id!r} needs a 'name' "
                        "column naming the dimension (e.g. 'order_number')."
                    )
                contexts.setdefault(machine_id, []).append(
                    ContextSpec.from_dict({**data, "name": str(name), "uuid": uuid})
                )
            elif role == "attribute":
                name = data.get("name")
                if not name:
                    raise ValueError(
                        f"Attribute row for machine {machine_id!r} needs a 'name' "
                        "column."
                    )
                attributes[machine_id][str(name)] = data.get("value")
            elif role in {"ideal_cycle_seconds", "ideal_cycle"}:
                nameplate = data.get("value", data.get("ideal_cycle_seconds"))
                if nameplate is None:
                    raise ValueError(
                        f"Row role={role!r} for machine {machine_id!r} needs the "
                        "cycle time in a 'value' or 'ideal_cycle_seconds' column."
                    )
                ideal[machine_id] = float(nameplate)
            else:
                raise ValueError(
                    f"Unknown {role_column}={role!r} for machine {machine_id!r}. "
                    "Expected one of: counter_ok, counter_nok, counter_total, "
                    "counter_rework, counter_scrap, status, cycle, context, "
                    "attribute, ideal_cycle_seconds."
                )

        return cls(
            machines=tuple(
                MachineSpec(
                    machine_id=machine_id,
                    counters=tuple(counters.get(machine_id, ())),
                    status=statuses.get(machine_id),
                    cycle=cycles.get(machine_id),
                    context=tuple(contexts.get(machine_id, ())),
                    ideal_cycle_seconds=ideal.get(machine_id),
                    attributes=attributes.get(machine_id, {}),
                )
                for machine_id in order
            )
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _is_missing(value: Any) -> bool:
    """True for ``None``, NaN and empty/whitespace strings.

    Config tables arrive from spreadsheets and CSVs where an unset cell is any
    of those; treating them all as "not given" is what lets one wide table
    describe several signal roles.
    """
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return isinstance(value, str) and not value.strip()


def _known_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the mapping keys that are fields of ``cls``.

    Config rows carry extra columns (plant, comment, historian id). Filtering
    instead of failing lets one row feed several spec types.
    """
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return {k: v for k, v in data.items() if k in fields}
