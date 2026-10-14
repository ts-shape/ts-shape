"""Signal semantics: what a uuid's values *mean*.

Every analysis on industrial time series silently depends on a few facts
about each signal:

* **kind** -- is it a continuous measurement, a cumulative counter, a state
  (run/idle, recipe, alarm), or a sparse event stream?
* **encoding** -- is it sampled periodically, or change-encoded (one row per
  change, the ``is_delta`` convention)?
* **timing** -- its nominal sampling period and the gap after which the
  signal must be treated as "no data" rather than "value held".

Getting any of these wrong produces silently wrong results: counters that
lose production at a reset, intervals that collapse to zero length on
change-encoded data, averages biased by sampling density, durations that
bridge logging outages. :class:`SignalSpec` records these facts once per
uuid so that the primitives in :mod:`ts_shape.signals.primitives` (and the
detectors built on them) get them right by construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd  # type: ignore

__all__ = ["SignalSpec", "SignalSpecs", "KINDS", "ENCODINGS"]

Kind = Literal["continuous", "counter", "state", "event"]
Encoding = Literal["sampled", "change_encoded"]

KINDS: tuple[str, ...] = ("continuous", "counter", "state", "event")
ENCODINGS: tuple[str, ...] = ("sampled", "change_encoded")
_VALUE_COLUMNS = ("value_double", "value_integer", "value_bool", "value_string")


@dataclass(frozen=True)
class SignalSpec:
    """Semantics of one signal (one ``uuid``).

    Attributes:
        uuid: The signal identifier.
        kind: ``"continuous"`` (a measured level), ``"counter"`` (cumulative,
            may reset), ``"state"`` (a held discrete value) or ``"event"``
            (sparse occurrences without a held value).
        encoding: ``"sampled"`` (rows at a regular period, values repeat) or
            ``"change_encoded"`` (one row per change; a value holds until the
            next row).
        value_column: The ``value_*`` column that carries the signal.
        unit: Engineering unit, informational (e.g. ``"kWh"``, ``"pcs"``).
        nominal_period: Typical time between samples.
        max_gap: Longest interval across which a value may be held. Longer
            spans are "no data". ``None`` means never cap.
        tz: Timezone name of ``systime`` (``None`` for naive timestamps).
        reset_policy: For counters: ``"reset_to_zero"`` (a drop restarts the
            count; the default) or ``"none"`` (decreases are real, e.g. a
            level that is not cumulative).
        notes: Free-form provenance, e.g. why a field was inferred.
    """

    uuid: str
    kind: Kind = "continuous"
    encoding: Encoding = "sampled"
    value_column: str = "value_double"
    unit: str | None = None
    nominal_period: pd.Timedelta | None = None
    max_gap: pd.Timedelta | None = None
    tz: str | None = None
    reset_policy: Literal["reset_to_zero", "none"] = "reset_to_zero"
    notes: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.encoding not in ENCODINGS:
            raise ValueError(
                f"encoding must be one of {ENCODINGS}, got {self.encoding!r}"
            )
        if self.value_column not in _VALUE_COLUMNS:
            raise ValueError(
                f"value_column must be one of {_VALUE_COLUMNS}, got {self.value_column!r}"
            )
        for name in ("nominal_period", "max_gap"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, pd.Timedelta):
                object.__setattr__(self, name, pd.Timedelta(value))

    @property
    def is_change_encoded(self) -> bool:
        """Whether rows are emitted only when the value changes."""
        return self.encoding == "change_encoded"

    @property
    def holds_value(self) -> bool:
        """Whether a value persists until the next row (state / counter / level)."""
        return self.kind != "event"

    @property
    def interval_end(self) -> str:
        """``interval_end`` mode for :func:`ts_shape.utils.intervals.state_intervals`.

        A held value lasts until the next change, for both encodings: on
        sampled data closing at the last sample undercounts every interval by
        one period, on change-encoded data it collapses them to zero.
        """
        return "next_change" if self.holds_value else "last_sample"

    def with_(self, **changes: Any) -> SignalSpec:
        """Return a copy with ``changes`` applied (validated)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Plain, serializable mapping (Timedeltas as strings)."""
        out = asdict(self)
        for name in ("nominal_period", "max_gap"):
            if out[name] is not None:
                out[name] = str(out[name])
        if not out["notes"]:
            out.pop("notes")
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SignalSpec:
        """Build a spec from a plain mapping of field names to values.

        Args:
            data: Mapping of `SignalSpec` field names to values, e.g. the
                output of `to_dict`.

        Returns:
            The validated SignalSpec.

        Raises:
            ValueError: If ``data`` contains unknown field names or invalid
                field values.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown SignalSpec field(s): {sorted(unknown)}")
        return cls(**dict(data))


class SignalSpecs(Mapping[str, SignalSpec]):
    """A read-only collection of :class:`SignalSpec`, keyed by uuid.

    Build it with :func:`ts_shape.signals.infer_specs`, then override what the
    data cannot tell::

        specs = infer_specs(df).override({
            "line1:parts": {"kind": "counter", "unit": "pcs"},
            "line1:state": {"max_gap": "15min"},
        })
    """

    def __init__(self, specs: Mapping[str, SignalSpec] | None = None) -> None:
        self._specs: dict[str, SignalSpec] = dict(specs or {})
        for key, spec in self._specs.items():
            if key != spec.uuid:
                raise ValueError(f"Key {key!r} does not match spec uuid {spec.uuid!r}")

    def __getitem__(self, uuid: str) -> SignalSpec:
        return self._specs[uuid]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"SignalSpecs({len(self)} signals)"

    def override(self, changes: Mapping[str, Mapping[str, Any]]) -> SignalSpecs:
        """Return new specs with per-uuid field overrides applied.

        A uuid not yet present is added with the given fields.

        Args:
            changes: Mapping of uuid to ``{field: value}``; the fields replace those of
                the existing spec (see `SignalSpec.with_`).
        """
        merged = dict(self._specs)
        for uuid, fields in changes.items():
            base = merged.get(uuid, SignalSpec(uuid=uuid))
            merged[uuid] = base.with_(**dict(fields))
        return SignalSpecs(merged)

    def to_frame(self) -> pd.DataFrame:
        """One row per signal, columns = spec fields (notes excluded)."""
        rows = []
        for spec in self._specs.values():
            row = spec.to_dict()
            row.pop("notes", None)
            rows.append(row)
        cols = [f for f in SignalSpec.__dataclass_fields__ if f != "notes"]
        return pd.DataFrame(rows, columns=cols)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize to ``{uuid: {field: value}}`` (the YAML/JSON layout).

        The ``uuid`` key is omitted from each inner mapping and ``notes`` are
        excluded.

        Returns:
            Mapping of uuid to that signal's spec fields.
        """
        out = {}
        for uuid, spec in self._specs.items():
            data = spec.to_dict()
            data.pop("uuid")
            data.pop("notes", None)
            out[uuid] = data
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, Any]]) -> SignalSpecs:
        """Build from ``{uuid: {field: value}}`` (the YAML/JSON layout).

        Args:
            data: Mapping of uuid to ``{field: value}``, e.g. the output of `to_dict`.
                Field names must be `SignalSpec` fields.
        """
        return cls(
            {u: SignalSpec.from_dict({**dict(f), "uuid": u}) for u, f in data.items()}
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SignalSpecs:
        """Load ``{uuid: {field: value}}`` from a YAML file (needs PyYAML).

        Args:
            path: Path to a YAML file with the ``{uuid: {field: value}}`` layout; an
                empty file gives empty specs.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "Reading specs from YAML needs PyYAML: pip install pyyaml"
            ) from exc
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)
