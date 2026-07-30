"""Normalize any part-counter encoding into canonical OK/NOK production events.

Counting parts is the one thing every production report needs and the thing
that is encoded differently on every machine. Across a large fleet you will
meet, on the same reporting line:

* a 16-bit totalizer that wraps 65535 -> 0 mid-shift,
* a totalizer that is reset to zero at every shift change or order change,
* a signal that already sends ``+1`` per part,
* a part-done pulse where one pulse means eight parts (multi-cavity mould),
* a boolean OK/NOK verdict that only makes sense at the instant a separate
  trigger says a part exists,
* an integer defect code where ``0`` means good and anything else is a reject,
* a batch quantity written once, at the end, as an absolute number.

Reporting logic that assumes one of these silently produces wrong numbers on
the others -- a wrapped counter reads as a reset, a reset reads as negative
production, a multi-cavity pulse under-counts by 8x. :class:`ProductionCounterNormalizer`
absorbs all of them and emits one canonical stream of quantities, so everything
downstream (windowing, context breakdowns, OEE) is encoding-agnostic.

It also reports what it had to interpret. ``anomalies()`` lists every reset,
wrap-around and implausible jump, and ``diagnostics()`` summarises each counter
-- because at fleet scale the question is rarely "what is the number" but
"which of my 300 counters is lying".

Example::

    from ts_shape.events.production.signals import (
        CounterKind, CounterMode, CounterSpec, ProductionCounterNormalizer,
    )

    norm = ProductionCounterNormalizer(
        df,
        counters=[
            CounterSpec(uuid="m1.total", kind=CounterKind.TOTAL,
                        mode=CounterMode.CUMULATIVE, rollover_at=65536),
            CounterSpec(uuid="m1.nok", kind=CounterKind.NOK,
                        mode=CounterMode.CUMULATIVE),
        ],
        machine_id="M1",
    )
    norm.count_events()           # one row per counted quantity
    norm.counts_by_window("1h")   # qty_ok / qty_nok / qty_total + rates
    norm.anomalies()              # resets, wrap-arounds, glitches
    norm.diagnostics()            # per-counter health, one row per signal

Window edges: a cumulative counter's first reading in the loaded range only
establishes a baseline, so production before it cannot be attributed. Load one
reading of look-back before the reporting window when exact edges matter.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_point_df,
    finalize_summary_df,
)
from ts_shape.events.production.signals.spec import (
    CounterKind,
    CounterMode,
    CounterSpec,
    ResetPolicy,
)
from ts_shape.events.production.signals.values import (
    coerce_bool,
    coerce_label,
    coerce_numeric,
    signal_frame,
)
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

__all__ = ["ProductionCounterNormalizer"]

#: Quantity columns of the canonical wide (windowed) output, in report order.
QUANTITY_COLUMNS: tuple[str, ...] = (
    "qty_ok",
    "qty_nok",
    "qty_total",
    "qty_rework",
    "qty_scrap",
)

_EVENT_COLUMNS: tuple[str, ...] = (
    "systime",
    "uuid",
    "source_uuid",
    "is_delta",
    "machine_id",
    "kind",
    "quantity",
    "is_reset",
    "is_rollover",
    "defect_code",
    "raw_value",
)

_ANOMALY_COLUMNS: tuple[str, ...] = (
    "systime",
    "uuid",
    "source_uuid",
    "is_delta",
    "machine_id",
    "kind",
    "anomaly",
    "severity",
    "value",
    "previous_value",
    "quantity",
)


class ProductionCounterNormalizer(Base):
    """Turn heterogeneous part counters into canonical OK/NOK/total quantities.

    Args:
        dataframe: Timeseries in the standard ts-shape schema.
        counters: The counting signals to read. Several specs of the same
            :class:`~ts_shape.events.production.signals.spec.CounterKind` are
            summed, which is what multi-lane and multi-station machines need.
        time_column: Timestamp column.
        machine_id: Asset identifier broadcast onto every output row. Optional
            for a single machine, essential when concatenating a fleet.
        event_uuid: ``uuid`` value stamped on emitted events.
        uuid_column: Column holding signal identifiers.

    Raises:
        ValueError: If ``counters`` is empty, or a spec is invalid.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        counters: Sequence[CounterSpec],
        *,
        time_column: str = "systime",
        machine_id: str | None = None,
        event_uuid: str = "prod:count",
        uuid_column: str = "uuid",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        if not counters:
            raise ValueError(
                "ProductionCounterNormalizer requires at least one CounterSpec. "
                "Nothing can be counted without a counting signal."
            )
        self.counters = tuple(counters)
        self.time_column = time_column
        self.machine_id = machine_id
        self.event_uuid = event_uuid
        self.uuid_column = uuid_column
        self._events: pd.DataFrame | None = None
        self._anomalies: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Configured kinds
    # ------------------------------------------------------------------
    @property
    def configured_kinds(self) -> frozenset[CounterKind]:
        """Kinds this machine actually measures.

        A quality-classifying counter (verdict flag or defect code) measures
        both OK and NOK from one signal, so it contributes both. Knowing what
        is *measured* is what makes derivation honest: an unmeasured kind is
        computed from the others where arithmetic allows, and left missing where
        it does not, rather than defaulting to zero and implying perfect
        quality.
        """
        kinds: set[CounterKind] = set()
        for spec in self.counters:
            if not spec.classifies_quality:
                kinds.add(spec.kind)
            elif spec.kind is CounterKind.NOK:
                # Configured to count rejects only: good parts are not observed
                # by this signal, so neither OK nor TOTAL is measured here.
                kinds.add(CounterKind.NOK)
            else:
                # Every part passes the verdict, so all three are measured.
                kinds.update({CounterKind.OK, CounterKind.NOK, CounterKind.TOTAL})
        return frozenset(kinds)

    # ------------------------------------------------------------------
    # Per-spec normalization
    # ------------------------------------------------------------------
    def _frame_for(self, spec: CounterSpec, uuid: str) -> pd.DataFrame:
        return signal_frame(
            self.dataframe,
            uuid,
            time_column=self.time_column,
            value_column=spec.value_column,
            uuid_column=self.uuid_column,
            duplicate_policy=spec.duplicate_policy,
        )

    def _edges(self, values: pd.Series, edge: str) -> pd.Series:
        """Boolean mask of the requested transitions of a boolean series.

        The first reading is compared against ``False``, so a signal that is
        already true on its first sample counts as a rising edge. Real signals
        are frequently only reported on change, which means the first sample
        *is* the transition.

        ``edge="any"`` skips edge detection entirely and takes every true
        reading, which is what a historian that logs one row per pulse -- and
        never the intervening false -- actually contains.
        """
        flags = coerce_bool(values)
        if edge == "any":
            return flags
        prev = flags.shift(fill_value=False)
        rising = flags & ~prev
        falling = ~flags & prev
        if edge == "rising":
            return rising
        if edge == "falling":
            return falling
        return rising | falling

    def _cumulative(self, spec: CounterSpec, frame: pd.DataFrame) -> pd.DataFrame:
        """Differentiate a totalizer, handling wrap-around, resets and glitches."""
        values = coerce_numeric(frame["value"])
        prev = values.shift()
        delta = values - prev
        index = values.index

        quantity = pd.Series(0.0, index=index)
        is_reset = pd.Series(False, index=index)
        is_rollover = pd.Series(False, index=index)
        is_glitch = pd.Series(False, index=index)

        has_prev = prev.notna() & values.notna()
        max_delta = spec.max_plausible_delta

        forward = has_prev & (delta >= 0)
        quantity.loc[forward] = delta.loc[forward]
        if max_delta is not None:
            # A register re-initialised to a huge number, or a garbage read,
            # arrives as an enormous positive step. Counting it would ruin the
            # window; dropping it silently would hide a broken tag, so it is
            # dropped *and* reported.
            glitch = forward & (delta > max_delta)
            quantity.loc[glitch] = 0.0
            is_glitch |= glitch

        backward = has_prev & (delta < 0)
        if backward.any():
            if spec.allow_negative:
                quantity.loc[backward] = delta.loc[backward]
            else:
                remaining = backward.copy()
                if spec.rollover_at is not None:
                    candidate = (spec.rollover_at - prev) + values
                    # A wrap that implies most of the register is not a wrap:
                    # 65534 -> 3 means 5 parts, but 9 -> 0 would mean 65527,
                    # which is a reset. Without this guard every reset on a
                    # counter that *can* wrap inflates production enormously.
                    plausible = (candidate >= 0) & (
                        candidate <= spec.rollover_at * spec.rollover_tolerance
                    )
                    if max_delta is not None:
                        plausible &= candidate <= max_delta
                    wrapped = remaining & plausible.fillna(False)
                    quantity.loc[wrapped] = candidate.loc[wrapped]
                    is_rollover |= wrapped
                    remaining &= ~wrapped
                if remaining.any():
                    if spec.reset_policy is ResetPolicy.RAISE:
                        first = frame.loc[remaining, self.time_column].iloc[0]
                        raise ValueError(
                            f"Counter {spec.uuid!r} decreased at {first} "
                            f"({prev.loc[remaining].iloc[0]} -> "
                            f"{values.loc[remaining].iloc[0]}) and "
                            "reset_policy=ResetPolicy.RAISE. Set rollover_at if the "
                            "register wraps, or ResetPolicy.COUNT_NEW to treat the "
                            "drop as a reset."
                        )
                    is_reset |= remaining
                    if spec.reset_policy is ResetPolicy.COUNT_NEW:
                        after = values.loc[remaining].clip(lower=0)
                        if max_delta is not None:
                            after = after.where(after <= max_delta, 0.0)
                        quantity.loc[remaining] = after

        return self._assemble(
            spec,
            frame,
            quantity=quantity * spec.parts_per_count,
            kinds=spec.kind,
            raw=values,
            is_reset=is_reset,
            is_rollover=is_rollover,
            is_glitch=is_glitch,
            previous=prev,
        )

    def _incremental(self, spec: CounterSpec, frame: pd.DataFrame) -> pd.DataFrame:
        """Sum per-reading quantities that the machine already sends as deltas."""
        values = coerce_numeric(frame["value"])
        quantity = values.fillna(0.0)
        is_glitch = pd.Series(False, index=values.index)
        if not spec.allow_negative:
            negative = quantity < 0
            if negative.any():
                is_glitch |= negative
                quantity = quantity.clip(lower=0.0)
        if spec.max_plausible_delta is not None:
            glitch = quantity.abs() > spec.max_plausible_delta
            quantity = quantity.where(~glitch, 0.0)
            is_glitch |= glitch
        return self._assemble(
            spec,
            frame,
            quantity=quantity * spec.parts_per_count,
            kinds=spec.kind,
            raw=values,
            is_glitch=is_glitch,
        )

    def _pulse(self, spec: CounterSpec, frame: pd.DataFrame) -> pd.DataFrame:
        """Count configured edges of a trigger signal."""
        edges = self._edges(frame["value"], spec.edge)
        quantity = edges.astype("float64") * spec.parts_per_count
        return self._assemble(
            spec,
            frame,
            quantity=quantity,
            kinds=spec.kind,
            raw=coerce_numeric(frame["value"]),
        )

    def _part_instants(self, spec: CounterSpec, frame: pd.DataFrame) -> pd.DataFrame:
        """Resolve the instants at which a classified part exists.

        Without ``trigger_uuid`` every reading of the verdict signal is one
        part. With it, the trigger's edges define the parts and the verdict is
        read as of each edge -- the correct reading when the verdict is a
        level-held flag that would otherwise be sampled at arbitrary moments,
        or repeated unchanged for a run of good parts.

        Returns:
            A frame with ``[time_column, "value"]`` where ``value`` is the
            verdict in force at each part instant.
        """
        if spec.trigger_uuid is None:
            return frame
        trigger = self._frame_for(spec, spec.trigger_uuid)
        if trigger.empty or frame.empty:
            return trigger.iloc[0:0] if frame.empty else frame.iloc[0:0]
        edges = trigger[self._edges(trigger["value"], spec.edge)]
        if edges.empty:
            return frame.iloc[0:0]
        verdicts = frame.rename(columns={"value": "_verdict"})
        merged = pd.merge_asof(
            edges[[self.time_column]].reset_index(drop=True),
            verdicts[[self.time_column, "_verdict"]].reset_index(drop=True),
            on=self.time_column,
            direction="backward",
        )
        merged = merged.rename(columns={"_verdict": "value"})
        return merged.dropna(subset=["value"]).reset_index(drop=True)

    def _quality_flag(self, spec: CounterSpec, frame: pd.DataFrame) -> pd.DataFrame:
        """Classify each part by a boolean verdict."""
        parts = self._part_instants(spec, frame)
        if parts.empty:
            return self._empty_long()
        verdict = coerce_bool(parts["value"])
        is_ok = ~verdict if spec.invert else verdict
        kinds = pd.Series(
            np.where(is_ok, CounterKind.OK.value, CounterKind.NOK.value),
            index=parts.index,
            dtype="object",
        )
        quantity = pd.Series(float(spec.parts_per_count), index=parts.index)
        if spec.kind is CounterKind.NOK:
            # Only rejects are being counted from this signal; good parts come
            # from elsewhere, so drop the OK rows instead of inventing them.
            keep = kinds == CounterKind.NOK.value
            parts, kinds, quantity = parts[keep], kinds[keep], quantity[keep]
            if parts.empty:
                return self._empty_long()
        return self._assemble(
            spec,
            parts,
            quantity=quantity,
            kinds=kinds,
            raw=coerce_numeric(parts["value"]),
        )

    def _defect_code(self, spec: CounterSpec, frame: pd.DataFrame) -> pd.DataFrame:
        """Classify each part by a defect code, keeping the code as the reason."""
        parts = self._part_instants(spec, frame)
        if parts.empty:
            return self._empty_long()
        codes = coerce_label(parts["value"], sentinels=())
        allowed = {v.strip().lower() for v in spec.ok_values}
        is_ok = codes.str.strip().str.lower().isin(allowed).fillna(False)
        kinds = pd.Series(
            np.where(is_ok, CounterKind.OK.value, CounterKind.NOK.value),
            index=parts.index,
            dtype="object",
        )
        quantity = pd.Series(float(spec.parts_per_count), index=parts.index)
        defect = codes.where(~is_ok, pd.NA)
        if spec.kind is CounterKind.NOK:
            keep = kinds == CounterKind.NOK.value
            parts, kinds, quantity, defect = (
                parts[keep],
                kinds[keep],
                quantity[keep],
                defect[keep],
            )
            if parts.empty:
                return self._empty_long()
        return self._assemble(
            spec,
            parts,
            quantity=quantity,
            kinds=kinds,
            raw=coerce_numeric(parts["value"]),
            defect_code=defect,
        )

    def _batch_total(self, spec: CounterSpec, frame: pd.DataFrame) -> pd.DataFrame:
        """Take an absolute per-batch quantity, once per change of the value.

        A batch total is republished on every scan cycle by many controllers;
        counting each reading would multiply production by the scan rate. Only
        a *change* of the value is new information.
        """
        values = coerce_numeric(frame["value"])
        changed = values.ne(values.shift()) & values.notna()
        quantity = values.where(changed, 0.0).fillna(0.0)
        is_glitch = pd.Series(False, index=values.index)
        if not spec.allow_negative:
            negative = quantity < 0
            is_glitch |= negative
            quantity = quantity.clip(lower=0.0)
        return self._assemble(
            spec,
            frame,
            quantity=quantity * spec.parts_per_count,
            kinds=spec.kind,
            raw=values,
            is_glitch=is_glitch,
        )

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def _empty_long(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.time_column: pd.Series(dtype="datetime64[ns]"),
                "source_uuid": pd.Series(dtype="object"),
                "kind": pd.Series(dtype="object"),
                "quantity": pd.Series(dtype="float64"),
                "is_reset": pd.Series(dtype="bool"),
                "is_rollover": pd.Series(dtype="bool"),
                "is_glitch": pd.Series(dtype="bool"),
                "defect_code": pd.Series(dtype="object"),
                "raw_value": pd.Series(dtype="float64"),
                "previous_value": pd.Series(dtype="float64"),
            }
        )

    def _assemble(
        self,
        spec: CounterSpec,
        frame: pd.DataFrame,
        *,
        quantity: pd.Series,
        kinds: CounterKind | pd.Series,
        raw: pd.Series,
        is_reset: pd.Series | None = None,
        is_rollover: pd.Series | None = None,
        is_glitch: pd.Series | None = None,
        defect_code: pd.Series | None = None,
        previous: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build the internal long-form frame for one counter spec."""
        index = frame.index
        false_series = pd.Series(False, index=index)
        kind_values = (
            kinds.astype("object")
            if isinstance(kinds, pd.Series)
            else pd.Series(kinds.value, index=index, dtype="object")
        )
        return pd.DataFrame(
            {
                self.time_column: frame[self.time_column].to_numpy(),
                "source_uuid": spec.uuid,
                "kind": kind_values.to_numpy(),
                "quantity": quantity.astype("float64").to_numpy(),
                "is_reset": (false_series if is_reset is None else is_reset).to_numpy(
                    dtype=bool
                ),
                "is_rollover": (
                    false_series if is_rollover is None else is_rollover
                ).to_numpy(dtype=bool),
                "is_glitch": (
                    false_series if is_glitch is None else is_glitch
                ).to_numpy(dtype=bool),
                "defect_code": (
                    pd.Series(pd.NA, index=index, dtype="object")
                    if defect_code is None
                    else defect_code.astype("object")
                ).to_numpy(),
                "raw_value": raw.astype("float64").to_numpy(),
                "previous_value": (
                    pd.Series(np.nan, index=index)
                    if previous is None
                    else previous.astype("float64")
                ).to_numpy(),
            }
        )

    def _normalize(self) -> pd.DataFrame:
        """Normalize every configured counter into one long frame (cached)."""
        if self._events is not None:
            return self._events

        handlers = {
            CounterMode.CUMULATIVE: self._cumulative,
            CounterMode.INCREMENTAL: self._incremental,
            CounterMode.PULSE: self._pulse,
            CounterMode.QUALITY_FLAG: self._quality_flag,
            CounterMode.DEFECT_CODE: self._defect_code,
            CounterMode.BATCH_TOTAL: self._batch_total,
        }
        parts: list[pd.DataFrame] = []
        for spec in self.counters:
            frame = self._frame_for(spec, spec.uuid)
            if frame.empty:
                logger.info(
                    "Counter %s has no readings in the loaded range; it contributes "
                    "nothing to this report.",
                    spec.uuid,
                )
                continue
            parts.append(handlers[spec.mode](spec, frame))

        if parts:
            events = pd.concat(parts, ignore_index=True)
            events = events.sort_values(
                [self.time_column, "source_uuid"], kind="mergesort"
            ).reset_index(drop=True)
        else:
            events = self._empty_long()
        self._events = events
        return events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def count_events(self, *, include_zero: bool = False) -> pd.DataFrame:
        """Return canonical point events, one per counted quantity.

        Args:
            include_zero: Also emit readings that contributed no parts (an
                unchanged counter, a trigger staying low). Off by default:
                across a fleet the zero rows outnumber the real ones by orders
                of magnitude, and every consumer sums ``quantity`` anyway.

        Returns:
            Point-shaped frame with ``systime``, ``uuid``, ``source_uuid``,
            ``is_delta``, ``machine_id``, ``kind`` (``ok`` / ``nok`` /
            ``total`` / ``rework`` / ``scrap``), ``quantity``, ``is_reset``,
            ``is_rollover``, ``defect_code`` and ``raw_value``.

            ``quantity`` is always a *delta* -- the parts attributable to that
            instant -- whatever the source encoding was, which is what makes
            the rows summable over any window or context.
        """
        events = self._normalize()
        if events.empty:
            return empty_event_df("point", _EVENT_COLUMNS[3:])
        out = events
        if not include_zero:
            out = out[out["quantity"].ne(0) | out["is_reset"] | out["is_rollover"]]
        out = out.drop(columns=["is_glitch", "previous_value"]).copy()
        out["is_delta"] = True
        out["machine_id"] = self.machine_id
        out = finalize_point_df(
            out, uuid=self.event_uuid, source_uuid=None, time_col=self.time_column
        )
        return out.reindex(columns=list(_EVENT_COLUMNS)).reset_index(drop=True)

    def anomalies(self) -> pd.DataFrame:
        """Return every reading the normalizer had to interpret.

        Three things happen to real counters, and all three are reported here
        rather than being silently absorbed:

        * ``rollover`` -- the register wrapped; the parts across the wrap were
          recovered (``severity`` ``info``).
        * ``reset`` -- the counter jumped backwards without a configured
          wrap-around; handled per
          :class:`~ts_shape.events.production.signals.spec.ResetPolicy`
          (``warning``).
        * ``implausible_jump`` -- a step beyond ``max_plausible_delta``, or a
          negative step on a counter that should not decrease; **not** counted
          (``error``).

        A machine whose counter appears here every shift has a tag-mapping or
        data-quality problem, not a production problem. That distinction is the
        difference between chasing operators and fixing the gateway.

        Returns:
            Point-shaped frame with ``anomaly``, ``severity``, ``value``,
            ``previous_value`` and the ``quantity`` that was booked.
        """
        events = self._normalize()
        if events.empty:
            return empty_event_df("point", _ANOMALY_COLUMNS[3:])
        flagged = events[
            events["is_reset"] | events["is_rollover"] | events["is_glitch"]
        ].copy()
        if flagged.empty:
            return empty_event_df("point", _ANOMALY_COLUMNS[3:])

        anomaly = pd.Series("reset", index=flagged.index, dtype="object")
        anomaly = anomaly.mask(flagged["is_rollover"], "rollover")
        anomaly = anomaly.mask(flagged["is_glitch"], "implausible_jump")
        severity = anomaly.map(
            {
                "rollover": "info",
                "reset": "warning",
                "implausible_jump": "error",
            }
        )
        out = pd.DataFrame(
            {
                self.time_column: flagged[self.time_column],
                "source_uuid": flagged["source_uuid"],
                "is_delta": True,
                "machine_id": self.machine_id,
                "kind": flagged["kind"],
                "anomaly": anomaly,
                "severity": severity,
                "value": flagged["raw_value"],
                "previous_value": flagged["previous_value"],
                "quantity": flagged["quantity"],
            }
        )
        out = finalize_point_df(
            out,
            uuid=f"{self.event_uuid}:anomaly",
            source_uuid=None,
            time_col=self.time_column,
        )
        return out.reindex(columns=list(_ANOMALY_COLUMNS)).reset_index(drop=True)

    def diagnostics(self) -> pd.DataFrame:
        """Return one health row per configured counter.

        The commissioning and monitoring view for a fleet: which counters
        reported at all, whether they behave like the configured mode, and how
        much they contributed. ``readings == 0`` means the tag is wrong or the
        machine sent nothing; a ``resets`` count in the hundreds means the
        signal is not a totalizer; ``implausible`` above zero means a broken
        gateway.

        Returns:
            Static-shaped frame (no time semantics) with one row per spec:
            ``machine_id``, ``source_uuid``, ``kind``, ``mode``,
            ``value_column``, ``readings``, ``first_seen``, ``last_seen``,
            ``first_value``, ``last_value``, ``monotonic``, ``resets``,
            ``rollovers``, ``implausible``, ``quantity`` and
            ``parts_per_count``.
        """
        events = self._normalize()
        rows: list[dict[str, Any]] = []
        for spec in self.counters:
            frame = self._frame_for(spec, spec.uuid)
            values = (
                coerce_numeric(frame["value"])
                if not frame.empty
                else pd.Series(dtype="float64")
            )
            mine = (
                events[events["source_uuid"] == spec.uuid]
                if not events.empty
                else events
            )
            rows.append(
                {
                    "machine_id": self.machine_id,
                    "source_uuid": spec.uuid,
                    "label": spec.label,
                    "kind": spec.kind.value,
                    "mode": spec.mode.value,
                    "value_column": spec.value_column,
                    "readings": int(len(frame)),
                    "first_seen": (
                        frame[self.time_column].iloc[0] if not frame.empty else pd.NaT
                    ),
                    "last_seen": (
                        frame[self.time_column].iloc[-1] if not frame.empty else pd.NaT
                    ),
                    "first_value": float(values.iloc[0]) if len(values) else np.nan,
                    "last_value": float(values.iloc[-1]) if len(values) else np.nan,
                    "monotonic": (
                        bool(values.dropna().is_monotonic_increasing)
                        if len(values.dropna())
                        else pd.NA
                    ),
                    "resets": int(mine["is_reset"].sum()) if not mine.empty else 0,
                    "rollovers": (
                        int(mine["is_rollover"].sum()) if not mine.empty else 0
                    ),
                    "implausible": (
                        int(mine["is_glitch"].sum()) if not mine.empty else 0
                    ),
                    "quantity": (
                        float(mine["quantity"].sum()) if not mine.empty else 0.0
                    ),
                    "parts_per_count": spec.parts_per_count,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def counts_by_window(
        self,
        freq: str = "1h",
        *,
        group_by: Sequence[str] = (),
        assume_all_ok: bool = False,
        drop_empty: bool = True,
    ) -> pd.DataFrame:
        """Aggregate counted quantities into fixed windows.

        Args:
            freq: Pandas offset alias for the window (``"15min"``, ``"1h"``,
                ``"1D"``, ``"1W"``).
            group_by: Extra columns to group by -- typically context columns
                attached by
                :class:`~ts_shape.events.production.signals.context.ProductionContextResolver`,
                so one call yields production per window *and* per order, part
                or batch.
            assume_all_ok: When the machine measures a total but no quality
                signal at all, treat every part as good. Off by default,
                because reporting 100% quality for a machine that simply does
                not measure it is the most common way an OEE number becomes
                fiction. Turn it on deliberately, per machine, when it is true.
            drop_empty: Drop windows with no production. Keep them (``False``)
                when the report needs a continuous time axis.

        Returns:
            Summary-shaped frame with ``start``, ``end``, ``duration_seconds``,
            the ``group_by`` columns, ``qty_ok``, ``qty_nok``, ``qty_total``,
            ``qty_rework``, ``qty_scrap``, ``good_rate_pct``, ``nok_rate_pct``,
            ``nok_ppm`` and ``machine_id``.

            Quantities that are neither measured nor derivable are ``NaN``, not
            zero -- the report says "unknown", which is a different statement
            from "none".
        """
        events = self.count_events(include_zero=True)
        extra = [c for c in group_by]
        if events.empty:
            return empty_event_df(
                "summary",
                (
                    *extra,
                    *QUANTITY_COLUMNS,
                    "good_rate_pct",
                    "nok_rate_pct",
                    "nok_ppm",
                    "machine_id",
                ),
            )
        missing = [c for c in extra if c not in events.columns]
        if missing:
            raise ValueError(
                f"group_by column(s) {missing} are not present on the count events. "
                "Attach them first, e.g. with "
                "ProductionContextResolver.attach(normalizer.count_events())."
            )

        wide = self._pivot_kinds(events, extra_keys=extra, freq=freq)
        wide = self._derive_quantities(wide, assume_all_ok=assume_all_ok)
        if drop_empty:
            measured = wide[list(QUANTITY_COLUMNS)].fillna(0.0)
            wide = wide[measured.abs().sum(axis=1) > 0]
        wide["machine_id"] = self.machine_id
        out = finalize_summary_df(wide, uuid=self.event_uuid, source_uuid=None)
        return out.reset_index(drop=True)

    def counts_by_context(
        self,
        dimensions: Sequence[str],
        *,
        freq: str | None = None,
        events: pd.DataFrame | None = None,
        assume_all_ok: bool = False,
    ) -> pd.DataFrame:
        """Aggregate counted quantities per production-context combination.

        This is what "how many good parts for customer part 4711 on order
        A-100" means mechanically: group the canonical quantities by the
        context columns already attached to the events.

        Args:
            dimensions: Context columns to group by, e.g.
                ``["order_number", "part_number"]``.
            freq: Optional window inside each context group; ``None`` gives one
                row per context combination over the whole range.
            events: Pre-built events carrying the context columns. Defaults to
                this normalizer's own events, which only works if the context
                was already attached to them.
            assume_all_ok: See :meth:`counts_by_window`.

        Returns:
            Summary-shaped frame as in :meth:`counts_by_window`, with one row
            per context combination (and window).

        Raises:
            ValueError: If a requested dimension is not a column on the events.
        """
        source = self.count_events(include_zero=True) if events is None else events
        dims = list(dimensions)
        if source.empty:
            return empty_event_df(
                "summary",
                (
                    *dims,
                    *QUANTITY_COLUMNS,
                    "good_rate_pct",
                    "nok_rate_pct",
                    "nok_ppm",
                    "machine_id",
                ),
            )
        missing = [d for d in dims if d not in source.columns]
        if missing:
            raise ValueError(
                f"Context dimension(s) {missing} are not columns on the events. "
                "Attach production context first: "
                "ProductionContextResolver(df, specs).attach(events)."
            )
        wide = self._pivot_kinds(source, extra_keys=dims, freq=freq)
        wide = self._derive_quantities(wide, assume_all_ok=assume_all_ok)
        wide["machine_id"] = self.machine_id
        out = finalize_summary_df(wide, uuid=self.event_uuid, source_uuid=None)
        return out.reset_index(drop=True)

    def totals(self, *, assume_all_ok: bool = False) -> pd.DataFrame:
        """Return one summary row for the whole loaded range.

        Args:
            assume_all_ok: See :meth:`counts_by_window`.

        Returns:
            Summary-shaped frame with a single row, or an empty frame when
            nothing was counted.
        """
        events = self.count_events(include_zero=True)
        if events.empty:
            return empty_event_df(
                "summary",
                (
                    *QUANTITY_COLUMNS,
                    "good_rate_pct",
                    "nok_rate_pct",
                    "nok_ppm",
                    "machine_id",
                ),
            )
        wide = self._pivot_kinds(events, extra_keys=[], freq=None)
        wide = self._derive_quantities(wide, assume_all_ok=assume_all_ok)
        wide["machine_id"] = self.machine_id
        return finalize_summary_df(
            wide, uuid=self.event_uuid, source_uuid=None
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Aggregation internals
    # ------------------------------------------------------------------
    def _pivot_kinds(
        self,
        events: pd.DataFrame,
        *,
        extra_keys: Sequence[str],
        freq: str | None,
    ) -> pd.DataFrame:
        """Sum ``quantity`` per kind into ``qty_*`` columns, keyed by window/context."""
        frame = events.copy()
        frame["systime"] = pd.to_datetime(frame["systime"])
        keys: list[Any] = []
        if freq is not None:
            frame["_window"] = frame["systime"].dt.floor(freq)
            keys.append("_window")
        keys.extend(extra_keys)

        if keys:
            grouped = (
                frame.groupby([*keys, "kind"], dropna=False, observed=True)["quantity"]
                .sum()
                .unstack("kind")
                .reset_index()
            )
        else:
            grouped = (
                frame.groupby("kind", dropna=False, observed=True)["quantity"]
                .sum()
                .to_frame()
                .T.reset_index(drop=True)
            )

        grouped = grouped.rename(
            columns={k.value: f"qty_{k.value}" for k in CounterKind}
        )
        # ``unstack`` leaves "kind" as the columns' name, which then prints as a
        # stray header line on every report.
        grouped.columns.name = None
        for column in QUANTITY_COLUMNS:
            if column not in grouped.columns:
                grouped[column] = np.nan

        if freq is not None:
            grouped["start"] = grouped.pop("_window")
            grouped["end"] = grouped["start"] + pd.tseries.frequencies.to_offset(freq)
        else:
            grouped["start"] = frame["systime"].min()
            grouped["end"] = frame["systime"].max()
        return grouped

    def _derive_quantities(
        self, wide: pd.DataFrame, *, assume_all_ok: bool
    ) -> pd.DataFrame:
        """Fill unmeasured quantity columns where arithmetic allows, then rate them."""
        out = wide.copy()
        measured = self.configured_kinds
        has_ok = CounterKind.OK in measured
        has_nok = CounterKind.NOK in measured
        has_total = CounterKind.TOTAL in measured

        # Only *unmeasured* columns are derived; a measured column always keeps
        # what the machine reported, even when the arithmetic disagrees. A
        # mismatch is a data-quality finding, not something to paper over.
        #
        # The zero-versus-unknown distinction is the whole point: a measured kind
        # with no events in a window really produced none of them (a shift with
        # no rejects), so it is 0. An unmeasured kind is unknown, and stays NaN
        # however convenient a 0 would be downstream.
        for column, present in (
            ("qty_ok", has_ok),
            ("qty_nok", has_nok),
            ("qty_total", has_total),
        ):
            out[column] = out[column].fillna(0.0) if present else np.nan

        if not has_total and has_ok and has_nok:
            out["qty_total"] = out["qty_ok"].fillna(0.0) + out["qty_nok"].fillna(0.0)
        elif not has_ok and has_total and has_nok:
            out["qty_ok"] = (
                out["qty_total"].fillna(0.0) - out["qty_nok"].fillna(0.0)
            ).clip(lower=0.0)
        elif not has_nok and has_total and has_ok:
            out["qty_nok"] = (
                out["qty_total"].fillna(0.0) - out["qty_ok"].fillna(0.0)
            ).clip(lower=0.0)
        elif assume_all_ok and has_total and not has_ok and not has_nok:
            out["qty_ok"] = out["qty_total"]
            out["qty_nok"] = 0.0

        for column in ("qty_rework", "qty_scrap"):
            if CounterKind(column.removeprefix("qty_")) in measured:
                out[column] = out[column].fillna(0.0)
            else:
                out[column] = np.nan

        total = out["qty_total"]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["good_rate_pct"] = np.where(
                total.notna() & (total != 0), out["qty_ok"] / total * 100.0, np.nan
            )
            out["nok_rate_pct"] = np.where(
                total.notna() & (total != 0), out["qty_nok"] / total * 100.0, np.nan
            )
            out["nok_ppm"] = np.where(
                total.notna() & (total != 0), out["qty_nok"] / total * 1e6, np.nan
            )
        ordered = [
            c
            for c in out.columns
            if c
            not in {
                *QUANTITY_COLUMNS,
                "good_rate_pct",
                "nok_rate_pct",
                "nok_ppm",
                "start",
                "end",
            }
        ]
        return out[
            [
                "start",
                "end",
                *ordered,
                *QUANTITY_COLUMNS,
                "good_rate_pct",
                "nok_rate_pct",
                "nok_ppm",
            ]
        ]
