"""Attach production context -- order, batch, part, serial, customer part -- to events.

A production number on its own is not a report. "1,240 parts, 18 rejects" only
becomes actionable as "1,240 parts on order A-100, part 4711-B for customer
Bosch, batch 22-07, on tool T-3, by operator 4412". Every one of those is a
separate sparse signal that has to be joined to the counted parts *as it was in
force at that instant*.

The joins are where reports go wrong, and the failure modes are specific:

* **Stale context.** An order number written once at order start, forward-filled
  without limit, attaches last month's order to today's parts. ``valid_for``
  bounds how long a value stays believable.
* **Late-bound identifiers.** A serial number, a measured weight or a test
  verdict is published *after* the part that carries it. A backward as-of join
  attaches it to the *next* part -- off by one, systematically, forever.
  :attr:`~ts_shape.events.production.signals.spec.ContextBinding.NEXT_KNOWN`
  exists for exactly this.
* **Placeholders read as real values.** ``""``, ``"n/a"``, ``"#NV"``, ``"0"``
  become their own row in every ``GROUP BY``, splitting one order across two
  lines of the report.
* **Cosmetic variation.** ``" a-100 "``, ``"A-100"`` and ``"000100"`` are one
  order that reports as three.

:class:`ProductionContextResolver` handles all of it declaratively, one
:class:`~ts_shape.events.production.signals.spec.ContextSpec` per dimension, and
then :meth:`~ProductionContextResolver.attach` puts the columns on any event
frame -- counts, cycles, state intervals -- so production, quality and cycle time
can all be grouped the same way.

Example::

    from ts_shape.events.production.signals import (
        ContextBinding, ContextSpec, ProductionContextResolver,
    )

    context = ProductionContextResolver(
        df,
        specs=[
            ContextSpec(name="order_number", uuid="m1.order", valid_for="12h",
                        upper=True),
            ContextSpec(name="part_number", uuid="m1.part", upper=True),
            ContextSpec(name="customer_part_number", uuid="m1.cust_part"),
            # published *after* the part it identifies:
            ContextSpec(name="serial_number", uuid="m1.serial",
                        binding=ContextBinding.NEXT_KNOWN, valid_for="1min",
                        zero_is_sentinel=True),
        ],
        machine_id="M1",
    )

    events = context.attach(counter_normalizer.count_events())
    counter_normalizer.counts_by_context(["order_number", "part_number"],
                                         events=events)
    context.segments(["order_number"])   # when did each order run?
    context.coverage()                   # how much of the span is identified?
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import (
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
)
from ts_shape.events.production.signals._intervals import collapse_runs
from ts_shape.events.production.signals.spec import (
    ContextBinding,
    ContextSpec,
)
from ts_shape.events.production.signals.values import (
    DEFAULT_SENTINELS,
    coerce_label,
    coerce_numeric,
    signal_frame,
)
from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

__all__ = ["ProductionContextResolver"]

_DIRECTIONS: dict[ContextBinding, str] = {
    ContextBinding.LAST_KNOWN: "backward",
    ContextBinding.NEXT_KNOWN: "forward",
    ContextBinding.NEAREST: "nearest",
}


class ProductionContextResolver(Base):
    """Resolve production-context signals and attach them to event frames.

    Args:
        dataframe: Timeseries in the standard ts-shape schema.
        specs: One :class:`~ts_shape.events.production.signals.spec.ContextSpec`
            per dimension to resolve.
        time_column: Timestamp column.
        machine_id: Asset identifier broadcast onto emitted frames.
        event_uuid: ``uuid`` value stamped on emitted events.
        uuid_column: Column holding signal identifiers.

    Raises:
        ValueError: If ``specs`` is empty or two specs share a ``name``.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        specs: Sequence[ContextSpec],
        *,
        time_column: str = "systime",
        machine_id: str | None = None,
        event_uuid: str = "prod:context",
        uuid_column: str = "uuid",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        if not specs:
            raise ValueError(
                "ProductionContextResolver requires at least one ContextSpec."
            )
        names = [s.name for s in specs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate context dimension name(s) {duplicates}; each name becomes "
                "one output column, so they must be unique."
            )
        self.specs = tuple(specs)
        self.time_column = time_column
        self.machine_id = machine_id
        self.event_uuid = event_uuid
        self.uuid_column = uuid_column
        self._resolved: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def dimensions(self) -> tuple[str, ...]:
        """Configured dimension names, in declaration order."""
        return tuple(s.name for s in self.specs)

    def spec_for(self, name: str) -> ContextSpec:
        """Return the spec for one dimension.

        Raises:
            KeyError: If no dimension has that name.
        """
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(
            f"No context dimension {name!r}. Configured: {list(self.dimensions)}"
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def _resolve(self, spec: ContextSpec) -> pd.DataFrame:
        """Return cleaned ``[time, value]`` readings for one dimension (cached).

        Placeholder readings become ``pd.NA``. Whether they are *kept* as an
        explicit "context ended" marker or dropped so the previous value carries
        on is the ``clear_on_sentinel`` decision -- see
        :class:`~ts_shape.events.production.signals.spec.ContextSpec`.
        """
        if spec.name in self._resolved:
            return self._resolved[spec.name]

        frame = signal_frame(
            self.dataframe,
            spec.uuid,
            time_column=self.time_column,
            value_column=spec.value_column,
            uuid_column=self.uuid_column,
        )
        if frame.empty:
            logger.info(
                "Context signal %s (%s) has no readings in the loaded range.",
                spec.uuid,
                spec.name,
            )
            out = pd.DataFrame(
                {
                    self.time_column: pd.Series(dtype="datetime64[ns]"),
                    "value": pd.Series(dtype="object"),
                }
            )
            self._resolved[spec.name] = out
            return out

        if spec.numeric:
            values: pd.Series = coerce_numeric(frame["value"])
        else:
            values = coerce_label(
                frame["value"],
                sentinels=(*DEFAULT_SENTINELS, *spec.sentinels),
                upper=spec.upper,
                zero_is_sentinel=spec.zero_is_sentinel,
                strip_leading_zeros=spec.strip_leading_zeros,
            )
        out = pd.DataFrame(
            {
                self.time_column: pd.to_datetime(frame[self.time_column]).astype(
                    "datetime64[ns]"
                ),
                "value": values.to_numpy(),
            }
        )
        if not spec.clear_on_sentinel:
            out = out[out["value"].notna()]
        # Collapse consecutive repeats: a value republished every scan cycle adds
        # nothing to an as-of join and makes changes() unusable.
        keep = out["value"].astype("string").ne(out["value"].astype("string").shift())
        keep.iloc[0] = True if len(keep) else False
        out = out[keep.fillna(True)].reset_index(drop=True)
        self._resolved[spec.name] = out
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attach(
        self,
        events: pd.DataFrame,
        *,
        time_column: str | None = None,
        dimensions: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Return ``events`` with one column added per context dimension.

        The events keep their original order and index; only columns are added.
        Works on any canonical shape -- point events are joined on ``systime``,
        interval and summary rows on ``start``.

        Args:
            events: Frame to enrich, e.g. from
                :meth:`~ts_shape.events.production.signals.counters.ProductionCounterNormalizer.count_events`
                or :meth:`~ts_shape.events.production.signals.cycles.CycleTimeNormalizer.cycles`.
            time_column: Timestamp column to join on. Auto-detected from
                ``systime`` then ``start`` when omitted.
            dimensions: Restrict to these dimensions; defaults to all.
            overwrite: Replace a column that already exists. Off by default so
                a second ``attach`` cannot silently clobber context that came
                from somewhere else.

        Returns:
            A copy of ``events`` with the context columns appended. Rows with no
            value in force carry ``pd.NA`` -- unknown context, reported as
            unknown.

        Raises:
            ValueError: If no timestamp column can be found, or a target column
                exists and ``overwrite`` is false.
        """
        wanted = list(dimensions) if dimensions is not None else list(self.dimensions)
        specs = [self.spec_for(name) for name in wanted]
        out = events.copy()
        if out.empty:
            for spec in specs:
                if spec.name not in out.columns:
                    out[spec.name] = pd.Series(dtype="object")
            return out

        join_column = time_column
        if join_column is None:
            for candidate in ("systime", "start", self.time_column):
                if candidate in out.columns:
                    join_column = candidate
                    break
        if join_column is None or join_column not in out.columns:
            raise ValueError(
                "Cannot attach context: no timestamp column found on the events "
                f"(looked for 'systime', 'start', {self.time_column!r}). "
                "Pass time_column= explicitly."
            )

        existing = [s.name for s in specs if s.name in out.columns]
        if existing and not overwrite:
            raise ValueError(
                f"Column(s) {existing} already exist on the events. Pass "
                "overwrite=True to replace them, or select other dimensions."
            )

        # merge_asof needs a sorted key; remember the original positions so the
        # caller's row order (and index) survives untouched.
        probe = pd.DataFrame(
            {"_t": pd.to_datetime(out[join_column]).astype("datetime64[ns]").to_numpy()}
        )
        order = np.argsort(probe["_t"].to_numpy(), kind="mergesort")
        sorted_probe = probe.iloc[order].reset_index(drop=True)

        for spec in specs:
            resolved = self._resolve(spec)
            if resolved.empty:
                out[spec.name] = pd.NA
                continue
            lookup = resolved.rename(
                columns={self.time_column: "_t", "value": spec.name}
            )
            tolerance = (
                pd.to_timedelta(spec.valid_for) if spec.valid_for is not None else None
            )
            merged = pd.merge_asof(
                sorted_probe,
                lookup,
                on="_t",
                direction=_DIRECTIONS[spec.binding],
                tolerance=tolerance,
            )
            column = np.empty(len(out), dtype=object)
            column[order] = merged[spec.name].to_numpy()
            out[spec.name] = pd.Series(column, index=out.index)
        return out

    def timeline(self) -> pd.DataFrame:
        """Return every context value, long-form, one row per reading.

        Returns:
            Point-shaped frame with ``systime``, ``uuid``, ``source_uuid``,
            ``machine_id``, ``dimension``, ``value`` and ``previous_value``,
            sorted by time. Repeated identical readings are collapsed, so each
            row is a real change.
        """
        extras = ("machine_id", "dimension", "value", "previous_value")
        parts = []
        for spec in self.specs:
            resolved = self._resolve(spec)
            if resolved.empty:
                continue
            parts.append(
                pd.DataFrame(
                    {
                        self.time_column: resolved[self.time_column],
                        "source_uuid": spec.uuid,
                        "machine_id": self.machine_id,
                        "dimension": spec.name,
                        "value": resolved["value"],
                        "previous_value": resolved["value"].shift(),
                    }
                )
            )
        if not parts:
            return empty_event_df("point", extras)
        combined = (
            pd.concat(parts, ignore_index=True)
            .sort_values([self.time_column, "dimension"], kind="mergesort")
            .reset_index(drop=True)
        )
        out = finalize_point_df(
            combined, uuid=self.event_uuid, source_uuid=None, time_col=self.time_column
        )
        return out.reindex(columns=["systime", "uuid", "source_uuid", *extras])

    def changes(self, *, dimensions: Sequence[str] | None = None) -> pd.DataFrame:
        """Return point events where a context dimension changed value.

        These are the changeover events of a production line: an order change, a
        part change, a tool change. Filtering on ``dimension`` gives the
        changeover log for that dimension.

        Args:
            dimensions: Restrict to these dimensions; defaults to all.

        Returns:
            Point-shaped frame as :meth:`timeline`, excluding each dimension's
            first reading (which has nothing to change from) -- plus
            ``transition`` (``"A-100->A-101"``). A context ending or starting is
            a change too, rendered as ``"A-101-><none>"`` /
            ``"<none>->A-102"``.
        """
        wanted = set(dimensions) if dimensions is not None else set(self.dimensions)
        timeline = self.timeline()
        extras = (
            "machine_id",
            "dimension",
            "value",
            "previous_value",
            "transition",
        )
        if timeline.empty:
            return empty_event_df("point", extras)
        selected = timeline[timeline["dimension"].isin(wanted)]
        if selected.empty:
            return empty_event_df("point", extras)
        # Drop each dimension's first reading only. Filtering on "has a previous
        # value" would also drop a genuine change *out of* a cleared context.
        is_first = selected.groupby("dimension", sort=False).cumcount().eq(0)
        out = selected[~is_first.to_numpy()].copy()
        if out.empty:
            return empty_event_df("point", extras)
        out["transition"] = (
            out["previous_value"].astype("string").fillna("<none>")
            + "->"
            + out["value"].astype("string").fillna("<none>")
        )
        return out.reset_index(drop=True)

    def segments(
        self,
        dimensions: Sequence[str] | None = None,
        *,
        until: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Return intervals over which the given context stayed constant.

        Answers "when did order A-100 run, and on what". One row per unbroken
        run of one context combination, which is also the natural key for
        joining production numbers back to an ERP order or a batch record.

        Args:
            dimensions: Dimensions forming the combination; defaults to all.
            until: Closing timestamp for the final, still-open segment. Defaults
                to the last timestamp in the input data -- so when the context
                changes on the very last reading, that final segment has no
                measured duration and is omitted. Pass the end of the reporting
                window to see it.

        Returns:
            Interval-shaped frame with ``start``, ``end``, ``duration_seconds``,
            ``uuid``, ``source_uuid``, ``machine_id`` and one column per
            dimension. Zero-length segments are omitted.
        """
        wanted = list(dimensions) if dimensions is not None else list(self.dimensions)
        specs = [self.spec_for(name) for name in wanted]
        if until is not None:
            horizon = pd.Timestamp(until)
        elif not self.dataframe.empty and self.time_column in self.dataframe.columns:
            horizon = pd.to_datetime(
                self.dataframe[self.time_column], errors="coerce"
            ).max()
        else:
            horizon = None

        instants = sorted(
            {
                timestamp
                for spec in specs
                for timestamp in self._resolve(spec)[self.time_column].tolist()
            }
        )
        if not instants:
            return empty_event_df("interval", ("machine_id", *wanted))

        probe = pd.DataFrame({self.time_column: pd.to_datetime(instants)})
        snapshot = self.attach(probe, time_column=self.time_column, dimensions=wanted)
        snapshot["_key"] = (
            snapshot[wanted].astype("string").fillna("<none>").agg("\x1f".join, axis=1)
        )
        runs = collapse_runs(
            snapshot,
            value_column="_key",
            time_column=self.time_column,
            until=horizon,
            carry_columns=tuple(wanted),
        )
        runs = runs.drop(columns=["_key"])
        runs = runs[runs["duration_seconds"] > 0].reset_index(drop=True)
        if runs.empty:
            return empty_event_df("interval", ("machine_id", *wanted))
        runs["machine_id"] = self.machine_id
        out = finalize_interval_df(
            runs,
            uuid=f"{self.event_uuid}:segment",
            source_uuid=",".join(s.uuid for s in specs),
        )
        return out.reset_index(drop=True)

    def coverage(self) -> pd.DataFrame:
        """Return one row per dimension describing how well it is populated.

        The report to read before trusting any context breakdown. A dimension
        with ``fill_pct`` of 40 means 60% of the production in this range cannot
        be attributed to an order -- so an order-level report silently covers
        less than half the output. ``distinct_values`` of 1 usually means the tag
        is a constant, not an identifier.

        Returns:
            Static-shaped frame with ``machine_id``, ``dimension``,
            ``source_uuid``, ``binding``, ``readings``, ``changes``,
            ``distinct_values``, ``placeholder_readings``, ``first_seen``,
            ``last_seen`` and ``fill_pct`` (share of the observed span covered
            by a non-placeholder value).
        """
        span_start = span_end = None
        if not self.dataframe.empty and self.time_column in self.dataframe.columns:
            times = pd.to_datetime(self.dataframe[self.time_column], errors="coerce")
            span_start, span_end = times.min(), times.max()
        span_seconds = (
            (span_end - span_start).total_seconds()
            if span_start is not None and span_end is not None
            else 0.0
        )

        rows: list[dict[str, Any]] = []
        for spec in self.specs:
            raw = signal_frame(
                self.dataframe,
                spec.uuid,
                time_column=self.time_column,
                value_column=spec.value_column,
                uuid_column=self.uuid_column,
            )
            resolved = self._resolve(spec)
            valid = (
                resolved[resolved["value"].notna()] if not resolved.empty else resolved
            )

            filled = 0.0
            if not resolved.empty and span_seconds > 0:
                times = resolved[self.time_column]
                ends = times.shift(-1)
                ends.iloc[-1] = span_end
                covered = (ends - times).dt.total_seconds()
                filled = float(covered[resolved["value"].notna()].sum())

            placeholders = 0
            if not raw.empty:
                if spec.numeric:
                    placeholders = int(coerce_numeric(raw["value"]).isna().sum())
                else:
                    cleaned = coerce_label(
                        raw["value"],
                        sentinels=(*DEFAULT_SENTINELS, *spec.sentinels),
                        upper=spec.upper,
                        zero_is_sentinel=spec.zero_is_sentinel,
                    )
                    placeholders = int(cleaned.isna().sum())

            rows.append(
                {
                    "machine_id": self.machine_id,
                    "dimension": spec.name,
                    "source_uuid": spec.uuid,
                    "binding": spec.binding.value,
                    "readings": int(len(raw)),
                    "changes": int(max(len(resolved) - 1, 0)),
                    "distinct_values": (
                        int(valid["value"].nunique()) if not valid.empty else 0
                    ),
                    "placeholder_readings": placeholders,
                    "first_seen": (
                        valid[self.time_column].iloc[0] if not valid.empty else pd.NaT
                    ),
                    "last_seen": (
                        valid[self.time_column].iloc[-1] if not valid.empty else pd.NaT
                    ),
                    "fill_pct": (
                        filled / span_seconds * 100.0 if span_seconds > 0 else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows)
