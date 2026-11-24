"""Mapping-free order and scrap profiling.

On top of the cycle profile of one PLC
(:class:`~ts_shape.events.production.cycle_profiling.CycleProfileEvents`):

:func:`counter_roles`
    Which counter is the total, the good and the bad count -- learned from
    the identity ``good + bad = total`` per cycle (a lag of up to
    :data:`COUNTER_MAX_LAG_CYCLES` cycles allowed), not from names. Equal
    counters are aliases; a counter with non-integer increments is
    energy-like (a meter).

:class:`OrderProfileEvents`
    *Which orders run slow or scrap the most?* The order signal is found
    with :func:`~ts_shape.signals.identify.identify_objects` (an identifier
    held over many cycles), else the variant signal. Per order: cycles, run
    and stop time, speed against the variant's best, good and bad parts and,
    with a ``plan``, the attainment.

:class:`ScrapProfileEvents`
    *What precedes a bad part?* -- a scrap Pareto without reason codes. Every
    cycle with a bad-counter increment is a NOK unit; its context (a stop in
    the cycle, after a stop, after a changeover, a long cycle, an analog
    deviation, and every rare discrete event in the cycle) is compared with
    the context of good cycles by enrichment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.production._counters import (
    COUNTER_ALIAS_SHARE,
    COUNTER_MAX_LAG_CYCLES,
    COUNTER_SUM_SHARE,
    _CounterMixin,
    counter_roles,
)
from ts_shape.events.production._unit_profile import (
    assign_to_cycles,
    cycle_signal_stats,
    cycles_since,
    enrichment,
    memo,
    potential_register,
    robust_center_scale,
)
from ts_shape.events.production.cycle_profiling import CycleProfileEvents

__all__ = [
    "COUNTER_ALIAS_SHARE",
    "COUNTER_MAX_LAG_CYCLES",
    "COUNTER_SUM_SHARE",
    "OrderProfileEvents",
    "ScrapProfileEvents",
    "counter_roles",
]

#: An identifier held this many median cycles (at least) is an order.
ORDER_MIN_HOLD_CYCLES = 5.0
#: A cycle this close after a stop / changeover is "after" it.
AFTER_STOP_CYCLES = 5
AFTER_CHANGEOVER_CYCLES = 10
#: A discrete event at most this often per cycle is a rare context event.
RARE_EVENTS_PER_CYCLE = 0.2
#: An analog per-cycle statistic this far from its median (robust z).
ANALOG_DEVIATION_Z = 4.0
#: Scrap context needs this lift (it is rarer than a stop cause).
SCRAP_MIN_LIFT = 2.0

_ORDER_RECORDS = [
    "systime",
    "uuid",
    "source_uuid",
    "unit_uuid",
    "start_ns",
    "end_ns",
    "duration_seconds",
    "variant",
    "cycle_count",
    "run_seconds",
    "stop_seconds",
    "median_cycle_seconds",
    "speed_ratio",
    "loss_seconds",
    "good_count",
    "scrap_count",
    "scrap_ratio",
    "planned_quantity",
    "attainment_ratio",
]
_NOK_RECORDS = [
    "systime",
    "uuid",
    "source_uuid",
    "unit_uuid",
    "start_ns",
    "end_ns",
    "duration_seconds",
    "variant",
    "cycle_class",
    "bad_count",
    "context",
    "assigned_context",
]
_PARETO_COLUMNS = [
    "context",
    "nok_count",
    "nok_share_ratio",
    "rate_in_context_ratio",
    "rate_outside_ratio",
    "lift_factor",
    "p_value",
    "q_value",
    "is_significant",
]


class OrderProfileEvents(_CounterMixin):
    """Production orders: setup, speed against the best, scrap, attainment.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=300, seed=2, order_size=25, scrap_probability=0.03)
        >>> orders = ts_shape.OrderProfileEvents(frame)
        >>> orders.evaluation_window()["order_uuid"].iloc[0]
        'plc:order_no'
    """

    PROFILE_KIND = "orders"
    PROFILE_TABLES = ("order_records", "order_register", "counter_roles")
    PROFILE_OPTIONS = ("plan",)
    PROFILE_ORDER = 40

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        plan: pd.DataFrame | None = None,
        order_uuid: str | None = None,
        min_orders: int = 3,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:order_profile",
        **cycle_options: Any,
    ) -> None:
        """Initialize the order profiler.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            plan: Optional plan with ``order_id`` and ``planned_quantity``;
                adds ``planned_quantity`` and ``attainment_ratio``.
            order_uuid: The order signal; ``None`` (default) finds it (an
                identifier held over many cycles, else the variant signal).
            min_orders: Fewer orders are not evaluable.
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the order records.
            **cycle_options: Options of the cycle profile built when
                ``cycle_profile`` is None.
        """
        super().__init__(
            dataframe,
            cycle_profile=cycle_profile,
            signal_uuids=signal_uuids,
            time_column=time_column,
            uuid_column=uuid_column,
            event_uuid=event_uuid,
            cycle_options=cycle_options,
        )
        if plan is not None:
            missing = {"order_id", "planned_quantity"} - set(plan.columns)
            if missing:
                raise ValueError(
                    f"OrderProfileEvents: plan needs the columns {sorted(missing)}"
                )
            dupes = plan["order_id"].astype(str)
            dupes = sorted(set(dupes[dupes.duplicated()]))
            if dupes:
                raise ValueError(
                    f"OrderProfileEvents: plan lists order_id more than once: {dupes[:5]}"
                )
        self.plan = plan
        self.order_uuid = order_uuid
        self.min_orders = int(min_orders)

    @memo
    def _order_signal(self) -> tuple[str | None, str]:
        """``(uuid, how it was found)`` of the order signal."""
        if self.order_uuid is not None:
            return self.order_uuid, "given"
        from ts_shape.signals.identify import identify_objects

        cycles = self._units()
        specs = self._profile().signal_specs()
        found: tuple[str | None, str] = (None, "none")
        candidates = [
            u
            for u, s in specs.items()
            if s.value_column in ("value_string", "value_integer")
            and s.kind != "counter"
        ]
        if candidates and len(cycles):
            median = float(cycles["duration_seconds"].median())
            report = identify_objects(
                self.dataframe, signal_uuids=candidates
            ).to_frame()
            ids = report[report["role"] == "identifier"].copy()
            ids = ids[ids["median_hold_seconds"] >= ORDER_MIN_HOLD_CYCLES * median]
            variant = self._profile().variant_signal()
            ids = ids[ids["uuid"] != variant]
            if len(ids):
                order_types = {"sales_order", "purchase_order", "order", "batch", "lot"}
                ids["is_order"] = ids["object_type"].isin(order_types)
                ids = ids.sort_values(
                    ["is_order", "distinct_count"], ascending=[False, False]
                )
                found = (str(ids["uuid"].iloc[0]), "identifier")
        if found[0] is None and self._profile().variant_signal() is not None:
            found = (self._profile().variant_signal(), "variant")
        return found

    @memo
    def _orders(self) -> pd.DataFrame:
        uuid, _how = self._order_signal()
        cycles = self._cycles()
        columns = ["order_id", "start_ns", "end_ns", "first_cycle", "last_cycle"]
        signals = self._signals()
        if uuid is None or cycles.empty or uuid not in signals:
            return pd.DataFrame(columns=columns)
        spec = self._profile().signal_specs().get(uuid)
        column = spec.value_column if spec is not None else "value_string"
        ns = signals.ns(uuid)
        values = signals.column(uuid, column).astype(object).to_numpy()
        keep = pd.notna(values)
        ns, values = ns[keep], values[keep]
        if len(ns) == 0:
            return pd.DataFrame(columns=columns)
        change = np.r_[True, values[1:] != values[:-1]]
        starts = ns[change]
        ids = values[change]
        ends = np.r_[starts[1:], cycles["end_ns"].max()]
        cstart = cycles["start_ns"].to_numpy(dtype=np.int64)
        records = []
        for oid, a, b in zip(ids, starts, ends, strict=True):
            inside = np.flatnonzero((cstart >= a) & (cstart < b))
            if len(inside) == 0:
                continue
            records.append(
                {
                    "order_id": str(oid),
                    "start_ns": int(a),
                    "end_ns": int(b),
                    "first_cycle": int(inside[0]),
                    "last_cycle": int(inside[-1]),
                }
            )
        return pd.DataFrame(records, columns=columns)

    def order_records(self) -> pd.DataFrame:
        """One record per order.

        Returns:
            Point events at the order end: ``systime``, ``uuid``,
            ``source_uuid`` (the order signal), ``unit_uuid`` (the order id),
            ``start_ns``, ``end_ns``, ``duration_seconds``, ``variant``,
            ``cycle_count``, ``run_seconds`` (non-pause cycles),
            ``stop_seconds`` (pause cycles), ``median_cycle_seconds``,
            ``speed_ratio`` (variant's best cycle / the order's median),
            ``loss_seconds`` (cycle time beyond the variant's best, pause
            cycles included), ``good_count``, ``scrap_count``,
            ``scrap_ratio`` and, with a plan, ``planned_quantity`` and
            ``attainment_ratio``.
        """
        orders = self._orders()
        if orders.empty:
            return self._empty(_ORDER_RECORDS)
        cycles = self._cycles()
        good, bad = self._parts_per_cycle()
        normal = cycles[cycles["cycle_class"] == "normal"]
        best = normal.groupby("variant")["duration_seconds"].quantile(0.1)
        rows = []
        for o in orders.itertuples(index=False):
            part = cycles.iloc[o.first_cycle : o.last_cycle + 1]
            variant = str(part["variant"].mode().iloc[0])
            running = part[part["cycle_class"] != "pause"]
            median = (
                float(running["duration_seconds"].median()) if len(running) else np.nan
            )
            p10 = float(best.get(variant, np.nan))
            g = float(good.iloc[o.first_cycle : o.last_cycle + 1].sum())
            b = float(bad.iloc[o.first_cycle : o.last_cycle + 1].sum())
            rows.append(
                {
                    "unit_uuid": o.order_id,
                    "start_ns": o.start_ns,
                    "end_ns": int(part["end_ns"].max()),
                    "variant": variant,
                    "cycle_count": int(len(part)),
                    "run_seconds": float(running["duration_seconds"].sum()),
                    "stop_seconds": float(
                        part.loc[
                            part["cycle_class"] == "pause", "duration_seconds"
                        ].sum()
                    ),
                    "median_cycle_seconds": median,
                    "speed_ratio": p10 / median if median > 0 else np.nan,
                    "loss_seconds": float(
                        (part["duration_seconds"] - p10).clip(lower=0).sum()
                    ),
                    "good_count": g,
                    "scrap_count": b,
                    "scrap_ratio": b / (g + b) if g + b > 0 else np.nan,
                }
            )
        units = pd.DataFrame(rows)
        extra = units.drop(columns=["unit_uuid", "start_ns", "end_ns"])
        planned = pd.Series(np.nan, index=units.index)
        if self.plan is not None:
            plan = self.plan.assign(order_id=self.plan["order_id"].astype(str))
            by_id = plan.set_index("order_id")["planned_quantity"]
            planned = units["unit_uuid"].map(by_id).astype(float)
        return self._point_records(
            units,
            columns=_ORDER_RECORDS,
            source=self._order_signal()[0],
            **{c: extra[c] for c in extra.columns},
            planned_quantity=planned,
            attainment_ratio=units["good_count"] / planned,
        )

    def counter_roles(self) -> pd.DataFrame:
        """Roles of the counters (total, good, bad ...) learned from the data.

        Returns:
            See :func:`~ts_shape.events.production._counters.counter_roles`.
        """
        return self._roles().copy()

    def order_register(self) -> pd.DataFrame:
        """Orders ranked by lost time, per variant.

        Returns:
            :func:`~ts_shape.events.production._unit_profile.potential_register`
            of the orders' ``loss_seconds`` per ``variant``, plus
            ``scrap_count`` and ``worst_order`` (the order with the most loss).
        """
        records = self.order_records()
        table = records.assign(seconds=records["loss_seconds"])
        register = potential_register(table, keys=["variant"])
        if len(records):
            scrap = records.groupby("variant")["scrap_count"].sum()
            worst = records.loc[
                records.groupby("variant")["loss_seconds"].idxmax(),
                ["variant", "unit_uuid"],
            ].set_index("variant")["unit_uuid"]
            register["scrap_count"] = register["variant"].map(scrap).astype(float)
            register["worst_order"] = register["variant"].map(worst)
        else:
            register["scrap_count"] = pd.Series(dtype=float)
            register["worst_order"] = pd.Series(dtype=object)
        return register

    def evaluation_window(self) -> pd.DataFrame:
        """Whether the order profile can be trusted, and why not.

        Returns:
            One row: ``scope``, ``unit_count`` (orders), ``evaluable``,
            ``reason`` (``no_candidate_signal: order id`` without an order
            or variant signal), ``time_resolution_seconds``, ``order_uuid``
            and ``order_source`` (``given``, ``identifier`` or ``variant``).
        """
        uuid, how = self._order_signal()
        return self._evaluation_row(
            len(self._orders()),
            self.min_orders,
            missing=None if uuid is not None else "order id",
            scope="orders",
            order_uuid=uuid,
            order_source=how,
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``order_count``, ``order_uuid``,
            ``worst_order``, ``worst_order_loss_hours`` and ``scrap_count``.
        """
        records = self.order_records()
        worst = records.loc[records["loss_seconds"].idxmax()] if len(records) else None
        return {
            **self._summary_head("order_count"),
            "order_uuid": self._order_signal()[0],
            "worst_order": None if worst is None else str(worst["unit_uuid"]),
            "worst_order_loss_hours": (
                None if worst is None else float(worst["loss_seconds"] / 3600.0)
            ),
            "scrap_count": float(records["scrap_count"].sum()) if len(records) else 0.0,
        }


class ScrapProfileEvents(_CounterMixin):
    """What precedes a bad part: a scrap Pareto without reason codes.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=600, seed=4, scrap_probability=0.02, rampup_cycles=4,
        ...     curve_faults={"late_rise": 0.05})
        >>> scrap = ts_shape.ScrapProfileEvents(frame)
        >>> scrap.evaluation_window()["bad_uuid"].iloc[0]
        'plc:bad_counter'
    """

    PROFILE_KIND = "scrap"
    PROFILE_TABLES = ("nok_records", "scrap_pareto", "counter_roles")
    PROFILE_ORDER = 50

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        bad_uuid: str | None = None,
        min_nok: int = 10,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:scrap_profile",
        **cycle_options: Any,
    ) -> None:
        """Initialize the scrap profiler.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            bad_uuid: The bad-part counter; ``None`` (default) learns it
                (:func:`counter_roles`).
            min_nok: Fewer cycles with a bad part are not evaluable.
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the NOK records.
            **cycle_options: Options of the cycle profile built when
                ``cycle_profile`` is None.
        """
        super().__init__(
            dataframe,
            cycle_profile=cycle_profile,
            signal_uuids=signal_uuids,
            time_column=time_column,
            uuid_column=uuid_column,
            event_uuid=event_uuid,
            cycle_options=cycle_options,
        )
        self.bad_uuid = bad_uuid
        self.min_nok = int(min_nok)

    def _bad(self) -> tuple[pd.Series, str | None, int]:
        """``(bad parts per cycle, bad uuid, lag)``."""
        inc = self._increments()
        uuid = self.bad_uuid or self._role("bad")
        if uuid is None or uuid not in inc.columns:
            return pd.Series(0.0, index=range(len(inc))), None, 0
        roles = self._roles().set_index("uuid")
        lag = int(roles.at[uuid, "lag_cycles"]) if uuid in roles.index else 0
        # A bad count reported late belongs to the cycle it was made in.
        return self._aligned(uuid), uuid, lag

    @memo
    def _contexts(self) -> list[set[str]]:
        """The context labels of every cycle."""
        cycles = self._cycles()
        n = len(cycles)
        contexts: list[set[str]] = [set() for _ in range(n)]
        if n == 0:
            return contexts
        classes = cycles["cycle_class"].to_numpy(dtype=object)
        restarts = self._restarts()
        pause = classes == "pause"
        since_stop = cycles_since(np.flatnonzero(pause), n)
        since_change = cycles_since(
            restarts.loc[restarts["trigger"] == "changeover", "position"], n
        )
        for i in range(n):
            if pause[i]:
                contexts[i].add("stop_in_cycle")
            elif 0 < since_stop[i] <= AFTER_STOP_CYCLES:
                contexts[i].add("after_stop")
            if 0 <= since_change[i] <= AFTER_CHANGEOVER_CYCLES:
                contexts[i].add("after_changeover")
            if classes[i] == "long":
                contexts[i].add("long_cycle")
            if classes[i] == "short":
                contexts[i].add("short_cycle")
        stats = cycle_signal_stats(
            self._signals(), cycles, self._profile().signal_specs()
        )
        normal = classes == "normal"
        for column in stats.columns:
            ref = stats.loc[normal, column].dropna()
            if len(ref) < 10:
                continue
            median, scale = robust_center_scale(ref.to_numpy())
            if not scale > 0:
                continue
            z = ((stats[column] - median) / scale).to_numpy(dtype=float)
            for i in (int(k) for k in np.flatnonzero(np.abs(z) > ANALOG_DEVIATION_Z)):
                contexts[i].add(f"deviation {column}")
        # Curve shapes that leave the band of all cycles (the cycle profile's
        # curve scores).
        for i, signal in self._curve_anomalies(cycles):
            contexts[i].add(f"curve {signal}")
        # Rare discrete events inside the cycle (not the counters themselves).
        events = self._events()
        counters = {
            u
            for u, spec in self._profile().signal_specs().items()
            if spec.kind == "counter"
        }
        events = events[
            (events["signal_kind"] != "continuous")
            & ~events["source_uuid"].astype(str).isin(counters)
        ]
        counts = events["event"].value_counts()
        rare = set(counts[counts <= RARE_EVENTS_PER_CYCLE * n].index)
        ev = events[events["event"].isin(rare)]
        pos, inside = assign_to_cycles(
            ev["ns"].to_numpy(dtype=np.int64),
            cycles["start_ns"].to_numpy(dtype=np.int64),
            cycles["end_ns"].to_numpy(dtype=np.int64),
        )
        for p, label in zip(
            pos[inside], ev["event"].to_numpy(dtype=object)[inside], strict=True
        ):
            contexts[int(p)].add(f"event {label}")
        return contexts

    def _curve_anomalies(self, cycles: pd.DataFrame) -> list[tuple[int, str]]:
        """``(cycle position, signal)`` of every curve-shape anomaly."""
        from ts_shape.cycle_profile import curve_profiles, signal_inventory

        try:
            records = self._profile().cycle_records()
            _bands, scores, _flagged = curve_profiles(
                self.dataframe,
                records,
                signal_inventory(self.dataframe),
                resolution=self._profile().resolution_seconds(),
            )
        except (KeyError, ValueError):  # no analog signal or no cycle
            return []
        if scores.empty:
            return []
        flagged = scores[scores["is_shape_anomaly"].astype(bool)]
        position = {str(u): i for i, u in enumerate(cycles["cycle_uuid"].astype(str))}
        return [
            (position[str(u)], str(sig))
            for u, sig in zip(flagged["cycle_uuid"], flagged["signal"], strict=True)
            if str(u) in position
        ]

    def nok_records(self) -> pd.DataFrame:
        """One record per cycle with a bad part.

        Returns:
            Point events at the cycle end: ``systime``, ``uuid``,
            ``source_uuid`` (the bad counter), ``unit_uuid`` (the cycle),
            ``start_ns``, ``end_ns``, ``duration_seconds``, ``variant``,
            ``cycle_class``, ``bad_count``, ``context`` (every context label,
            ``;``-joined) and ``assigned_context`` (the significant context
            with the highest lift, or ``unexplained``).
        """
        bad, uuid, _lag = self._bad()
        cycles = self._cycles()
        if uuid is None or cycles.empty:
            return self._empty(_NOK_RECORDS)
        idx = np.flatnonzero(bad.to_numpy(dtype=float) > 0)
        contexts = self._contexts()
        pareto = self.scrap_pareto()
        significant = pareto[pareto["is_significant"].astype(bool)]
        order = significant["context"].tolist()  # highest lift first
        part = cycles.iloc[idx].assign(unit_uuid=lambda d: d["cycle_uuid"].astype(str))
        return self._point_records(
            part,
            columns=_NOK_RECORDS,
            source=uuid,
            variant=part["variant"].astype(str),
            cycle_class=part["cycle_class"],
            bad_count=bad.iloc[idx].to_numpy(dtype=float),
            context=[";".join(sorted(contexts[i])) for i in idx],
            assigned_context=[
                next((c for c in order if c in contexts[i]), "unexplained") for i in idx
            ],
        )

    def counter_roles(self) -> pd.DataFrame:
        """Roles of the counters (total, good, bad ...) learned from the data.

        Returns:
            See :func:`~ts_shape.events.production._counters.counter_roles`.
        """
        return self._roles().copy()

    def scrap_pareto(self) -> pd.DataFrame:
        """Scrap contexts ranked: where bad parts are over-represented.

        Returns:
            One row per context (``stop_in_cycle``, ``after_stop``,
            ``after_changeover``, ``long_cycle``, ``short_cycle``,
            ``deviation <signal>:<stat>``, ``curve <signal>`` (a curve-shape
            anomaly of the cycle profile), ``event <label>``): ``context``,
            ``nok_count`` (bad cycles with it), ``nok_share_ratio``,
            ``rate_in_context_ratio`` (bad-cycle rate with the context),
            ``rate_outside_ratio``, ``lift_factor``, ``p_value``,
            ``q_value`` and ``is_significant`` (q < 0.05, lift >= 2),
            significant contexts first by lift.
        """
        return self._pareto().copy()

    @memo
    def _pareto(self) -> pd.DataFrame:
        columns = _PARETO_COLUMNS
        bad, uuid, _lag = self._bad()
        if uuid is None or bad.empty:
            return pd.DataFrame(columns=columns)
        contexts = self._contexts()
        is_bad = bad.to_numpy(dtype=float) > 0
        nok_sets = [c for c, b in zip(contexts, is_bad, strict=True) if b]
        ok_sets = [c for c, b in zip(contexts, is_bad, strict=True) if not b]
        ev = enrichment(nok_sets, ok_sets, min_units=3, min_lift=SCRAP_MIN_LIFT)
        if ev.empty:
            return pd.DataFrame(columns=columns)
        n_nok = max(len(nok_sets), 1)
        rows = []
        for r in ev.itertuples(index=False):
            with_ctx = sum(r.label in c for c in contexts)
            nok_with = int(r.unit_count)
            outside = len(contexts) - with_ctx
            rows.append(
                {
                    "context": r.label,
                    "nok_count": nok_with,
                    "nok_share_ratio": nok_with / n_nok,
                    "rate_in_context_ratio": (
                        nok_with / with_ctx if with_ctx else np.nan
                    ),
                    "rate_outside_ratio": (
                        (n_nok - nok_with) / outside if outside else np.nan
                    ),
                    "lift_factor": r.lift_factor,
                    "p_value": r.p_value,
                    "q_value": r.q_value,
                    "is_significant": bool(r.is_significant),
                }
            )
        out = pd.DataFrame(rows, columns=columns).sort_values(
            ["is_significant", "lift_factor"], ascending=[False, False], kind="stable"
        )
        return out.reset_index(drop=True)

    def evaluation_window(self) -> pd.DataFrame:
        """Whether the scrap profile can be trusted, and why not.

        Returns:
            One row: ``scope`` (``scrap``), ``unit_count`` (cycles with a bad
            part), ``evaluable``, ``reason`` (``no_candidate_signal: bad
            counter`` without one), ``time_resolution_seconds``, ``bad_uuid``
            and ``explained_share_ratio`` (bad cycles with a significant
            context).
        """
        bad, uuid, _lag = self._bad()
        n = int((bad > 0).sum()) if len(bad) else 0
        records = self.nok_records()
        explained = (
            float((records["assigned_context"] != "unexplained").mean())
            if len(records)
            else np.nan
        )
        return self._evaluation_row(
            n,
            self.min_nok,
            missing=None if uuid is not None else "bad counter",
            scope="scrap",
            bad_uuid=uuid,
            explained_share_ratio=explained,
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``nok_cycle_count``, ``bad_uuid``,
            ``top_context`` (the significant context with most bad cycles)
            and ``explained_share_ratio``.
        """
        window = self.evaluation_window().iloc[0]
        pareto = self._pareto()
        sig = pareto[pareto["is_significant"].astype(bool)] if len(pareto) else pareto
        top = (
            sig.sort_values("nok_count", ascending=False).iloc[0] if len(sig) else None
        )
        share = window["explained_share_ratio"]
        return {
            **self._summary_head("nok_cycle_count"),
            "bad_uuid": window["bad_uuid"],
            "top_context": None if top is None else str(top["context"]),
            "explained_share_ratio": None if pd.isna(share) else float(share),
        }
