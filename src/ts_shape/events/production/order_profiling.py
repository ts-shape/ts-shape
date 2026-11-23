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

import itertools
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.production._unit_profile import (
    REASON_NO_CANDIDATE,
    _UnitProfiler,
    cycle_signal_stats,
    enrichment,
    evaluation_frame,
    evaluation_reason,
    potential_register,
)
from ts_shape.events.production.cycle_profiling import CycleProfileEvents
from ts_shape.utils.counters import counter_increments
from ts_shape.utils.frames import to_ns

__all__ = ["OrderProfileEvents", "ScrapProfileEvents", "counter_roles"]

#: ``good + bad = total`` must hold in this share of the cycles.
COUNTER_SUM_SHARE = 0.98
#: Lags (in cycles) tried between the counters of an identity.
COUNTER_MAX_LAG_CYCLES = 3
#: Two counters equal in this share of the cycles are aliases.
COUNTER_ALIAS_SHARE = 0.98
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

_ROLE_COLUMNS = [
    "uuid",
    "role",
    "rate_per_cycle",
    "total_uuid",
    "lag_cycles",
    "identity_share_ratio",
    "confidence",
]


def _increments_per_cycle(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    specs: dict[str, Any],
    *,
    uuid_column: str,
    time_column: str,
) -> pd.DataFrame:
    """Reset-aware increments of every counter, summed per cycle."""
    out = pd.DataFrame(index=cycles.index)
    if cycles.empty:
        return out
    starts = cycles["start_ns"].to_numpy(dtype=np.int64)
    ends = cycles["end_ns"].to_numpy(dtype=np.int64)
    for uuid, spec in specs.items():
        if spec.kind != "counter":
            continue
        rows = frame[frame[uuid_column].astype(str) == uuid]
        rows = rows.iloc[np.argsort(to_ns(rows[time_column]), kind="stable")]
        ns = np.asarray(to_ns(rows[time_column]), dtype=np.int64)
        inc = counter_increments(rows[spec.value_column]).to_numpy(dtype=float)
        # A count reported at a cycle boundary belongs to the cycle that just
        # ended (the part is counted when it leaves): (start, end].
        pos = np.searchsorted(starts, ns, side="left") - 1
        ok = (pos >= 0) & np.isfinite(inc)
        ok[ok] &= ns[ok] <= ends[pos[ok]]
        sums = pd.Series(inc[ok]).groupby(pos[ok]).sum()
        out[uuid] = sums.reindex(range(len(cycles)), fill_value=0.0).to_numpy()
    return out


def counter_roles(increments: pd.DataFrame) -> pd.DataFrame:
    """Roles of the counters from their per-cycle increments.

    Args:
        increments: One row per cycle, one column per counter (see
            :class:`ScrapProfileEvents` / :class:`OrderProfileEvents`, which
            build it from the signals).

    Returns:
        One row per counter: ``uuid``, ``role`` (``total``, ``good``,
        ``bad``, ``alias``, ``energy_like`` or ``counter``),
        ``rate_per_cycle``, ``total_uuid`` (the total it adds up to),
        ``lag_cycles``, ``identity_share_ratio`` and ``confidence``
        (``high`` for an identity, ``low`` otherwise).
    """
    if increments.empty or increments.shape[1] == 0:
        return pd.DataFrame(columns=_ROLE_COLUMNS)
    data = increments.fillna(0.0)
    rate = data.mean()
    roles = {
        u: {"role": "counter", "total_uuid": None, "lag_cycles": 0, "share": np.nan}
        for u in data.columns
    }
    for u in data.columns:
        v = data[u].to_numpy(dtype=float)
        if len(v) and not np.allclose(v, np.round(v)):
            roles[u]["role"] = "energy_like"
    counters = [u for u in data.columns if roles[u]["role"] == "counter"]
    for a, b in itertools.combinations(counters, 2):
        share = float(np.mean(np.isclose(data[a], data[b])))
        if share >= COUNTER_ALIAS_SHARE and rate[a] > 0:
            roles[b].update(role="alias", total_uuid=a, share=share)
    counters = [u for u in counters if roles[u]["role"] == "counter"]
    best: tuple[float, str, str, str, int] | None = None
    for total in counters:
        for a, b in itertools.combinations([c for c in counters if c != total], 2):
            for lag in range(-COUNTER_MAX_LAG_CYCLES, COUNTER_MAX_LAG_CYCLES + 1):
                summed = data[a] + data[b].shift(lag).fillna(0.0)
                share = float(np.mean(np.isclose(summed, data[total])))
                if share >= COUNTER_SUM_SHARE and (best is None or share > best[0]):
                    best = (share, total, a, b, lag)
    if best is not None:
        share, total, a, b, lag = best
        good, bad = (a, b) if rate[a] >= rate[b] else (b, a)
        roles[total].update(role="total", share=share)
        roles[good].update(role="good", total_uuid=total, share=share)
        roles[bad].update(role="bad", total_uuid=total, share=share, lag_cycles=lag)
    rows = [
        {
            "uuid": u,
            "role": r["role"],
            "rate_per_cycle": float(rate[u]),
            "total_uuid": r["total_uuid"],
            "lag_cycles": int(r["lag_cycles"] or 0),
            "identity_share_ratio": r["share"],
            "confidence": "high" if r["role"] in ("total", "good", "bad") else "low",
        }
        for u, r in roles.items()
    ]
    return pd.DataFrame(rows, columns=_ROLE_COLUMNS)


class _CounterMixin(_UnitProfiler):
    """Per-cycle counter increments and roles, shared by orders and scrap."""

    def _increments(self) -> pd.DataFrame:
        if "increments" not in self._cache:
            cycles = self._units().reset_index(drop=True)
            self._cache["increments"] = _increments_per_cycle(
                self.dataframe,
                cycles,
                self._profile().signal_specs(),
                uuid_column=self.uuid_column,
                time_column=self.time_column,
            )
        out: pd.DataFrame = self._cache["increments"]
        return out

    def counter_roles(self) -> pd.DataFrame:
        """Roles of the counters (total, good, bad ...) learned from the data.

        Returns:
            See :func:`counter_roles`.
        """
        if "roles" not in self._cache:
            self._cache["roles"] = counter_roles(self._increments())
        out: pd.DataFrame = self._cache["roles"]
        return out

    def _role(self, role: str) -> str | None:
        roles = self.counter_roles()
        match = roles[roles["role"] == role]
        return None if match.empty else str(match["uuid"].iloc[0])

    def _parts_per_cycle(self) -> tuple[pd.Series, pd.Series]:
        """``(good, bad)`` per cycle (bad 0 and good = total without roles)."""
        inc = self._increments()
        bad_uuid, good_uuid = self._role("bad"), self._role("good")
        n = len(inc)
        bad = inc[bad_uuid] if bad_uuid else pd.Series(0.0, index=range(n))
        if good_uuid:
            good = inc[good_uuid]
        else:
            anchor = self._profile().cycle_anchor()
            source = anchor[1] if anchor else None
            good = (
                inc[source] if source in inc.columns else pd.Series(1.0, index=range(n))
            )
        return good.reset_index(drop=True), bad.reset_index(drop=True)


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
        self.plan = plan
        self.order_uuid = order_uuid
        self.min_orders = int(min_orders)

    def _order_signal(self) -> tuple[str | None, str]:
        """``(uuid, how it was found)`` of the order signal."""
        if self.order_uuid is not None:
            return self.order_uuid, "given"
        if "order_signal" in self._cache:
            out: tuple[str | None, str] = self._cache["order_signal"]
            return out
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
        self._cache["order_signal"] = found
        return found

    def _orders(self) -> pd.DataFrame:
        if "orders" in self._cache:
            out: pd.DataFrame = self._cache["orders"]
            return out
        uuid, _how = self._order_signal()
        cycles = self._units().reset_index(drop=True)
        columns = ["order_id", "start_ns", "end_ns", "first_cycle", "last_cycle"]
        if uuid is None or cycles.empty:
            out = pd.DataFrame(columns=columns)
            self._cache["orders"] = out
            return out
        rows = self.dataframe[self.dataframe[self.uuid_column].astype(str) == uuid]
        spec = self._profile().signal_specs().get(uuid)
        column = spec.value_column if spec is not None else "value_string"
        ns = np.asarray(to_ns(rows[self.time_column]), dtype=np.int64)
        order = np.argsort(ns, kind="stable")
        ns = ns[order]
        values = rows[column].iloc[order].astype(object).to_numpy()
        keep = pd.notna(values)
        ns, values = ns[keep], values[keep]
        if len(ns) == 0:
            out = pd.DataFrame(columns=columns)
            self._cache["orders"] = out
            return out
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
        out = pd.DataFrame(records, columns=columns)
        self._cache["orders"] = out
        return out

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
        columns = [
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
        if orders.empty:
            return pd.DataFrame(columns=columns)
        cycles = self._units().reset_index(drop=True)
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
        out = pd.DataFrame(rows)
        out["duration_seconds"] = (out["end_ns"] - out["start_ns"]) / 1e9
        out["systime"] = self._times(out["end_ns"].to_numpy())
        out["uuid"] = self.event_uuid
        out["source_uuid"] = self._order_signal()[0]
        if self.plan is not None:
            plan = self.plan.assign(order_id=self.plan["order_id"].astype(str))
            planned = plan.set_index("order_id")["planned_quantity"]
            out["planned_quantity"] = out["unit_uuid"].map(planned).astype(float)
            out["attainment_ratio"] = out["good_count"] / out["planned_quantity"]
        else:
            out["planned_quantity"] = np.nan
            out["attainment_ratio"] = np.nan
        return out[columns]

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
        profile = self._profile()
        uuid, how = self._order_signal()
        orders = self._orders()
        if self._units().empty:
            evaluable, reason = (
                False,
                f"upstream_not_evaluable: {profile.evaluation_reason()}",
            )
        elif uuid is None:
            evaluable, reason = False, f"{REASON_NO_CANDIDATE}: order id"
        else:
            evaluable, reason = evaluation_reason(len(orders), self.min_orders)
        return evaluation_frame(
            [
                {
                    "scope": "orders",
                    "unit_count": int(len(orders)),
                    "evaluable": evaluable,
                    "reason": reason,
                    "time_resolution_seconds": profile.resolution_seconds(),
                    "order_uuid": uuid,
                    "order_source": how,
                }
            ],
            ["order_uuid", "order_source"],
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``order_count``, ``order_uuid``,
            ``worst_order``, ``worst_order_loss_hours`` and ``scrap_count``.
        """
        window = self.evaluation_window().iloc[0]
        records = self.order_records()
        worst = records.loc[records["loss_seconds"].idxmax()] if len(records) else None
        return {
            "evaluable": bool(window["evaluable"]),
            "reason": str(window["reason"]),
            "order_count": int(window["unit_count"]),
            "order_uuid": window["order_uuid"],
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
        roles = self.counter_roles().set_index("uuid")
        lag = int(roles.at[uuid, "lag_cycles"]) if uuid in roles.index else 0
        bad = inc[uuid].reset_index(drop=True)
        # The bad count reported ``lag`` cycles late belongs to the earlier cycle.
        return bad.shift(-lag).fillna(0.0) if lag else bad, uuid, lag

    def _contexts(self) -> list[set[str]]:
        """The context labels of every cycle."""
        if "contexts" in self._cache:
            out: list[set[str]] = self._cache["contexts"]
            return out
        cycles = self._units().reset_index(drop=True)
        n = len(cycles)
        contexts: list[set[str]] = [set() for _ in range(n)]
        if n == 0:
            self._cache["contexts"] = contexts
            return contexts
        classes = cycles["cycle_class"].to_numpy(dtype=object)
        variant = cycles["variant"].astype(str).to_numpy()
        last_stop = -(10**9)
        last_change = -(10**9)
        for i in range(n):
            if i > 0 and variant[i] != variant[i - 1]:
                last_change = i
            if classes[i] == "pause":
                contexts[i].add("stop_in_cycle")
                last_stop = i
            elif 0 < i - last_stop <= AFTER_STOP_CYCLES:
                contexts[i].add("after_stop")
            if 0 <= i - last_change <= AFTER_CHANGEOVER_CYCLES:
                contexts[i].add("after_changeover")
            if classes[i] == "long":
                contexts[i].add("long_cycle")
            if classes[i] == "short":
                contexts[i].add("short_cycle")
        stats = cycle_signal_stats(
            self.dataframe,
            cycles,
            self._profile().signal_specs(),
            uuid_column=self.uuid_column,
            time_column=self.time_column,
        )
        normal = classes == "normal"
        for column in stats.columns:
            ref = stats.loc[normal, column].dropna()
            if len(ref) < 10:
                continue
            median = float(ref.median())
            mad = 1.4826 * float((ref - median).abs().median())
            scale = mad if mad > 0 else float(ref.std())
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
        starts = cycles["start_ns"].to_numpy(dtype=np.int64)
        ends = cycles["end_ns"].to_numpy(dtype=np.int64)
        ns = ev["ns"].to_numpy(dtype=np.int64)
        pos = np.searchsorted(starts, ns, side="right") - 1
        ok = pos >= 0
        ok[ok] &= ns[ok] < ends[pos[ok]]
        for p, label in zip(
            pos[ok], ev["event"].to_numpy(dtype=object)[ok], strict=True
        ):
            contexts[int(p)].add(f"event {label}")
        self._cache["contexts"] = contexts
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
        columns = [
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
        bad, uuid, _lag = self._bad()
        cycles = self._units().reset_index(drop=True)
        if uuid is None or cycles.empty:
            return pd.DataFrame(columns=columns)
        idx = np.flatnonzero(bad.to_numpy(dtype=float) > 0)
        contexts = self._contexts()
        pareto = self.scrap_pareto()
        significant = pareto[pareto["is_significant"].astype(bool)]
        order = significant["context"].tolist()  # highest lift first
        part = cycles.iloc[idx]
        out = pd.DataFrame(
            {
                "systime": self._times(part["end_ns"].to_numpy()),
                "uuid": self.event_uuid,
                "source_uuid": uuid,
                "unit_uuid": part["cycle_uuid"].astype(str).to_numpy(),
                "start_ns": part["start_ns"].to_numpy(dtype=np.int64),
                "end_ns": part["end_ns"].to_numpy(dtype=np.int64),
                "duration_seconds": part["duration_seconds"].to_numpy(dtype=float),
                "variant": part["variant"].astype(str).to_numpy(),
                "cycle_class": part["cycle_class"].to_numpy(),
                "bad_count": bad.iloc[idx].to_numpy(dtype=float),
                "context": [";".join(sorted(contexts[i])) for i in idx],
                "assigned_context": [
                    next((c for c in order if c in contexts[i]), "unexplained")
                    for i in idx
                ],
            }
        )
        return out[columns]

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
        columns = [
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
        if "pareto" in self._cache:
            cached: pd.DataFrame = self._cache["pareto"]
            return cached
        bad, uuid, _lag = self._bad()
        if uuid is None or bad.empty:
            return pd.DataFrame(columns=columns)
        contexts = self._contexts()
        is_bad = bad.to_numpy(dtype=float) > 0
        nok_sets = [c for c, b in zip(contexts, is_bad, strict=True) if b]
        ok_sets = [c for c, b in zip(contexts, is_bad, strict=True) if not b]
        ev = enrichment(nok_sets, ok_sets, min_units=3, min_lift=SCRAP_MIN_LIFT)
        if ev.empty:
            out = pd.DataFrame(columns=columns)
            self._cache["pareto"] = out
            return out
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
        out = out.reset_index(drop=True)
        self._cache["pareto"] = out
        return out

    def evaluation_window(self) -> pd.DataFrame:
        """Whether the scrap profile can be trusted, and why not.

        Returns:
            One row: ``scope`` (``scrap``), ``unit_count`` (cycles with a bad
            part), ``evaluable``, ``reason`` (``no_candidate_signal: bad
            counter`` without one), ``time_resolution_seconds``, ``bad_uuid``
            and ``explained_share_ratio`` (bad cycles with a significant
            context).
        """
        profile = self._profile()
        bad, uuid, _lag = self._bad()
        n = int((bad > 0).sum()) if len(bad) else 0
        if self._units().empty:
            evaluable, reason = (
                False,
                f"upstream_not_evaluable: {profile.evaluation_reason()}",
            )
        elif uuid is None:
            evaluable, reason = False, f"{REASON_NO_CANDIDATE}: bad counter"
        else:
            evaluable, reason = evaluation_reason(n, self.min_nok)
        records = self.nok_records()
        explained = (
            float((records["assigned_context"] != "unexplained").mean())
            if len(records)
            else np.nan
        )
        return evaluation_frame(
            [
                {
                    "scope": "scrap",
                    "unit_count": n,
                    "evaluable": evaluable,
                    "reason": reason,
                    "time_resolution_seconds": profile.resolution_seconds(),
                    "bad_uuid": uuid,
                    "explained_share_ratio": explained,
                }
            ],
            ["bad_uuid", "explained_share_ratio"],
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``nok_cycle_count``, ``bad_uuid``,
            ``top_context`` (the significant context with most bad cycles)
            and ``explained_share_ratio``.
        """
        window = self.evaluation_window().iloc[0]
        pareto = self.scrap_pareto()
        sig = pareto[pareto["is_significant"].astype(bool)] if len(pareto) else pareto
        top = (
            sig.sort_values("nok_count", ascending=False).iloc[0] if len(sig) else None
        )
        share = window["explained_share_ratio"]
        return {
            "evaluable": bool(window["evaluable"]),
            "reason": str(window["reason"]),
            "nok_cycle_count": int(window["unit_count"]),
            "bad_uuid": window["bad_uuid"],
            "top_context": None if top is None else str(top["context"]),
            "explained_share_ratio": None if pd.isna(share) else float(share),
        }
