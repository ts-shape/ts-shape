"""Counters of one PLC: per-cycle increments and their learned roles.

Shared by the order, scrap and setting-impact profilers. Which counter is
the total, the good and the bad count is learned from the identity
``good + bad = total`` per cycle (a lag of up to
:data:`COUNTER_MAX_LAG_CYCLES` cycles allowed), not from names. Equal
counters are aliases; a counter with non-integer increments is energy-like
(a meter).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events.profiling._base import (
    SignalIndex,
    _UnitProfiler,
    assign_to_cycles,
    memo,
)
from ts_shape.utils.counters import counter_increments

__all__ = ["counter_roles"]

#: ``good + bad = total`` must hold in this share of the cycles.
COUNTER_SUM_SHARE = 0.98
#: Lags (in cycles) tried between the counters of an identity.
COUNTER_MAX_LAG_CYCLES = 3
#: Two counters equal in this share of the cycles in which either moves are
#: aliases ...
COUNTER_ALIAS_SHARE = 0.98
#: ... judged on at least this many such cycles.
ALIAS_MIN_MOVES = 5

_ROLE_COLUMNS = [
    "uuid",
    "role",
    "rate_per_cycle",
    "total_uuid",
    "lag_cycles",
    "identity_share_ratio",
    "confidence",
]


def increments_per_cycle(
    signals: SignalIndex,
    cycles: pd.DataFrame,
    specs: dict[str, Any],
) -> pd.DataFrame:
    """Reset-aware increments of every counter, summed per cycle.

    A count reported at a cycle boundary belongs to the cycle that just
    ended (the part is counted when it leaves): ``(start, end]``.
    """
    out = pd.DataFrame(index=cycles.index)
    if cycles.empty:
        return out
    starts = cycles["start_ns"].to_numpy(dtype=np.int64)
    ends = cycles["end_ns"].to_numpy(dtype=np.int64)
    columns = {}
    for uuid, spec in specs.items():
        if spec.kind != "counter" or uuid not in signals:
            continue
        inc = counter_increments(signals.column(uuid, spec.value_column)).to_numpy(
            dtype=float
        )
        pos, inside = assign_to_cycles(signals.ns(uuid), starts, ends, closed="right")
        ok = inside & np.isfinite(inc)
        sums = pd.Series(inc[ok]).groupby(pos[ok]).sum()
        columns[uuid] = sums.reindex(range(len(cycles)), fill_value=0.0).to_numpy()
    return pd.DataFrame(columns, index=cycles.index) if columns else out


def counter_roles(increments: pd.DataFrame) -> pd.DataFrame:
    """Roles of the counters from their per-cycle increments.

    Args:
        increments: One row per cycle, one column per counter (see
            :class:`ScrapProfileEvents` / :class:`OrderProfileEvents`, which
            build it from the signals).

    Returns:
        One row per counter: ``uuid``, ``role`` (``total``, ``good``,
        ``bad``, ``alias``, ``energy_like`` or ``counter``),
        ``rate_per_cycle``, ``total_uuid`` (the total it adds up to, or the
        counter it is an alias of), ``lag_cycles`` (the counter aligned with
        the total is ``increments.shift(lag_cycles)``: -1 when it reports
        one cycle late), ``identity_share_ratio`` and ``confidence``
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
    moving = [u for u in counters if rate[u] > 0]
    # The identity first: with little scrap the good count equals the total
    # in most cycles, and an alias test run first would swallow it.
    best: tuple[float, str, str, str, int] | None = None
    for total in moving:
        for a, b in itertools.combinations([c for c in moving if c != total], 2):
            for lag in range(-COUNTER_MAX_LAG_CYCLES, COUNTER_MAX_LAG_CYCLES + 1):
                summed = data[a] + data[b].shift(lag).fillna(0.0)
                share = float(np.mean(np.isclose(summed, data[total])))
                if share >= COUNTER_SUM_SHARE and (best is None or share > best[0]):
                    best = (share, total, a, b, lag)
    if best is not None:
        share, total, a, b, lag = best
        good, bad = (a, b) if rate[a] >= rate[b] else (b, a)
        roles[total].update(role="total", share=share)
        # ``lag`` aligns ``b`` with the total (``b.shift(lag)``); ``a`` is aligned.
        roles[good].update(
            role="good",
            total_uuid=total,
            share=share,
            lag_cycles=lag if good == b else 0,
        )
        roles[bad].update(
            role="bad", total_uuid=total, share=share, lag_cycles=lag if bad == b else 0
        )
    # Aliases: the same count, judged on the cycles in which either moves (two
    # rarely moving counters are not equal because both are mostly zero).
    for a, b in itertools.combinations(counters, 2):
        if roles[b]["role"] != "counter":
            continue
        moved = (data[a] != 0) | (data[b] != 0)
        if int(moved.sum()) < ALIAS_MIN_MOVES:
            continue
        share = float(np.mean(np.isclose(data.loc[moved, a], data.loc[moved, b])))
        if share >= COUNTER_ALIAS_SHARE:
            roles[b].update(role="alias", total_uuid=a, share=share)
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
    """Per-cycle counter increments and roles (orders, scrap, settings)."""

    @memo
    def _increments(self) -> pd.DataFrame:
        return increments_per_cycle(
            self._signals(), self._cycles(), self._profile().signal_specs()
        )

    @memo
    def _roles(self) -> pd.DataFrame:
        return counter_roles(self._increments())

    def _aligned(self, uuid: str) -> pd.Series:
        """``uuid``'s increments per cycle, aligned with the total (lag)."""
        inc = self._increments()[uuid].reset_index(drop=True)
        roles = self._roles().set_index("uuid")
        lag = int(roles.at[uuid, "lag_cycles"]) if uuid in roles.index else 0
        return inc.shift(lag).fillna(0.0) if lag else inc

    def _role(self, role: str) -> str | None:
        roles = self._roles()
        match = roles[roles["role"] == role]
        return None if match.empty else str(match["uuid"].iloc[0])

    def _parts_per_cycle(self) -> tuple[pd.Series, pd.Series]:
        """``(good, bad)`` per cycle (bad 0 and good = total without roles)."""
        inc = self._increments()
        bad_uuid, good_uuid = self._role("bad"), self._role("good")
        n = len(inc)
        bad = self._aligned(bad_uuid) if bad_uuid else pd.Series(0.0, index=range(n))
        if good_uuid:
            good = self._aligned(good_uuid)
        else:
            anchor = self._profile().cycle_anchor()
            source = anchor[1] if anchor else None
            good = (
                inc[source] if source in inc.columns else pd.Series(1.0, index=range(n))
            )
        return good.reset_index(drop=True), bad.reset_index(drop=True)
