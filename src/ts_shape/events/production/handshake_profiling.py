"""Mapping-free handshake profiling: who makes the machine wait?

:class:`HandshakeProfileEvents` finds request / acknowledge pairs between the
boolean signals of one PLC without being told which signal talks to which:
an edge ``A`` that is (almost) always answered by an edge ``B`` of another
signal before ``A`` happens again, and ``B`` (almost) never comes without a
preceding ``A`` -- ``robot_ready↑ → press_open↑``, ``req_conveyor↑ →
ack_conveyor↑``. Per pair:

* the waiting time of every occurrence and its register (median, P10, P90,
  potential = median - P10);
* ``pair_kind``: ``four_phase`` when the full request/acknowledge protocol
  is seen (A↑ → B↑ → A↓ → B↓), else ``two_phase``;
* ``waits_on``: ``partner`` when the answer time varies (someone else
  decides when to answer), ``fixed`` when it is a constant delay (a timer or
  a fixed sequence in the same program);
* with cycles: the share of the cycle the wait takes and how much of the
  cycle time variance it explains.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ts_shape.events.production._unit_profile import (
    FALL,
    REASON_OK,
    RISE,
    _UnitProfiler,
    assign_to_cycles,
    evaluation_frame,
    evaluation_reason,
    memo,
    potential_register,
)
from ts_shape.events.production.cycle_profiling import (
    MAX_EVENTS_PER_CYCLE,
    CycleProfileEvents,
)

__all__ = ["HandshakeProfileEvents"]

#: A request is answered (and an answer has a request) at least this often.
MIN_FOLLOW_RATIO = 0.95
#: The answer's P90 delay is at most this share of the request interval.
HANDSHAKE_MAX_DELAY_SHARE = 0.25
#: Edge counts of the two sides within this ratio.
COUNT_RATIO = (0.8, 1.25)
#: At most this many candidate edge labels (the most frequent).
HANDSHAKE_MAX_LABELS = 200
#: The resolution must be below half the median wait: ``evaluation_reason``
#: tests ``resolution < 0.25 x median``, so the wait is scaled by this.
RESOLUTION_SHARE_FACTOR = 0.25 / 0.5
#: The answer is the next edge of another signal after the request in at
#: least this share of the occurrences (direct succession; a step later in
#: the sequence is not an answer).
DIRECT_SHARE = 0.8
#: A delay whose (P90 - P10) / median is below this is a fixed delay.
FIXED_DELAY_SPREAD = 0.15

_RECORD_COLUMNS = ["start", "end", "duration_seconds", "uuid", "source_uuid", "pair"]
_PAIR_COLUMNS = [
    "pair",
    "request",
    "answer",
    "request_uuid",
    "answer_uuid",
    "pair_kind",
    "occurrence_count",
    "follow_ratio",
    "precede_ratio",
    "median_delay_seconds",
    "waits_on",
]


def _next_after(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each ``a``: the first ``b`` after it, and whether it comes before the next ``a``."""
    idx = np.searchsorted(b, a, side="right")
    has = idx < len(b)
    nxt = np.full(len(a), np.iinfo(np.int64).max, dtype=np.int64)
    nxt[has] = b[idx[has]]
    next_a = np.r_[a[1:], np.iinfo(np.int64).max]
    return nxt, has & (nxt < next_a)


def _ratios(a: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """``(follow ratio, precede ratio, delays ns of answered requests)``."""
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0, np.zeros(0, dtype=np.int64)
    nxt, ok = _next_after(a, b)
    follow = float(ok.mean())
    # Every answer has a request since the previous answer.
    idx = np.searchsorted(a, b, side="left") - 1
    has = idx >= 0
    prev_b = np.r_[np.iinfo(np.int64).min, b[:-1]]
    precede_ok = np.zeros(len(b), dtype=bool)
    precede_ok[has] = a[idx[has]] > prev_b[has]
    return follow, float(precede_ok.mean()), (nxt[ok] - a[ok])


def _direct_share(
    ns_a: np.ndarray,
    source_a: str,
    b_index: int,
    all_ns: np.ndarray,
    all_label: np.ndarray,
    all_source: np.ndarray,
    max_look: int = 50,
) -> float:
    """Share of requests whose next edge of another signal is the answer."""
    if len(ns_a) == 0:
        return 0.0
    start = np.searchsorted(all_ns, ns_a, side="right")
    hits = 0
    for i in start:
        j = i
        while j < len(all_ns) and j < i + max_look and all_source[j] == source_a:
            j += 1
        if j < len(all_ns) and all_label[j] == b_index:
            hits += 1
    return hits / len(ns_a)


class HandshakeProfileEvents(_UnitProfiler):
    """Request / acknowledge pairs and the time spent waiting on them.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=300, seed=1, handshake_partner=True)
        >>> pairs = ts_shape.HandshakeProfileEvents(frame).handshake_pairs()
        >>> "plc:req_conveyor↑ → plc:ack_conveyor↑" in set(pairs["pair"])
        True
    """

    PROFILE_KIND = "handshakes"
    PROFILE_TABLES = ("handshake_pairs", "handshake_records", "handshake_register")
    PROFILE_ORDER = 60

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        min_follow_ratio: float = MIN_FOLLOW_RATIO,
        min_occurrences: int = 30,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:handshake_profile",
        **cycle_options: Any,
    ) -> None:
        """Initialize the handshake profiler.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            min_follow_ratio: A request must be answered, and an answer
                preceded by a request, at least this often.
            min_occurrences: Edges seen fewer times are not candidates; a pair
                with fewer occurrences is not evaluable.
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the handshake records.
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
        self.min_follow_ratio = float(min_follow_ratio)
        self.min_occurrences = int(min_occurrences)

    @memo
    def _edges(self) -> dict[str, tuple[str, np.ndarray]]:
        """Candidate edge label -> (source uuid, sorted ns)."""
        events = self._events()
        events = events[events["signal_kind"] == "bool"]
        cycles = self._units()
        counts = events["event"].value_counts()
        counts = counts[counts >= self.min_occurrences]
        if len(cycles):
            counts = counts[counts <= MAX_EVENTS_PER_CYCLE * len(cycles)]
        counts = counts.head(HANDSHAKE_MAX_LABELS)
        out = {}
        keep = events[events["event"].isin(set(counts.index))]
        groups = keep.groupby("event", sort=False)
        for label in counts.index:
            part = groups.get_group(label)
            out[str(label)] = (
                str(part["source_uuid"].iloc[0]),
                np.sort(part["ns"].to_numpy(dtype=np.int64)),
            )
        return out

    @memo
    def _pairs(self) -> dict[str, Any]:
        edges = self._edges()
        # The next edge of each request, among the other signals' edges.
        labels = list(edges)
        label_index = {x: i for i, x in enumerate(labels)}
        all_ns = (
            np.concatenate([edges[x][1] for x in labels])
            if labels
            else np.zeros(0, np.int64)
        )
        all_label = (
            np.concatenate([np.full(len(edges[x][1]), i) for i, x in enumerate(labels)])
            if labels
            else np.zeros(0, np.int64)
        )
        all_source = np.array([edges[labels[i]][0] for i in all_label], dtype=object)
        order = np.argsort(all_ns, kind="stable")
        all_ns, all_label, all_source = (
            all_ns[order],
            all_label[order],
            all_source[order],
        )
        found: dict[tuple[str, str], dict[str, Any]] = {}
        for a, b in itertools.permutations(edges, 2):
            source_a, ns_a = edges[a]
            source_b, ns_b = edges[b]
            if source_a == source_b:
                continue
            ratio = len(ns_b) / len(ns_a)
            if not COUNT_RATIO[0] <= ratio <= COUNT_RATIO[1]:
                continue
            follow, precede, delays = _ratios(ns_a, ns_b)
            if follow < self.min_follow_ratio or precede < self.min_follow_ratio:
                continue
            if len(delays) < self.min_occurrences:
                continue
            interval = float(np.median(np.diff(ns_a))) if len(ns_a) > 1 else np.inf
            if np.percentile(delays, 90) > HANDSHAKE_MAX_DELAY_SHARE * interval:
                continue
            direct = _direct_share(
                ns_a, source_a, label_index[b], all_ns, all_label, all_source
            )
            if direct < DIRECT_SHARE:
                continue
            found[(a, b)] = {
                "direct": direct,
                "follow": follow,
                "precede": precede,
                "delays": delays,
                "source_a": source_a,
                "source_b": source_b,
            }
        # A request is answered by its first answer: keep, per request, the
        # answer with the shortest median delay (per answering signal pair).
        best: dict[tuple[str, str], tuple[str, str]] = {}
        for (a, b), info in found.items():
            key = (a, info["source_b"])
            if key not in best or np.median(info["delays"]) < np.median(
                found[best[key]]["delays"]
            ):
                best[key] = (a, b)
        kept = {k: found[k] for k in best.values()}
        return {"found": found, "kept": kept}

    def _kind(self, a: str, b: str) -> str:
        found = self._pairs()["found"]
        if a.endswith(RISE) and b.endswith(RISE):
            a_fall, b_fall = a[: -len(RISE)] + FALL, b[: -len(RISE)] + FALL
            if (b, a_fall) in found and (a_fall, b_fall) in found:
                return "four_phase"
        return "two_phase"

    def handshake_pairs(self) -> pd.DataFrame:
        """The request / answer pairs found, with their evidence.

        Returns:
            One row per pair: ``pair`` (``"<request> → <answer>"``),
            ``request``, ``answer``, ``request_uuid``, ``answer_uuid``,
            ``pair_kind`` (``four_phase`` / ``two_phase``),
            ``occurrence_count``, ``follow_ratio``, ``precede_ratio``,
            ``median_delay_seconds`` and ``waits_on`` (``partner`` for a
            varying answer time, ``fixed`` for a constant delay).
        """
        kept = self._pairs()["kept"]
        rows = []
        for (a, b), info in kept.items():
            d = info["delays"] / 1e9
            median = float(np.median(d))
            spread = float(np.percentile(d, 90) - np.percentile(d, 10))
            rows.append(
                {
                    "pair": f"{a} → {b}",
                    "request": a,
                    "answer": b,
                    "request_uuid": info["source_a"],
                    "answer_uuid": info["source_b"],
                    "pair_kind": self._kind(a, b),
                    "occurrence_count": int(len(d)),
                    "follow_ratio": info["follow"],
                    "precede_ratio": info["precede"],
                    "median_delay_seconds": median,
                    "waits_on": (
                        "fixed"
                        if median > 0 and spread / median < FIXED_DELAY_SPREAD
                        else "partner"
                    ),
                }
            )
        out = pd.DataFrame(rows, columns=_PAIR_COLUMNS)
        return out.sort_values(
            ["waits_on", "median_delay_seconds"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)

    def handshake_records(self) -> pd.DataFrame:
        """One interval per answered request.

        Returns:
            Intervals: ``start`` (request), ``end`` (answer),
            ``duration_seconds`` (the wait), ``uuid``, ``source_uuid`` (the
            answering signal), ``pair``.
        """
        kept = self._pairs()["kept"]
        edges = self._edges()
        frames = []
        for (a, b), info in kept.items():
            ns_a = edges[a][1]
            nxt, ok = _next_after(ns_a, edges[b][1])
            frames.append(
                pd.DataFrame(
                    {
                        "start": self._times(ns_a[ok]),
                        "end": self._times(nxt[ok]),
                        "duration_seconds": (nxt[ok] - ns_a[ok]) / 1e9,
                        "uuid": self.event_uuid,
                        "source_uuid": info["source_b"],
                        "pair": f"{a} → {b}",
                    }
                )
            )
        if not frames:
            return self._empty(_RECORD_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values("start", kind="stable").reset_index(drop=True)
        return out[_RECORD_COLUMNS]

    def handshake_register(self) -> pd.DataFrame:
        """What each handshake wait costs, linked to the cycle.

        Returns:
            :func:`~ts_shape.events.production._unit_profile.potential_register`
            of the waits per ``pair`` plus ``pair_kind``, ``waits_on``,
            ``cycle_share_ratio`` (median wait / median cycle),
            ``cycle_spearman_factor`` (rank correlation of the wait with the
            duration of its cycle) and ``variance_contribution_factor``
            (cov(wait, cycle) / var(cycle) over the normal cycles).
        """
        records = self.handshake_records()
        table = records.rename(columns={"duration_seconds": "seconds"})
        register = potential_register(table, keys=["pair"])
        pairs = self.handshake_pairs().set_index("pair")
        register["pair_kind"] = register["pair"].map(pairs["pair_kind"])
        register["waits_on"] = register["pair"].map(pairs["waits_on"])
        cycles = self._units()
        share, rho, var_share = [], [], []
        normal = cycles[cycles["cycle_class"] == "normal"]
        starts = normal["start_ns"].to_numpy(dtype=np.int64)
        ends = normal["end_ns"].to_numpy(dtype=np.int64)
        durations = normal["duration_seconds"].to_numpy(dtype=float)
        median_cycle = float(np.median(durations)) if len(durations) else np.nan
        edges = self._edges()
        for pair in register["pair"]:
            a, b = pairs.at[pair, "request"], pairs.at[pair, "answer"]
            ns_a = edges[a][1]
            nxt, ok = _next_after(ns_a, edges[b][1])
            share.append(
                float(np.median((nxt[ok] - ns_a[ok]) / 1e9)) / median_cycle
                if np.isfinite(median_cycle)
                else np.nan
            )
            if len(starts) < 10:
                rho.append(np.nan)
                var_share.append(np.nan)
                continue
            pos, inside = assign_to_cycles(ns_a[ok], starts, ends)
            waits = (
                pd.Series((nxt[ok] - ns_a[ok])[inside] / 1e9).groupby(pos[inside]).sum()
            )
            cyc = durations[waits.index.to_numpy()]
            if len(waits) < 10 or len(np.unique(cyc)) < 2 or waits.nunique() < 2:
                rho.append(np.nan)
                var_share.append(np.nan)
                continue
            rho.append(float(sps.spearmanr(waits.to_numpy(), cyc).statistic))
            var_share.append(
                float(np.cov(waits.to_numpy(), cyc)[0, 1] / np.var(cyc, ddof=1))
            )
        register["cycle_share_ratio"] = share
        register["cycle_spearman_factor"] = rho
        register["variance_contribution_factor"] = var_share
        return register

    def evaluation_window(self) -> pd.DataFrame:
        """Whether each pair can be trusted, and why not.

        Returns:
            One row per pair (``scope`` = the pair; one ``all`` row when none
            is found): ``unit_count`` (answered requests), ``evaluable``,
            ``reason`` (``below_resolution`` when the median wait is shorter
            than twice the time resolution) and ``time_resolution_seconds``.
        """
        resolution = self._resolution()
        pairs = self.handshake_pairs()
        rows = []
        for r in pairs.itertuples(index=False):
            evaluable, reason = evaluation_reason(
                int(r.occurrence_count),
                self.min_occurrences,
                resolution=resolution,
                # A wait is resolved when the resolution is below half of it.
                median_seconds=2.0 * RESOLUTION_SHARE_FACTOR * r.median_delay_seconds,
            )
            rows.append(
                {
                    "scope": r.pair,
                    "unit_count": int(r.occurrence_count),
                    "evaluable": evaluable,
                    "reason": reason,
                    "time_resolution_seconds": resolution,
                }
            )
        if not rows:
            # Handshakes are edges of the signals: no cycle is needed.
            return self._evaluation_row(
                0,
                self.min_occurrences,
                missing="request/answer pair",
                upstream=REASON_OK,
                resolution=resolution,
            )
        return evaluation_frame(rows)

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable`` (any pair evaluable), ``reason``, ``pair_count``,
            ``partner_pair_count`` and ``top_wait_pair`` (the partner pair
            with the largest total potential, four-phase pairs first).
        """
        pairs = self.handshake_pairs()
        register = self.handshake_register()
        partner = register[register["waits_on"] == "partner"].copy()
        # A full request/acknowledge protocol is the strongest evidence.
        partner["_four"] = partner["pair_kind"] == "four_phase"
        partner = partner.sort_values(
            ["_four", "total_potential_seconds"], ascending=[False, False]
        )
        top = partner.iloc[0] if len(partner) else None
        return {
            **self._summary_head(),
            "pair_count": int(len(pairs)),
            "partner_pair_count": int((pairs["waits_on"] == "partner").sum()),
            "top_wait_pair": None if top is None else str(top["pair"]),
        }
