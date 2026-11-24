"""Mapping-free setting-change impact: did that parameter change help?

:class:`SettingImpactEvents` finds the *held settings* among the signals of
one PLC -- numeric signals that change rarely and then hold their value for
many cycles (a speed override, a temperature or pressure set value, a hold
time) -- and compares the cycles before and after every change, a natural
before/after experiment:

* metrics: cycle time (non-pause cycles), every segment of the cycle
  profile's standard cycle, the bad-part rate and the energy per cycle when
  the counters for them are found
  (:func:`~ts_shape.events.production.order_profiling.counter_roles`);
* effect: shift (Hodges-Lehmann), Cliff's delta and a Mann-Whitney p-value,
  Benjamini-Hochberg q-values over all tests;
* windows: only cycles of the variant running at the change are compared
  (cycle times differ between variants);
* confounders: another setting change within the windows, a variant change
  or a pause at the change, and a trend already running before the change;
* confidence: ``high`` (unconfounded, q < 0.05, enough cycles), ``medium``
  (significant but confounded by a pause at the change or another
  setting) or ``low``.

A setting whose changes all come with a variant change is a recipe
parameter: listed, not tested (its effect cannot be told from the
variant's).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ts_shape.events.production._counters import _CounterMixin
from ts_shape.events.production._unit_profile import bh_qvalues, effect_size, memo
from ts_shape.events.production.cycle_profiling import CycleProfileEvents

__all__ = ["SettingImpactEvents"]

#: A setting holds its value for at least this many cycles (median) ...
SETTING_MIN_HOLD_CYCLES = 10
#: ... so it changes at most this often per cycle.
SETTING_MAX_CHANGES_PER_CYCLE = 1.0 / SETTING_MIN_HOLD_CYCLES
#: A change within this many cycles of a variant change comes with it.
WITH_VARIANT_CYCLES = 1
#: Raw span (in windows) searched for cycles of the variant at the change.
VARIANT_SPAN = 3
#: Significance level of the before/after tests.
FDR = 0.05


class SettingImpactEvents(_CounterMixin):
    """Before/after effect of every change of a held setting.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(
        ...     n_cycles=400, seed=3, setting_changes=[(150, 130.0)])
        >>> impact = ts_shape.SettingImpactEvents(frame).setting_impact()
        >>> row = impact[impact["metric"] == "cycle_seconds"].iloc[0]
        >>> bool(row["shift"] < 0), row["confidence"]
        (True, 'high')
    """

    PROFILE_KIND = "settings"
    PROFILE_TABLES = (
        "setting_changes",
        "setting_impact",
        "setting_register",
    )
    PROFILE_ORDER = 80

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        cycle_profile: CycleProfileEvents | None = None,
        signal_uuids: list[str] | None = None,
        window_cycles: int = 30,
        settle_cycles: int = 3,
        min_window_cycles: int = 10,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        event_uuid: str = "prod:setting_impact",
        **cycle_options: Any,
    ) -> None:
        """Initialize the setting-impact profiler.

        Args:
            dataframe: Long-format signals of one PLC.
            cycle_profile: A computed cycle profile of the same signals to
                reuse; ``None`` (default) builds one.
            signal_uuids: Signals to use (default: all).
            window_cycles: Cycles compared before and after a change (at
                most; a window ends at the previous or next setting change).
            settle_cycles: Cycles right after a change left out (the process
                settles).
            min_window_cycles: A side with fewer cycles makes the test
                ``low`` confidence.
            time_column: Timestamp column.
            uuid_column: Signal column.
            event_uuid: ``uuid`` of the change records.
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
        self.window_cycles = int(window_cycles)
        self.settle_cycles = int(settle_cycles)
        self.min_window_cycles = int(min_window_cycles)

    # ------------------------------------------------------------------
    # Settings and their changes
    # ------------------------------------------------------------------

    @memo
    def _settings(self) -> pd.DataFrame:
        """Held settings: uuid, change times and values."""
        cycles = self._units()
        signals = self._signals()
        specs = self._profile().signal_specs()
        variant = self._profile().variant_signal()
        rows = []
        n = max(len(cycles), 1)
        median_cycle = (
            float(cycles["duration_seconds"].median()) if len(cycles) else 0.0
        )
        for uuid, spec in specs.items():
            if (
                spec.kind == "counter"
                or uuid == variant
                or spec.value_column not in ("value_double", "value_integer")
            ):
                continue
            ns = signals.ns(uuid)
            values = signals.numeric(uuid, spec.value_column)
            ok = np.isfinite(values)
            ns, values = ns[ok], values[ok]
            if len(values) < 2:
                continue
            change = np.flatnonzero(values[1:] != values[:-1]) + 1
            if len(change) == 0 or len(change) > SETTING_MAX_CHANGES_PER_CYCLE * n:
                continue
            holds = np.diff(np.r_[ns[0], ns[change], ns[-1]]) / 1e9
            if (
                median_cycle > 0
                and np.median(holds) < SETTING_MIN_HOLD_CYCLES * median_cycle
            ):
                continue
            for k in change.tolist():
                rows.append(
                    {
                        "setting": uuid,
                        "change_ns": int(ns[k]),
                        "before_value": float(values[k - 1]),
                        "after_value": float(values[k]),
                    }
                )
        return pd.DataFrame(
            rows, columns=["setting", "change_ns", "before_value", "after_value"]
        )

    @memo
    def _metrics(self) -> pd.DataFrame:
        """One row per cycle: start, class, variant and every metric."""
        cycles = self._cycles()
        out = cycles[
            ["cycle_uuid", "start_ns", "end_ns", "variant", "cycle_class"]
        ].copy()
        out["cycle_seconds"] = cycles["duration_seconds"].to_numpy(dtype=float)
        segments = self._profile().segment_durations()
        if len(segments):
            pivot = segments.pivot_table(
                index="cycle_uuid",
                columns="segment",
                values="duration_seconds",
                aggfunc="sum",
            )
            pivot.columns = [f"segment {c}" for c in pivot.columns]
            out = out.merge(pivot, left_on="cycle_uuid", right_index=True, how="left")
        increments = self._increments()
        roles = self._roles()
        for uuid, role in zip(roles["uuid"], roles["role"], strict=True):
            if role == "bad":
                out["bad_parts"] = self._aligned(uuid).to_numpy(dtype=float)
            elif role == "energy_like":
                out[f"energy {uuid}"] = increments[uuid].to_numpy(dtype=float)
        return out

    @memo
    def _compute(self) -> pd.DataFrame:
        changes = self._settings().sort_values("change_ns").reset_index(drop=True)
        metrics = self._metrics()
        columns = [
            "setting",
            "change_ns",
            "before_value",
            "after_value",
            "metric",
            "before_count",
            "after_count",
            "before_median",
            "after_median",
            "shift",
            "cliffs_delta_ratio",
            "p_value",
            "q_value",
            "confounders",
            "trend_before",
            "recipe_parameter",
            "confidence",
        ]
        if changes.empty or metrics.empty:
            return pd.DataFrame(columns=columns)
        starts = metrics["start_ns"].to_numpy(dtype=np.int64)
        ends = metrics["end_ns"].to_numpy(dtype=np.int64)
        variant = metrics["variant"].astype(str).to_numpy()
        variant_change_ns = metrics["start_ns"].to_numpy()[
            np.r_[False, variant[1:] != variant[:-1]]
        ]
        median_cycle = float(metrics["cycle_seconds"].median()) * 1e9
        metric_columns = [
            c
            for c in metrics.columns
            if c == "cycle_seconds"
            or c == "bad_parts"
            or c.startswith(("segment ", "energy "))
        ]
        all_change_ns = changes["change_ns"].to_numpy(dtype=np.int64)
        rows = []
        for ch in changes.itertuples(index=False):
            i = int(np.searchsorted(starts, ch.change_ns, side="left"))
            prev_change = all_change_ns[all_change_ns < ch.change_ns]
            next_change = all_change_ns[all_change_ns > ch.change_ns]
            # Compare cycles of the variant running at the change only: the
            # raw span is wider so that other variants' cycles can be dropped.
            lo = max(0, i - VARIANT_SPAN * self.window_cycles)
            if len(prev_change):
                lo = max(
                    lo, int(np.searchsorted(starts, prev_change.max(), side="left"))
                )
            hi = min(
                len(metrics),
                i + self.settle_cycles + VARIANT_SPAN * self.window_cycles,
            )
            if len(next_change):
                hi = min(
                    hi, int(np.searchsorted(starts, next_change.min(), side="left"))
                )
            current = variant[min(i, len(variant) - 1)]
            # The cycle the change falls in ran with both values: neither side.
            before_end = i - 1 if i > 0 and ch.change_ns < ends[i - 1] else i
            before = metrics.iloc[lo:before_end]
            before = before[before["variant"].astype(str) == current].tail(
                self.window_cycles
            )
            after = metrics.iloc[i + self.settle_cycles : hi]
            after = after[after["variant"].astype(str) == current].head(
                self.window_cycles
            )
            # A variant change at the change: the last cycle before and the
            # first compared cycle after run different variants, or a
            # variant change lies within one median cycle of it.
            first_after = min(i + self.settle_cycles, len(variant) - 1)
            near_variant = bool(
                (i > 0 and variant[i - 1] != variant[first_after])
                or np.any(
                    np.abs(variant_change_ns - ch.change_ns)
                    <= WITH_VARIANT_CYCLES * median_cycle
                )
            )
            confounders = []
            compared = pd.concat([before, after])
            if near_variant:
                confounders.append("variant change")
            near = metrics.iloc[
                max(0, i - self.settle_cycles) : i + 2 * self.settle_cycles
            ]
            if (near["cycle_class"] == "pause").any():
                confounders.append("pause")
            # Another setting changed while the compared cycles ran.
            others = changes[
                (changes["setting"] != ch.setting)
                & (
                    changes["change_ns"]
                    >= (compared["start_ns"].min() if len(compared) else ch.change_ns)
                )
                & (
                    changes["change_ns"]
                    <= (compared["end_ns"].max() if len(compared) else ch.change_ns)
                )
            ]
            if len(others):
                confounders.append("other setting")
            b_ok = before[before["cycle_class"] != "pause"]
            a_ok = after[after["cycle_class"] != "pause"]
            for metric in metric_columns:
                effect = effect_size(b_ok[metric], a_ok[metric])
                trend = False
                values = b_ok[metric].dropna()
                if len(values) >= 8 and values.nunique() > 1:
                    rho = sps.spearmanr(np.arange(len(values)), values.to_numpy())
                    trend = bool(rho.pvalue < FDR)
                rows.append(
                    {
                        "setting": ch.setting,
                        "change_ns": int(ch.change_ns),
                        "before_value": ch.before_value,
                        "after_value": ch.after_value,
                        "metric": metric,
                        "before_count": int(b_ok[metric].notna().sum()),
                        "after_count": int(a_ok[metric].notna().sum()),
                        **effect,
                        "confounders": ";".join(confounders),
                        "trend_before": trend,
                        "recipe_parameter": near_variant,
                    }
                )
        out = pd.DataFrame(rows)
        # A recipe parameter is listed, not tested: its effect cannot be told
        # from the variant's, and it would dilute the other tests.
        out.loc[out["recipe_parameter"].astype(bool), "p_value"] = np.nan
        p = out["p_value"].to_numpy(dtype=float)
        q = np.full(len(p), np.nan)
        ok = np.isfinite(p)
        if ok.any():
            q[ok] = bh_qvalues(p[ok])
        out["q_value"] = q
        enough = (out["before_count"] >= self.min_window_cycles) & (
            out["after_count"] >= self.min_window_cycles
        )
        significant = out["q_value"] < FDR
        clean = (out["confounders"] == "") & ~out["trend_before"]
        variant_conf = (
            out["confounders"].str.contains("variant") | out["recipe_parameter"]
        )
        out["confidence"] = np.where(
            significant & enough & clean,
            "high",
            np.where(significant & enough & ~variant_conf, "medium", "low"),
        )
        return out[columns]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setting_changes(self) -> pd.DataFrame:
        """Every change of a held setting.

        Returns:
            Point events: ``systime``, ``uuid``, ``source_uuid`` (the
            setting), ``before_value``, ``after_value`` and
            ``recipe_parameter`` (the change came with a variant change).
        """
        changes = self._settings()
        impact = self._compute()
        columns = [
            "systime",
            "uuid",
            "source_uuid",
            "before_value",
            "after_value",
            "recipe_parameter",
        ]
        if changes.empty:
            return self._empty(columns)
        recipe = (
            impact.groupby("change_ns")["recipe_parameter"].first()
            if len(impact)
            else pd.Series(dtype=bool)
        )
        out = pd.DataFrame(
            {
                "systime": self._times(changes["change_ns"].to_numpy()),
                "uuid": self.event_uuid,
                "source_uuid": changes["setting"].to_numpy(),
                "before_value": changes["before_value"].to_numpy(),
                "after_value": changes["after_value"].to_numpy(),
                "recipe_parameter": changes["change_ns"]
                .map(recipe)
                .fillna(False)
                .to_numpy(dtype=bool),
            }
        )
        return out[columns]

    def setting_impact(self) -> pd.DataFrame:
        """Before/after effect of every change on every metric.

        Returns:
            One row per change and metric: ``setting``, ``change_ns``,
            ``before_value``, ``after_value``, ``metric``
            (``cycle_seconds``, ``segment <from → to>``, ``bad_parts``,
            ``energy <meter>``), ``before_count``, ``after_count``,
            ``before_median``, ``after_median``, ``shift`` (Hodges-Lehmann,
            after - before), ``cliffs_delta_ratio``, ``p_value``,
            ``q_value``, ``confounders``, ``trend_before``,
            ``recipe_parameter`` and ``confidence``.
        """
        return self._compute().copy()

    def setting_register(self) -> pd.DataFrame:
        """What each setting does to the cycle time, over all its holds.

        Returns:
            One row per setting: ``setting``, ``change_count``,
            ``values_seen``, ``sensitivity_seconds_per_unit`` (Theil-Sen
            slope of the hold's median cycle time on the setting value),
            ``best_value_observed`` (the value with the shortest median
            cycle) and ``potential_seconds`` (median cycle of the current
            value minus that of the best).
        """
        changes = self._settings()
        metrics = self._metrics()
        columns = [
            "setting",
            "change_count",
            "values_seen",
            "sensitivity_seconds_per_unit",
            "best_value_observed",
            "potential_seconds",
        ]
        if changes.empty or metrics.empty:
            return pd.DataFrame(columns=columns)
        running = metrics[metrics["cycle_class"] != "pause"]
        starts = running["start_ns"].to_numpy(dtype=np.int64)
        rows = []
        for setting, part in changes.groupby("setting"):
            part = part.sort_values("change_ns")
            bounds = np.r_[starts.min(), part["change_ns"].to_numpy(), starts.max() + 1]
            values = np.r_[part["before_value"].iloc[0], part["after_value"].to_numpy()]
            medians = []
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
                sel = running[(starts >= lo) & (starts < hi)]["cycle_seconds"]
                medians.append(float(sel.median()) if len(sel) else np.nan)
            med = np.array(medians)
            ok = np.isfinite(med)
            slope = (
                float(sps.theilslopes(med[ok], values[ok]).slope)
                if ok.sum() >= 2 and len(np.unique(values[ok])) >= 2
                else np.nan
            )
            best = int(np.nanargmin(med)) if ok.any() else 0
            rows.append(
                {
                    "setting": setting,
                    "change_count": int(len(part)),
                    "values_seen": int(len(np.unique(values))),
                    "sensitivity_seconds_per_unit": slope,
                    "best_value_observed": float(values[best]),
                    "potential_seconds": (
                        float(med[-1] - med[best]) if ok.any() else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def evaluation_window(self) -> pd.DataFrame:
        """Whether the setting tests can be trusted, and why not.

        Returns:
            One row: ``scope`` (``all``), ``unit_count`` (setting changes),
            ``evaluable``, ``reason`` (``no_candidate_signal: held setting``
            without one), ``time_resolution_seconds``, ``setting_count`` and
            ``high_confidence_count`` (tests with ``high`` confidence).
        """
        changes = self._settings()
        impact = self._compute()
        return self._evaluation_row(
            len(changes),
            1,
            missing="held setting" if changes.empty else None,
            setting_count=int(changes["setting"].nunique()) if len(changes) else 0,
            high_confidence_count=(
                int((impact["confidence"] == "high").sum()) if len(impact) else 0
            ),
        )

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-able).

        Returns:
            ``evaluable``, ``reason``, ``change_count``, ``setting_count`` and
            ``best_cycle_time_change`` (the high-confidence change with the
            largest cycle-time reduction, as ``"<setting>: a -> b"``).
        """
        window = self.evaluation_window().iloc[0]
        impact = self._compute()
        best = None
        if len(impact):
            cand = impact[
                (impact["metric"] == "cycle_seconds") & (impact["confidence"] == "high")
            ].sort_values("shift")
            if len(cand) and cand["shift"].iloc[0] < 0:
                r = cand.iloc[0]
                best = f"{r['setting']}: {r['before_value']:g} -> {r['after_value']:g}"
        return {
            **self._summary_head("change_count"),
            "setting_count": int(window["setting_count"]),
            "best_cycle_time_change": best,
        }
