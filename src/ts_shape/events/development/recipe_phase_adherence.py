"""Recipe-phase adherence: check each batch phase against a spec.

Batch processes execute a sequence of named phases (heat-up, hold,
cool-down, ...). Each phase has a development-defined spec: an expected
duration window, a hold-value window, a maximum ramp rate, sometimes a
peak ceiling. This detector iterates the phase intervals in a batch and
emits one event per phase carrying pass/fail plus which criterion failed.

The spec format is intentionally a plain dict so it can be authored by a
process engineer (or loaded from YAML/JSON) without inventing a new
schema class -- see :meth:`evaluate` for the recognised keys.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class RecipePhaseAdherenceEvents(Base):
    """Evaluate batch recipe phases against a declarative spec.

    The spec is a mapping ``{phase_name: criteria_dict}``. Recognised
    criteria keys (all optional) per phase:

    * ``duration_s``        -- ``(min, max)`` seconds.
    * ``hold_value``        -- ``(min, max)`` for the mean value during
      the phase.
    * ``ramp_rate_max``     -- maximum absolute slope (value-units per
      second) computed from the first to the last sample.
    * ``peak_value``        -- ``(min, max)`` for the phase max.
    * ``trough_value``      -- ``(min, max)`` for the phase min.

    Missing criteria are not checked. A phase whose name is absent from
    the spec is reported with ``pass=True`` and an empty
    ``criteria_failed`` list (the phase was observed but not constrained).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        phase_uuid: str,
        value_uuid: str,
        spec: dict[str, dict[str, Any]],
        *,
        event_uuid: str = "dev:recipe_phase_adherence",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.phase_uuid = phase_uuid
        self.value_uuid = value_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.spec = dict(spec)
        self._validate_uuid(self.dataframe, phase_uuid)
        self._validate_uuid(self.dataframe, value_uuid)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _phase_intervals(self) -> pd.DataFrame:
        """Return contiguous phase intervals from the phase signal."""
        sel = self.dataframe[self.dataframe["uuid"] == self.phase_uuid].copy()
        if sel.empty:
            return pd.DataFrame(columns=["start", "end", "phase"])
        # Phase name lives in value_string if present, else value_column.
        phase_col = (
            "value_string" if "value_string" in sel.columns else self.value_column
        )
        sel = (
            sel[[self.time_column, phase_col]]
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )
        sel = sel.dropna(subset=[phase_col])
        if sel.empty:
            return pd.DataFrame(columns=["start", "end", "phase"])
        sel["_chg"] = (sel[phase_col] != sel[phase_col].shift()).cumsum()
        out = (
            sel.groupby("_chg")
            .agg(
                start=(self.time_column, "first"),
                end=(self.time_column, "last"),
                phase=(phase_col, "first"),
            )
            .reset_index(drop=True)
        )
        return out

    @staticmethod
    def _check_range(value: float, bounds: Any) -> bool:
        try:
            lo, hi = bounds
        except (TypeError, ValueError):
            return True  # malformed spec entry -> skip silently
        return (value >= lo) and (value <= hi)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> pd.DataFrame:
        """Evaluate every observed phase interval against the spec.

        Returns:
            Interval-shape DataFrame, one row per phase occurrence,
            columns: ``start``, ``end``, ``duration_seconds``, ``uuid``,
            ``phase``, ``mean_value``, ``min_value``, ``max_value``,
            ``observed_ramp_per_s``, ``criteria_failed`` (list[str]),
            ``pass``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "phase",
            "mean_value",
            "min_value",
            "max_value",
            "observed_ramp_per_s",
            "criteria_failed",
            "pass",
        ]
        intervals = self._phase_intervals()
        if intervals.empty:
            return pd.DataFrame(columns=cols)

        # Pre-extract the value series sorted by time.
        vals = (
            self.dataframe[self.dataframe["uuid"] == self.value_uuid][
                [self.time_column, self.value_column]
            ]
            .copy()
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )
        if vals.empty:
            return pd.DataFrame(columns=cols)
        vals[self.time_column] = pd.to_datetime(vals[self.time_column])

        rows: list[dict[str, Any]] = []
        for _, ivl in intervals.iterrows():
            mask = (vals[self.time_column] >= ivl["start"]) & (
                vals[self.time_column] <= ivl["end"]
            )
            window = vals.loc[mask]
            if window.empty:
                continue
            v = window[self.value_column].to_numpy(dtype=float)
            duration_s = (ivl["end"] - ivl["start"]).total_seconds()
            mean_v = float(np.mean(v))
            min_v = float(np.min(v))
            max_v = float(np.max(v))
            # End-to-end ramp rate; for spec checking we use absolute value.
            if duration_s > 0 and len(v) >= 2:
                ramp = float((v[-1] - v[0]) / duration_s)
            else:
                ramp = 0.0

            failed: list[str] = []
            phase_spec = self.spec.get(ivl["phase"], {})

            if "duration_s" in phase_spec and not self._check_range(
                duration_s, phase_spec["duration_s"]
            ):
                failed.append("duration_s")
            if "hold_value" in phase_spec and not self._check_range(
                mean_v, phase_spec["hold_value"]
            ):
                failed.append("hold_value")
            if "peak_value" in phase_spec and not self._check_range(
                max_v, phase_spec["peak_value"]
            ):
                failed.append("peak_value")
            if "trough_value" in phase_spec and not self._check_range(
                min_v, phase_spec["trough_value"]
            ):
                failed.append("trough_value")
            if "ramp_rate_max" in phase_spec:
                limit = float(phase_spec["ramp_rate_max"])
                if abs(ramp) > limit:
                    failed.append("ramp_rate_max")

            rows.append(
                {
                    "start": ivl["start"],
                    "end": ivl["end"],
                    "duration_seconds": duration_s,
                    "uuid": self.event_uuid,
                    "phase": ivl["phase"],
                    "mean_value": mean_v,
                    "min_value": min_v,
                    "max_value": max_v,
                    "observed_ramp_per_s": ramp,
                    "criteria_failed": failed,
                    "pass": len(failed) == 0,
                }
            )

        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
