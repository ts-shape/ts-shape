"""Minimal hit-count backtest for lambda rules.

The first-slice MVP supports "run rule, count hits" only — no
label-matching, no precision/recall. The returned
:class:`BacktestResult` still carries the full :class:`EventLog` so the
caller can inspect any individual hit and run their own downstream
analysis (e.g. ``to_event_log_xes`` for pm4py, ``filter_by_pack`` for
domain-specific drill-down).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import pandas as pd

from ..model import EventLog
from ..schema import OCEL_TYPE, OCEL_OID, TS_SEVERITY
from .detector import LambdaDetector


@dataclass
class BacktestResult:
    detector: str
    hit_count: int
    by_severity: dict[str, int]
    by_asset: dict[str, int]
    event_log: EventLog

    def __repr__(self) -> str:
        return (
            f"BacktestResult(detector={self.detector!r}, "
            f"hits={self.hit_count}, "
            f"severity={self.by_severity}, "
            f"assets={len(self.by_asset)})"
        )


def run_backtest(
    detector: LambdaDetector,
    df: pd.DataFrame,
    *,
    objects: Mapping[str, object] | None = None,
    qualifiers: Mapping[str, str] | None = None,
) -> BacktestResult:
    """Run ``detector`` over ``df`` and summarize hit counts."""
    log = detector.to_event_log(df, objects=objects, qualifiers=qualifiers)

    by_severity: dict[str, int] = {}
    if TS_SEVERITY in log.events.columns and not log.events.empty:
        by_severity = {
            (str(k) if pd.notna(k) else "unset"): int(v)
            for k, v in log.events[TS_SEVERITY].value_counts(dropna=False).items()
        }

    by_asset: dict[str, int] = {}
    if not log.relations.empty:
        asset_rel = log.relations[log.relations[OCEL_TYPE] == "asset"]
        by_asset = {
            str(k): int(v) for k, v in asset_rel[OCEL_OID].value_counts().items()
        }

    return BacktestResult(
        detector=detector.detector_name,
        hit_count=len(log.events),
        by_severity=by_severity,
        by_asset=by_asset,
        event_log=log,
    )
