"""Run the cycle profile and the mapping-free unit profilers in one go.

:func:`profile_units` prepares the signals of one PLC once
(:func:`ts_shape.cycle_profile.prepare_signals`), computes the cycle profile
once, and runs every requested unit profiler on top of it -- stops,
changeovers, start-ups, orders, scrap, handshakes, curve signatures,
setting changes and control loops. Each profiler's tables and summary land
in one :class:`~ts_shape._profile_result.ProfileResult`; the
:class:`~ts_shape._profile_result.UnitProfilesResult` collects them and says,
per profiler, whether it is evaluable and why not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from ts_shape._profile_result import ProfileResult, UnitProfilesResult
from ts_shape.cycle_profile import DEFAULT_EXCLUDE, cycle_tables, prepare_signals
from ts_shape.events.production.cycle_profiling import CycleProfileEvents

__all__ = ["DEFAULT_PROFILERS", "PROFILERS", "profile_units"]

logger = logging.getLogger(__name__)


def _stops(frame: pd.DataFrame, profile: CycleProfileEvents, **_: Any) -> ProfileResult:
    from ts_shape.events.production.stop_profiling import StopProfileEvents

    det = StopProfileEvents(frame, cycle_profile=profile)
    return ProfileResult(
        tables={
            "stop_records": det.stop_records(),
            "stop_signatures": det.stop_signatures(),
            "signature_evidence": det.signature_evidence(),
            "evaluation_window": det.evaluation_window(),
        },
        summary=det.summary(),
        kind="stops",
    )


def _changeovers(
    frame: pd.DataFrame, profile: CycleProfileEvents, **_: Any
) -> ProfileResult:
    from ts_shape.events.production.changeover_profiling import (
        ChangeoverProfileEvents,
    )

    det = ChangeoverProfileEvents(frame, cycle_profile=profile)
    return ProfileResult(
        tables={
            "changeover_records": det.changeover_records(),
            "changeover_view": det.changeover_view(),
            "changeover_register": det.changeover_register(),
            "changeover_matrix": det.changeover_matrix(),
            "evaluation_window": det.evaluation_window(),
        },
        summary=det.summary(),
        kind="changeovers",
    )


def _startups(
    frame: pd.DataFrame, profile: CycleProfileEvents, **_: Any
) -> ProfileResult:
    from ts_shape.events.production.changeover_profiling import StartupProfileEvents

    det = StartupProfileEvents(frame, cycle_profile=profile)
    return ProfileResult(
        tables={
            "startup_records": det.startup_records(),
            "startup_signals": det.startup_signals(),
            "startup_register": det.startup_register(),
            "evaluation_window": det.evaluation_window(),
        },
        summary=det.summary(),
        kind="startups",
    )


#: Profiler name -> runner ``(frame, cycle profile, **options) -> ProfileResult``,
#: in dependency order.
PROFILERS: dict[str, Callable[..., ProfileResult]] = {
    "stops": _stops,
    "changeovers": _changeovers,
    "startups": _startups,
}
#: The profilers :func:`profile_units` runs by default (all of them).
DEFAULT_PROFILERS: tuple[str, ...] = tuple(PROFILERS)


def profile_units(
    timeseries: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    *,
    profilers: Sequence[str] = DEFAULT_PROFILERS,
    plan: pd.DataFrame | None = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    top_n: int = 10,
    **cycle_options: Any,
) -> UnitProfilesResult:
    """Profile one PLC: the cycle profile plus the requested unit profilers.

    A profiler that fails is reported (``evaluable`` False, ``reason``
    ``error: ...``) instead of stopping the others.

    Args:
        timeseries: Raw frame of a ts-shape timeseries loader (one PLC).
        metadata: Optional metadata table; labels only.
        profilers: Names from :data:`PROFILERS` to run besides the cycle
            profile (default: all).
        plan: Optional production plan for the order profiler
            (``order_id``, ``planned_quantity``).
        exclude: Signals to drop (personal data; see
            :func:`~ts_shape.cycle_profile.prepare_signals`).
        top_n: Influencers kept per comparison (cycle profile).
        **cycle_options: Passed to :class:`~ts_shape.CycleProfileEvents`.

    Returns:
        :class:`~ts_shape._profile_result.UnitProfilesResult` with
        ``results["cycles"]`` (the :func:`~ts_shape.cycle_profile.profile_cycles`
        tables) and one result per profiler; ``summary`` holds each
        profiler's headline numbers.

    Raises:
        ValueError: For an unknown profiler name.

    Example:
        >>> import ts_shape
        >>> from ts_shape.unit_profiles import profile_units
        >>> frame, _ = ts_shape.make_plc_cycles(
        ...     n_cycles=300, seed=3, stop_causes=4, stop_probability=0.08)
        >>> result = profile_units(frame, profilers=["stops"])
        >>> sorted(result.results)
        ['cycles', 'stops']
    """
    unknown = [p for p in profilers if p not in PROFILERS]
    if unknown:
        raise ValueError(
            f"profile_units: unknown profiler(s) {unknown}; known: {list(PROFILERS)}"
        )
    frame = prepare_signals(timeseries, metadata, exclude=exclude)
    profile = CycleProfileEvents(frame, **cycle_options)
    cycles = cycle_tables(frame, profile, top_n=top_n)
    results: dict[str, ProfileResult] = {"cycles": cycles}
    summary: dict[str, Any] = {
        "cycles": {
            "evaluable": bool(cycles.summary.get("evaluable", False)),
            "reason": cycles.summary.get("reason"),
        }
    }
    for name in PROFILERS:
        if name not in profilers:
            continue
        try:
            result = PROFILERS[name](frame, profile, plan=plan, results=results)
        except Exception as exc:  # noqa: BLE001 - one profiler must not stop the rest
            logger.warning("profile_units: %s failed: %s", name, exc)
            result = ProfileResult(
                tables={},
                summary={"evaluable": False, "reason": f"error: {exc}"},
                kind=name,
            )
        results[name] = result
        summary[name] = result.summary
    return UnitProfilesResult(results=results, summary=summary)
