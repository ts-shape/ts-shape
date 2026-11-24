"""Run the cycle profile and the mapping-free unit profilers in one go.

:func:`profile_units` prepares the signals of one PLC once
(:func:`ts_shape.events.profiling.runner.prepare_signals`), computes the cycle profile
once, and runs every requested unit profiler on top of it -- stops,
changeovers, start-ups, orders, scrap, handshakes, curve signatures,
setting changes and control loops. Each profiler's tables and summary land
in one :class:`~ts_shape.events.profiling.result.ProfileResult`; the
:class:`~ts_shape.events.profiling.result.UnitProfilesResult` collects them and says,
per profiler, whether it is evaluable and why not.

The profilers are not listed here: every class that declares
``PROFILE_KIND`` (see :class:`~ts_shape.events.profiling._base.
_ProfilerBase`) is found, with the tables it names in ``PROFILE_TABLES``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ts_shape.events.profiling._base import _ProfilerBase
from ts_shape.events.profiling.cycles import CycleProfileEvents
from ts_shape.events.profiling.result import ProfileResult, UnitProfilesResult
from ts_shape.events.profiling.runner import (
    DEFAULT_EXCLUDE,
    cycle_tables,
    prepare_signals,
)

__all__ = [
    "DEFAULT_PROFILERS",
    "PROFILERS",
    "UnitProfilesResult",
    "profile_units",
    "run_profiler",
]

logger = logging.getLogger(__name__)

#: Modules whose import registers the profiler classes (every class that
#: declares ``PROFILE_KIND`` registers itself in ``_base.PROFILER_CLASSES``).
_PROFILER_MODULES = tuple(
    f"ts_shape.events.profiling.{name}"
    for name in (
        "changeovers",
        "control_loops",
        "curves",
        "handshakes",
        "orders",
        "settings",
        "stops",
    )
)


def _profiler_classes() -> dict[str, type[_ProfilerBase]]:
    """Profiler kind -> class, in run order (``PROFILE_ORDER``)."""
    for module in _PROFILER_MODULES:
        importlib.import_module(module)
    from ts_shape.events.profiling._base import PROFILER_CLASSES

    return dict(
        sorted(PROFILER_CLASSES.items(), key=lambda kv: (kv[1].PROFILE_ORDER, kv[0]))
    )


#: Profiler name -> class, in dependency order (filled from the classes).
PROFILERS: dict[str, type[_ProfilerBase]] = _profiler_classes()
#: The profilers :func:`profile_units` runs by default (all of them).
DEFAULT_PROFILERS: tuple[str, ...] = tuple(PROFILERS)


def run_profiler(
    cls: type[_ProfilerBase],
    frame: pd.DataFrame,
    profile: CycleProfileEvents | None = None,
    **options: Any,
) -> ProfileResult:
    """Run one profiler class and collect its declared tables and summary.

    Args:
        cls: A profiler class (it declares ``PROFILE_KIND`` and
            ``PROFILE_TABLES``).
        frame: Prepared long-format signals.
        profile: The cycle profile to reuse (passed when the class needs
            cycles).
        **options: Options; those the class names in ``PROFILE_OPTIONS`` are
            passed to its constructor, the rest are ignored.

    Returns:
        :class:`~ts_shape.events.profiling.result.ProfileResult` of ``kind``
        ``PROFILE_KIND``.
    """
    kwargs = {k: v for k, v in options.items() if k in cls.PROFILE_OPTIONS}
    if cls.NEEDS_CYCLES and profile is not None:
        kwargs["cycle_profile"] = profile
    det: Any = cls(frame, **kwargs)
    return ProfileResult(
        tables={name: getattr(det, name)() for name in cls.profile_tables()},
        summary=det.summary(),
        kind=str(cls.PROFILE_KIND),
    )


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
            :func:`~ts_shape.events.profiling.runner.prepare_signals`).
        top_n: Influencers kept per comparison (cycle profile).
        **cycle_options: Passed to :class:`~ts_shape.CycleProfileEvents`.

    Returns:
        :class:`~ts_shape.events.profiling.result.UnitProfilesResult` with
        ``results["cycles"]`` (the :func:`~ts_shape.events.profiling.runner.profile_cycles`
        tables) and one result per profiler; ``summary`` holds each
        profiler's headline numbers.

    Raises:
        ValueError: For an unknown profiler name.

    Example:
        >>> import ts_shape
        >>> from ts_shape.events.profiling.units import profile_units
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
            result = run_profiler(PROFILERS[name], frame, profile, plan=plan)
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
