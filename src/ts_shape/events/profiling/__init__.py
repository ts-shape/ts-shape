"""Profiling: mapping-free profiles of a machine from its raw PLC signals

Give the profilers every raw signal of a machine controller, without a
signal mapping: :class:`CycleProfileEvents` finds the machine cycle and
classifies each one, and the unit profilers build on it (stops, changeovers,
start-ups, handshakes, orders, scrap, curves, settings). The control-loop
profiler needs no cycle. :func:`profile_units` runs them all and
:func:`profile_cycles` (also ``python -m ts_shape.events.profiling``) loads a
PLC's data and profiles its cycles in one call.

Profilers:
- CycleProfileEvents: Mapping-free cycle time profiling of one PLC.
  - cycle_anchor_candidates: Find the cycle event without a mapping.
  - evaluation_window: Enough coherent cycles per variant?
  - standard_view: Learned standard sequence of a cycle.
  - segment_durations: Cycles split into segments (sum = cycle time).
  - contribution_register: Potential, spread and variance share per segment.
  - cycle_time_statistics: Robust statistics and class counts per variant.
  - detect_cycle_influencers: What separates long/short from normal cycles.
  - cycle_records: One record per cycle at one timestamp.
  - detect_unusual_cycles: Long and short cycles.
  - signal_events: The events every signal was turned into.
  - units: The classified cycles (for unit profilers).
  - events_frame: Every discrete event (for unit profilers).
  - signal_specs: Inferred spec per signal.
  - variant_limits: Classification limits per variant.
  - cycle_anchor: The anchor event and its signal.
  - variant_signal: The variant signal.
  - resolution_seconds: Time resolution of the cycle view.
  - evaluation_reason: Reason of the all-variants row.

- SettingImpactEvents: Before/after effect of changes of held settings.
  - setting_changes: Every change of a held setting.
  - setting_impact: Before/after tests per change and metric.
  - setting_register: Sensitivity and best value per setting.
  - evaluation_window: Whether the tests can be trusted.
  - summary: Headline numbers.

- CurveSignatureEvents: Golden curves and named fault patterns of analog signals per cycle.
  - curve_golden: Golden curve band per variant and signal.
  - curve_cycle_scores: Deviation score, phase and pattern per cycle.
  - curve_patterns: Clustered fault patterns with lost time.
  - evaluation_window: Evaluability per variant and signal.
  - summary: Headline numbers.

- HandshakeProfileEvents: Mapping-free request/answer pairs and the waits on partners.
  - handshake_pairs: Discovered request/answer pairs, two- or four-phase.
  - handshake_records: One wait interval per request.
  - handshake_register: Wait per pair with potential and cycle-time share.
  - evaluation_window: Evaluability per pair.
  - summary: Headline numbers.

- ScrapProfileEvents: Mapping-free scrap Pareto without reason codes.
  - nok_records: One record per cycle with a bad part.
  - scrap_pareto: Contexts where bad parts are over-represented.
  - counter_roles: Total, good and bad counters learned from the data.
  - evaluation_window: Enough bad parts to trust the profile?
  - summary: Headline numbers.

- OrderProfileEvents: Mapping-free order profiling: speed, scrap, attainment.
  - order_records: One record per production order.
  - order_register: Lost time of orders per variant.
  - counter_roles: Total, good and bad counters learned from the data.
  - evaluation_window: Enough orders to trust the profile?
  - summary: Headline numbers.

- StartupProfileEvents: Mapping-free start-up profiling after stops and changeovers.
  - startup_records: One record per start-up.
  - startup_signals: Which signals are last back in their running band.
  - startup_register: Start-up time and potential per trigger.
  - evaluation_window: Enough start-ups to trust the profile?
  - summary: Headline numbers.

- ChangeoverProfileEvents: Mapping-free changeover (SMED) profiling.
  - changeover_records: One record per changeover.
  - changeover_view: The learned standard step sequence.
  - changeover_register: Potential per changeover segment.
  - changeover_matrix: Changeover times per from -> to variant.
  - evaluation_window: Enough changeovers to trust the profile?
  - summary: Headline numbers.

- StopProfileEvents: Mapping-free stop profiling: causes and lost time.
  - stop_records: One record per stop with its signature.
  - stop_signatures: Lost time, MTTR and potential per stop signature.
  - signature_evidence: Why an event counts as a stop signature.
  - evaluation_window: Enough stops to trust the profile?
  - summary: Headline numbers.

- ControlLoopProfileEvents: Mapping-free control loops: find setpoint/PV/output, rank loop health.
  - loop_pairs: Setpoint, process value and output of every loop found.
  - loop_steps: Response metrics of every setpoint step.
  - loop_register: Health, diagnosis and rank per loop.
  - evaluation_window: Enough steps to trust each loop?
  - summary: Headline numbers.

Runners:
- profile_cycles: Load and profile one PLC's cycles; returns a CycleProfileResult.
- profile_units: Run every unit profiler on one machine; returns a UnitProfilesResult.
- run_profiler: Run one profiler by its kind ("stops", "orders", ...).

Results and registry:
- CycleProfileResult: Every table of profile_cycles plus a summary of plain numbers.
- ProfileResult: The tables and summary of one profiler.
- UnitProfilesResult: The ProfileResult of every profiler run by profile_units.
- PROFILERS: Profiler kind -> class, in run order.
- DEFAULT_PROFILERS: The kinds profile_units runs by default (all of them).
"""

from .changeovers import ChangeoverProfileEvents, StartupProfileEvents
from .control_loops import ControlLoopProfileEvents
from .curves import CurveSignatureEvents
from .cycles import CycleProfileEvents
from .handshakes import HandshakeProfileEvents
from .orders import OrderProfileEvents, ScrapProfileEvents
from .result import ProfileResult, UnitProfilesResult
from .runner import CycleProfileResult, profile_cycles
from .settings import SettingImpactEvents
from .stops import StopProfileEvents
from .units import DEFAULT_PROFILERS, PROFILERS, profile_units, run_profiler

__all__ = [
    "ChangeoverProfileEvents",
    "ControlLoopProfileEvents",
    "CurveSignatureEvents",
    "CycleProfileEvents",
    "CycleProfileResult",
    "DEFAULT_PROFILERS",
    "HandshakeProfileEvents",
    "OrderProfileEvents",
    "PROFILERS",
    "ProfileResult",
    "ScrapProfileEvents",
    "SettingImpactEvents",
    "StartupProfileEvents",
    "StopProfileEvents",
    "UnitProfilesResult",
    "profile_cycles",
    "profile_units",
    "run_profiler",
]
