"""Development Events

Detectors for product and process development (R&D) workflows: design of
experiments, design-space qualification, golden-batch comparison, recipe
phase adherence, and outcome-driven critical parameter ranking.

These detectors target activities that happen before (or alongside)
commercial production -- the work of process development engineers,
formulation scientists, and validation teams -- as opposed to the
operations-focused detectors in the other event packs.

Classes:
- DesignOfExperimentsEvents: Segment continuous signals into DOE runs and
  estimate main effects per factor.
  - detect_runs: Identify stable factor-setting intervals as DOE runs.
  - compute_effects: Aggregate run response and main effects per factor.

- DesignSpaceEvents: Establish a multivariate qualified operating window
  from R&D data and monitor commercial operation against it.
  - fit_box: Per-axis bounds (quantile or min/max).
  - fit_hull: scipy convex hull of the qualified region.
  - detect_excursions: Intervals where operation leaves the design space.
  - boundary_proximity: Per-sample distance to the design-space boundary.

- GoldenBatchDeviationEvents: Quantify deviation of new batch trajectories
  against a reference (golden) batch.
  - compare: Pointwise, area-between-curves, or DTW deviation.
  - phase_breakdown: Deviation broken down by recipe phase.

- RecipePhaseAdherenceEvents: Evaluate batch recipe phases against a
  declarative spec (duration, hold value, ramp rate, peak).
  - evaluate: One event per phase with pass/fail and failed criteria.

- CriticalParameterRankingEvents: Rank candidate input parameters by their
  statistical association with a quality outcome.
  - rank: Pearson / Spearman correlation or one-way ANOVA F-statistic.
  - top_drivers: Filter and sort the ranking to the top-k significant.
"""

from .critical_parameter_ranking import CriticalParameterRankingEvents
from .design_of_experiments import DesignOfExperimentsEvents
from .design_space import DesignSpaceEvents
from .golden_batch import GoldenBatchDeviationEvents
from .recipe_phase_adherence import RecipePhaseAdherenceEvents

__all__ = [
    "CriticalParameterRankingEvents",
    "DesignOfExperimentsEvents",
    "DesignSpaceEvents",
    "GoldenBatchDeviationEvents",
    "RecipePhaseAdherenceEvents",
]
