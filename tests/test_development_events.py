"""Tests for the events.development pack.

Five detectors target product- and process-development workflows:
``DesignOfExperimentsEvents``, ``DesignSpaceEvents``,
``GoldenBatchDeviationEvents``, ``RecipePhaseAdherenceEvents``,
``CriticalParameterRankingEvents``.
"""

from __future__ import annotations

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest

from ts_shape.events.development import (
    CriticalParameterRankingEvents,
    DesignOfExperimentsEvents,
    DesignSpaceEvents,
    GoldenBatchDeviationEvents,
    RecipePhaseAdherenceEvents,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_long_df(times, values, uuid):
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": [uuid] * len(times),
            "value_double": values,
            "is_delta": [True] * len(times),
        }
    )


def _make_multi_uuid_df(rows: dict[str, tuple[pd.DatetimeIndex, np.ndarray]]):
    parts = []
    for uid, (t, v) in rows.items():
        parts.append(_make_long_df(t, v, uid))
    return pd.concat(parts, ignore_index=True)


# ===========================================================================
# DesignOfExperimentsEvents
# ===========================================================================


class TestDesignOfExperimentsEvents:

    def _build_doe_dataset(self):
        """Two factors with three steps each, plus a response signal.

        Run layout (3 distinct levels per factor, 4 samples per run):

            run 0:  F1=10, F2=1
            run 1:  F1=10, F2=2
            run 2:  F1=20, F2=2
            run 3:  F1=20, F2=3
        """
        per_run = 4
        n_runs = 4
        t = pd.date_range("2024-01-01", periods=per_run * n_runs, freq="1min")
        f1 = np.concatenate([
            np.full(per_run, 10.0), np.full(per_run, 10.0),
            np.full(per_run, 20.0), np.full(per_run, 20.0),
        ])
        f2 = np.concatenate([
            np.full(per_run, 1.0), np.full(per_run, 2.0),
            np.full(per_run, 2.0), np.full(per_run, 3.0),
        ])
        # Response = 2*F1 + 5*F2, so F2 has the larger main effect.
        response = 2.0 * f1 + 5.0 * f2

        df = _make_multi_uuid_df(
            {
                "factor:F1": (t, f1),
                "factor:F2": (t, f2),
                "response:Y": (t, response),
            }
        )
        return df

    def test_detect_runs_recovers_segment_count(self):
        df = self._build_doe_dataset()
        det = DesignOfExperimentsEvents(df, factor_uuids=["factor:F1", "factor:F2"])
        runs = det.detect_runs(min_duration="2min", stability_tol=0.01)
        assert len(runs) == 4
        assert set(runs.columns) >= {
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "run_id",
            "factor__factor:F1_level",
            "factor__factor:F2_level",
        }
        # Each detected run carries the right factor level.
        f1_levels = runs["factor__factor:F1_level"].tolist()
        f2_levels = runs["factor__factor:F2_level"].tolist()
        assert f1_levels == [10.0, 10.0, 20.0, 20.0]
        assert f2_levels == [1.0, 2.0, 2.0, 3.0]

    def test_compute_effects_ranks_factor_impact(self):
        df = self._build_doe_dataset()
        det = DesignOfExperimentsEvents(df, factor_uuids=["factor:F1", "factor:F2"])
        effects = det.compute_effects(
            response_uuid="response:Y",
            statistic="mean",
            min_duration="2min",
            stability_tol=0.01,
        )
        assert not effects.empty
        # Factor F2 swings the response by 5 per level, F1 swings 2 per level → F2
        # should have larger |main_effect| magnitude at its extremes.
        f1_eff = effects[effects["factor"] == "factor:F1"]["main_effect"].abs().max()
        f2_eff = effects[effects["factor"] == "factor:F2"]["main_effect"].abs().max()
        assert f2_eff > f1_eff

    def test_empty_factors_raises(self):
        with pytest.raises(ValueError):
            DesignOfExperimentsEvents(pd.DataFrame(), factor_uuids=[])


# ===========================================================================
# DesignSpaceEvents
# ===========================================================================


class TestDesignSpaceEvents:

    def _qualification_df(self):
        """100 samples inside an axis-aligned box: temp in [60, 80], pH in [6.8, 7.2]."""
        np.random.seed(0)
        n = 100
        t = pd.date_range("2024-01-01", periods=n, freq="1min")
        temp = np.random.uniform(60.0, 80.0, n)
        ph = np.random.uniform(6.8, 7.2, n)
        return _make_multi_uuid_df({"cpp:temp": (t, temp), "cpp:ph": (t, ph)})

    def test_fit_box_then_detect_excursions(self):
        qual = self._qualification_df()
        det = DesignSpaceEvents(qual, cpp_uuids=["cpp:temp", "cpp:ph"])
        det.fit_box(quantiles=(0.0, 1.0))

        # Operation: 30 samples inside, 10 samples outside (temp = 90).
        t_op = pd.date_range("2024-02-01", periods=40, freq="1min")
        temp = np.concatenate([np.full(30, 70.0), np.full(10, 90.0)])
        ph = np.full(40, 7.0)
        op = _make_multi_uuid_df({"cpp:temp": (t_op, temp), "cpp:ph": (t_op, ph)})

        excursions = det.detect_excursions(op)
        assert len(excursions) == 1
        assert excursions["excursion_mode"].iloc[0] == "box"
        assert excursions["duration_seconds"].iloc[0] > 0

    def test_boundary_proximity_flags_near_boundary(self):
        qual = self._qualification_df()
        det = DesignSpaceEvents(qual, cpp_uuids=["cpp:temp", "cpp:ph"])
        det.fit_box(quantiles=(0.0, 1.0))

        # 5 samples comfortably inside, 5 samples almost at the upper temp bound.
        t_op = pd.date_range("2024-02-01", periods=10, freq="1min")
        temp = np.concatenate([np.full(5, 70.0), np.full(5, 79.5)])
        ph = np.full(10, 7.0)
        op = _make_multi_uuid_df({"cpp:temp": (t_op, temp), "cpp:ph": (t_op, ph)})

        near = det.boundary_proximity(op, warn_margin=0.1)
        assert not near.empty
        assert (near["closest_axis"] == "cpp:temp").all()

    def test_fit_hull_then_detect_excursions(self):
        qual = self._qualification_df()
        det = DesignSpaceEvents(qual, cpp_uuids=["cpp:temp", "cpp:ph"])
        det.fit_hull()

        # Sample far outside the hull on both axes.
        t_op = pd.date_range("2024-02-01", periods=5, freq="1min")
        temp = np.full(5, 200.0)
        ph = np.full(5, 4.0)
        op = _make_multi_uuid_df({"cpp:temp": (t_op, temp), "cpp:ph": (t_op, ph)})

        excursions = det.detect_excursions(op)
        assert len(excursions) == 1
        assert excursions["excursion_mode"].iloc[0] == "hull"

    def test_too_few_cpps_raises(self):
        with pytest.raises(ValueError):
            DesignSpaceEvents(pd.DataFrame(), cpp_uuids=["only:one"])

    def test_detect_excursions_before_fit_raises(self):
        qual = self._qualification_df()
        det = DesignSpaceEvents(qual, cpp_uuids=["cpp:temp", "cpp:ph"])
        with pytest.raises(RuntimeError):
            det.detect_excursions(qual)


# ===========================================================================
# GoldenBatchDeviationEvents
# ===========================================================================


class TestGoldenBatchDeviationEvents:

    def _golden_trace(self, n=128):
        t = pd.date_range("2024-01-01", periods=n, freq="1min")
        # Smooth ramp-then-plateau.
        v = np.concatenate([np.linspace(20.0, 80.0, n // 2), np.full(n - n // 2, 80.0)])
        return _make_long_df(t, v, "sig:temp")

    def test_compare_pointwise_zero_when_identical(self):
        ref = self._golden_trace()
        det = GoldenBatchDeviationEvents(ref, signal_uuid="sig:temp", n_resample=64)
        out = det.compare(ref, mode="pointwise")
        assert len(out) == 1
        assert out["mode"].iloc[0] == "pointwise"
        assert out["deviation_score"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out["max_abs_residual"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_compare_pointwise_detects_offset(self):
        ref = self._golden_trace()
        det = GoldenBatchDeviationEvents(ref, signal_uuid="sig:temp", n_resample=64)

        # Candidate batch is the golden trace shifted up by 5.
        t = pd.date_range("2024-02-01", periods=128, freq="1min")
        v = (
            np.concatenate([np.linspace(20.0, 80.0, 64), np.full(64, 80.0)]) + 5.0
        )
        cand = _make_long_df(t, v, "sig:temp")

        out = det.compare(cand, mode="pointwise")
        # Resampling onto the same normalised grid → max_abs_residual is ~5.
        assert out["max_abs_residual"].iloc[0] == pytest.approx(5.0, abs=0.05)

    def test_compare_area_mode_returns_positive_score_on_deviation(self):
        ref = self._golden_trace()
        det = GoldenBatchDeviationEvents(ref, signal_uuid="sig:temp", n_resample=64)
        t = pd.date_range("2024-02-01", periods=128, freq="1min")
        v = np.concatenate([np.linspace(20.0, 80.0, 64), np.full(64, 80.0)]) + 3.0
        cand = _make_long_df(t, v, "sig:temp")
        out = det.compare(cand, mode="area")
        assert out["mode"].iloc[0] == "area"
        assert out["deviation_score"].iloc[0] > 0

    def test_compare_dtw_mode_shape_distance(self):
        ref = self._golden_trace()
        det = GoldenBatchDeviationEvents(ref, signal_uuid="sig:temp", n_resample=32)
        # Same shape, slower batch (twice as many samples, same min/max).
        t = pd.date_range("2024-02-01", periods=256, freq="1min")
        v = np.concatenate([np.linspace(20.0, 80.0, 128), np.full(128, 80.0)])
        cand = _make_long_df(t, v, "sig:temp")
        out = det.compare(cand, mode="dtw")
        assert out["mode"].iloc[0] == "dtw"
        # Shape is preserved under time-warping; DTW distance should be small.
        assert out["deviation_score"].iloc[0] == pytest.approx(0.0, abs=0.5)

    def test_unknown_mode_raises(self):
        ref = self._golden_trace()
        det = GoldenBatchDeviationEvents(ref, signal_uuid="sig:temp")
        with pytest.raises(ValueError):
            det.compare(ref, mode="not-a-mode")


# ===========================================================================
# RecipePhaseAdherenceEvents
# ===========================================================================


class TestRecipePhaseAdherenceEvents:

    def _phase_and_value_df(self):
        # Three phases: heat_up (10 min, 20→80°C), hold (15 min @ 80°C),
        # cool_down (10 min, 80→25°C).
        t_heat = pd.date_range("2024-01-01 00:00", periods=10, freq="1min")
        t_hold = pd.date_range("2024-01-01 00:10", periods=15, freq="1min")
        t_cool = pd.date_range("2024-01-01 00:25", periods=10, freq="1min")
        t_all = t_heat.append([t_hold, t_cool])

        v_heat = np.linspace(20.0, 80.0, 10)
        v_hold = np.full(15, 80.0)
        v_cool = np.linspace(80.0, 25.0, 10)
        v_all = np.concatenate([v_heat, v_hold, v_cool])

        # Build the phase signal in long form with value_string.
        phase_names = ["heat_up"] * 10 + ["hold"] * 15 + ["cool_down"] * 10
        phase_df = pd.DataFrame(
            {
                "systime": t_all,
                "uuid": ["phase:reactor"] * len(t_all),
                "value_string": phase_names,
                "value_double": [float("nan")] * len(t_all),
            }
        )
        value_df = pd.DataFrame(
            {
                "systime": t_all,
                "uuid": ["temp:reactor"] * len(t_all),
                "value_double": v_all,
                "value_string": [None] * len(t_all),
            }
        )
        return pd.concat([phase_df, value_df], ignore_index=True)

    def test_evaluate_all_pass(self):
        df = self._phase_and_value_df()
        spec = {
            "heat_up": {"ramp_rate_max": 5.0},  # actual ramp ≈ 0.1°C/s
            "hold": {"hold_value": (78.0, 82.0)},
            "cool_down": {"trough_value": (20.0, 30.0)},
        }
        det = RecipePhaseAdherenceEvents(
            df, phase_uuid="phase:reactor", value_uuid="temp:reactor", spec=spec
        )
        out = det.evaluate()
        assert len(out) == 3
        assert out["pass"].all(), out

    def test_evaluate_flags_hold_value_violation(self):
        df = self._phase_and_value_df()
        spec = {"hold": {"hold_value": (60.0, 70.0)}}  # actual hold ≈ 80°C
        det = RecipePhaseAdherenceEvents(
            df, phase_uuid="phase:reactor", value_uuid="temp:reactor", spec=spec
        )
        out = det.evaluate()
        hold_row = out[out["phase"] == "hold"].iloc[0]
        assert hold_row["pass"] is False or hold_row["pass"] is np.False_
        assert "hold_value" in hold_row["criteria_failed"]

    def test_unconstrained_phase_passes_with_empty_criteria(self):
        df = self._phase_and_value_df()
        det = RecipePhaseAdherenceEvents(
            df, phase_uuid="phase:reactor", value_uuid="temp:reactor", spec={}
        )
        out = det.evaluate()
        # All phases are observed but unconstrained → pass with empty failed list.
        assert (out["pass"]).all()
        assert all(len(x) == 0 for x in out["criteria_failed"])


# ===========================================================================
# CriticalParameterRankingEvents
# ===========================================================================


class TestCriticalParameterRankingEvents:

    def _per_run_table(self, n=40):
        rng = np.random.default_rng(123)
        x_strong = rng.normal(size=n)
        x_weak = rng.normal(size=n)
        noise = rng.normal(scale=0.5, size=n)
        outcome = 3.0 * x_strong + 0.1 * x_weak + noise
        return pd.DataFrame(
            {"strong": x_strong, "weak": x_weak, "yield": outcome}
        )

    def test_rank_orders_strong_first_with_pearson(self):
        runs = self._per_run_table()
        det = CriticalParameterRankingEvents(pd.DataFrame())
        out = det.rank(
            runs, candidate_columns=["strong", "weak"], outcome_column="yield", method="pearson"
        )
        assert list(out["parameter"])[0] == "strong"
        assert out.iloc[0]["abs_effect_size"] > out.iloc[1]["abs_effect_size"]

    def test_rank_supports_spearman(self):
        runs = self._per_run_table()
        det = CriticalParameterRankingEvents(pd.DataFrame())
        out = det.rank(
            runs,
            candidate_columns=["strong", "weak"],
            outcome_column="yield",
            method="spearman",
        )
        assert list(out["parameter"])[0] == "strong"

    def test_rank_supports_anova_f(self):
        # Distinct groups: outcome jumps by group label.
        runs = pd.DataFrame(
            {
                "x": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
                "noise": np.random.default_rng(0).normal(size=12),
                "yield": [10, 10.2, 9.8, 20, 19.7, 20.3, 30, 30.2, 29.8, 40, 40.1, 39.9],
            }
        )
        det = CriticalParameterRankingEvents(pd.DataFrame())
        out = det.rank(
            runs,
            candidate_columns=["x", "noise"],
            outcome_column="yield",
            method="anova_f",
            anova_bins=4,
        )
        assert list(out["parameter"])[0] == "x"
        assert out.iloc[0]["p_value"] < 0.01

    def test_top_drivers_filters_by_alpha(self):
        runs = self._per_run_table()
        det = CriticalParameterRankingEvents(pd.DataFrame())
        out = det.top_drivers(
            runs,
            candidate_columns=["strong", "weak"],
            outcome_column="yield",
            method="pearson",
            alpha=1e-6,  # strict — only the strong driver should pass.
        )
        assert "weak" not in set(out["parameter"])

    def test_unknown_outcome_column_raises(self):
        runs = self._per_run_table()
        det = CriticalParameterRankingEvents(pd.DataFrame())
        with pytest.raises(ValueError):
            det.rank(
                runs, candidate_columns=["strong"], outcome_column="not-a-column"
            )

    def test_unknown_method_raises(self):
        runs = self._per_run_table()
        det = CriticalParameterRankingEvents(pd.DataFrame())
        with pytest.raises(ValueError):
            det.rank(
                runs,
                candidate_columns=["strong"],
                outcome_column="yield",
                method="bogus",
            )
