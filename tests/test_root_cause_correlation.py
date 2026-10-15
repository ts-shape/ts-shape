"""Tests for RootCauseCorrelationEvents (event-driven root-cause ranking)."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.correlation import RootCauseCorrelationEvents

GRID = pd.date_range("2024-06-01 00:00", periods=600, freq="1min")
# Every stoppage starts here; the true cause spikes five minutes earlier.
ONSETS = [90, 210, 330, 450, 570]
LEAD_BINS = 5


def _signal(uuid, values, times=GRID):
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": [uuid] * len(times),
            "value_double": np.asarray(values, dtype=float),
        }
    )


def _leading_signal(uuid="press:hydraulic", lead=LEAD_BINS):
    """Flat signal that spikes ``lead`` bins before every stoppage."""
    values = np.zeros(len(GRID))
    for onset in ONSETS:
        values[onset - lead] = 10.0
    return _signal(uuid, values)


def _unrelated_signal(uuid="flow:coolant", seed=3):
    rng = np.random.default_rng(seed)
    return _signal(uuid, rng.normal(20.0, 1.0, len(GRID)))


def _trigger_events(column="systime"):
    return pd.DataFrame(
        {
            column: GRID[ONSETS],
            "uuid": ["prod:long_downtime"] * len(ONSETS),
            "source_uuid": ["state:line-1"] * len(ONSETS),
        }
    )


def _detector():
    df = pd.concat([_leading_signal(), _unrelated_signal()])
    return RootCauseCorrelationEvents(df)


class TestOnsetExtraction:

    def test_interval_events_use_the_start_column(self):
        det = _detector()
        intervals = _trigger_events(column="start")
        intervals["end"] = intervals["start"] + pd.Timedelta("10min")
        ranked = det.rank_causes(intervals, ["press:hydraulic", "flow:coolant"])
        assert ranked["candidate_uuid"].iloc[0] == "press:hydraulic"

    def test_frame_without_onset_column_is_rejected(self):
        det = _detector()
        with pytest.raises(ValueError, match="no onset column"):
            det.rank_causes(pd.DataFrame({"foo": [1]}), ["press:hydraulic"])

    def test_non_dataframe_trigger_is_rejected(self):
        det = _detector()
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            det.rank_causes([{"systime": GRID[0]}], ["press:hydraulic"])

    def test_empty_trigger_frame_returns_typed_empty_result(self):
        det = _detector()
        out = det.rank_causes(pd.DataFrame(), ["press:hydraulic"])
        assert out.empty
        assert "correlation_coefficient" in out.columns


class TestRankCauses:

    def test_leading_signal_outranks_noise(self):
        ranked = _detector().rank_causes(
            _trigger_events(), ["press:hydraulic", "flow:coolant"]
        )
        assert list(ranked["rank"]) == [1, 2]
        assert ranked["candidate_uuid"].iloc[0] == "press:hydraulic"
        assert ranked["significant"].iloc[0]
        assert ranked["p_value"].iloc[0] < 0.05

    def test_recovered_lag_matches_the_injected_lead_time(self):
        ranked = _detector().rank_causes(
            _trigger_events(), ["press:hydraulic", "flow:coolant"]
        )
        assert ranked["lag_seconds"].iloc[0] == LEAD_BINS * 60

    def test_unrelated_signal_is_not_significant(self):
        ranked = _detector().rank_causes(
            _trigger_events(), ["press:hydraulic", "flow:coolant"]
        )
        noise = ranked[ranked["candidate_uuid"] == "flow:coolant"].iloc[0]
        assert not noise["significant"]

    def test_lag_scan_is_bonferroni_corrected(self):
        """Significance is judged on the corrected p-value, not the raw one."""
        ranked = _detector().rank_causes(
            _trigger_events(), ["press:hydraulic", "flow:coolant"], max_lag=30
        )
        noise = ranked[ranked["candidate_uuid"] == "flow:coolant"].iloc[0]
        assert noise["p_value_adjusted"] == pytest.approx(
            min(1.0, noise["p_value"] * 31), rel=1e-4
        )
        assert noise["p_value_adjusted"] > noise["p_value"]
        # A real cause survives the correction; the noise does not.
        assert ranked["significant"].tolist() == [True, False]

    def test_significant_only_filters_the_noise_out(self):
        ranked = _detector().rank_causes(
            _trigger_events(),
            ["press:hydraulic", "flow:coolant"],
            only_significant=True,
        )
        assert ranked["candidate_uuid"].tolist() == ["press:hydraulic"]

    def test_trigger_identity_is_carried_through(self):
        ranked = _detector().rank_causes(_trigger_events(), ["press:hydraulic"])
        assert ranked["trigger_event_type"].iloc[0] == "prod:long_downtime"
        assert ranked["trigger_uuid"].iloc[0] == "state:line-1"
        assert ranked["trigger_count"].iloc[0] == len(ONSETS)

    def test_explicit_trigger_labels_override_the_frame(self):
        ranked = _detector().rank_causes(
            _trigger_events(),
            ["press:hydraulic"],
            trigger_event_type="spc_violation",
            trigger_uuid="quality:diameter",
        )
        assert ranked["trigger_event_type"].iloc[0] == "spc_violation"
        assert ranked["trigger_uuid"].iloc[0] == "quality:diameter"

    def test_only_leading_lags_are_searched(self):
        """A signal that reacts *after* the event must not rank as its cause."""
        following = np.zeros(len(GRID))
        for onset in ONSETS:
            following[onset + LEAD_BINS] = 10.0
        df = pd.concat([_leading_signal(), _signal("temp:after", following)])
        ranked = RootCauseCorrelationEvents(df).rank_causes(
            _trigger_events(), ["press:hydraulic", "temp:after"]
        )
        assert ranked["candidate_uuid"].iloc[0] == "press:hydraulic"
        after = ranked[ranked["candidate_uuid"] == "temp:after"].iloc[0]
        assert (
            after["correlation_coefficient"] < ranked["correlation_coefficient"].iloc[0]
        )

    def test_max_lag_shorter_than_the_lead_misses_the_peak(self):
        ranked = _detector().rank_causes(
            _trigger_events(), ["press:hydraulic"], max_lag=2
        )
        assert ranked["lag_seconds"].iloc[0] <= 2 * 60
        assert not ranked["significant"].iloc[0]

    def test_negative_max_lag_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            _detector().rank_causes(_trigger_events(), ["press:hydraulic"], max_lag=-1)

    def test_unknown_candidate_is_skipped(self):
        ranked = _detector().rank_causes(
            _trigger_events(), ["press:hydraulic", "does:not-exist"]
        )
        assert ranked["candidate_uuid"].tolist() == ["press:hydraulic"]

    def test_constant_signal_produces_no_row(self):
        df = pd.concat([_signal("flat:signal", np.full(len(GRID), 7.0))])
        ranked = RootCauseCorrelationEvents(df).rank_causes(
            _trigger_events(), ["flat:signal"]
        )
        assert ranked.empty


class TestTopCause:

    def test_names_the_leading_signal(self):
        row = (
            _detector()
            .top_cause(_trigger_events(), ["press:hydraulic", "flow:coolant"])
            .iloc[0]
        )
        assert row["top_cause_uuid"] == "press:hydraulic"
        assert row["top_cause_lag_seconds"] == LEAD_BINS * 60
        assert row["candidates_tested"] == 2
        assert row["candidates_significant"] == 1
        assert 0.0 < row["confidence"] <= 1.0

    def test_reports_the_search_when_nothing_qualifies(self):
        df = pd.concat([_unrelated_signal(), _unrelated_signal("noise:2", seed=11)])
        row = (
            RootCauseCorrelationEvents(df)
            .top_cause(_trigger_events(), ["flow:coolant", "noise:2"])
            .iloc[0]
        )
        assert row["top_cause_uuid"] is None
        assert row["candidates_significant"] == 0
        assert row["candidates_tested"] == 2
        assert row["confidence"] == 0.0

    def test_empty_input_returns_typed_empty_frame(self):
        out = _detector().top_cause(pd.DataFrame(), ["press:hydraulic"])
        assert out.empty
        assert "top_cause_uuid" in out.columns


class TestLagProfile:

    def test_profile_peaks_at_the_injected_lead(self):
        profile = _detector().lag_profile(_trigger_events(), "press:hydraulic")
        best = profile[profile["is_best_lag"]]
        assert len(best) == 1
        assert best["lag_seconds"].iloc[0] == LEAD_BINS * 60

    def test_profile_covers_every_lag(self):
        profile = _detector().lag_profile(
            _trigger_events(), "press:hydraulic", max_lag=10
        )
        assert len(profile) == 11
        assert profile["lag_seconds"].min() == 0.0
        assert profile["lag_seconds"].max() == 10 * 60

    def test_unknown_candidate_returns_typed_empty_frame(self):
        out = _detector().lag_profile(_trigger_events(), "does:not-exist")
        assert out.empty
        assert "is_best_lag" in out.columns
