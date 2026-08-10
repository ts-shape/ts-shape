"""Tests for AssetHealthIndexEvents (multi-signal asset health fusion)."""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from ts_shape.events.maintenance import AssetHealthIndexEvents

TIMES = pd.date_range("2024-03-01 00:00", periods=48 * 6, freq="10min")


def _signal(times, values, uuid):
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": [uuid] * len(times),
            "value_double": values,
            "is_delta": [True] * len(times),
        }
    )


def _stable(uuid="temp", level=50.0, times=TIMES):
    rng = np.random.default_rng(7)
    return _signal(times, level + rng.normal(0, 0.05, len(times)), uuid)


def _degrading(uuid="vib", start=2.0, per_sample=0.05, times=TIMES):
    return _signal(times, start + np.arange(len(times)) * per_sample, uuid)


def _empty_df():
    return pd.DataFrame(
        {
            "systime": pd.Series(dtype="datetime64[ns]"),
            "uuid": pd.Series(dtype="str"),
            "value_double": pd.Series(dtype="float64"),
        }
    )


class TestConfiguration:

    def test_string_shorthand_expands_to_mode_and_weight(self):
        det = AssetHealthIndexEvents(
            _stable(), asset_id="press-01", signals={"temp": "thermal"}
        )
        assert det.signals == {"temp": {"failure_mode": "thermal", "weight": 1.0}}

    def test_mapping_spec_keeps_weight(self):
        det = AssetHealthIndexEvents(
            _stable(),
            asset_id="press-01",
            signals={"temp": {"failure_mode": "thermal", "weight": 3.0}},
        )
        assert det.signals["temp"]["weight"] == 3.0

    def test_empty_signal_map_rejected(self):
        with pytest.raises(ValueError, match="at least one signal"):
            AssetHealthIndexEvents(_stable(), asset_id="press-01", signals={})

    def test_non_positive_weight_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            AssetHealthIndexEvents(
                _stable(),
                asset_id="press-01",
                signals={"temp": {"failure_mode": "thermal", "weight": 0.0}},
            )


class TestHealthIndex:

    def test_stable_asset_scores_near_perfect(self):
        df = pd.concat([_stable("temp"), _stable("curr", level=12.0)])
        det = AssetHealthIndexEvents(
            df,
            asset_id="press-01",
            signals={"temp": "thermal", "curr": "electrical"},
        )
        out = det.health_index(window="4h", baseline_window="8h")
        assert not out.empty
        assert out["asset_health_score"].min() > 90.0
        assert (out["confidence_level"] == 1.0).all()
        assert (out["signals_available"] == 2).all()
        assert (out["source_uuid"] == "press-01").all()

    def test_degrading_signal_drags_the_index_down(self):
        df = pd.concat([_stable("temp"), _degrading("vib")])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        out = det.health_index(window="4h", baseline_window="8h")
        assert out["asset_health_score"].iloc[-1] < out["asset_health_score"].iloc[0]
        assert out["dominant_failure_mode"].iloc[-1] == "bearing"
        assert out["worst_signal_uuid"].iloc[-1] == "vib"

    def test_index_is_between_the_component_scores(self):
        """Equal weights → the fused score sits between the two signals."""
        df = pd.concat([_stable("temp"), _degrading("vib")])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        fused = det.health_index(window="4h", baseline_window="8h")
        temp_only = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal"}
        ).health_index(window="4h", baseline_window="8h")
        vib_only = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"vib": "bearing"}
        ).health_index(window="4h", baseline_window="8h")
        last = fused["asset_health_score"].iloc[-1]
        assert vib_only["asset_health_score"].iloc[-1] <= last
        assert last <= temp_only["asset_health_score"].iloc[-1]

    def test_weight_shifts_the_fused_score(self):
        df = pd.concat([_stable("temp"), _degrading("vib")])
        equal = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        ).health_index(window="4h", baseline_window="8h")
        vib_heavy = AssetHealthIndexEvents(
            df,
            asset_id="press-01",
            signals={
                "temp": "thermal",
                "vib": {"failure_mode": "bearing", "weight": 4.0},
            },
        ).health_index(window="4h", baseline_window="8h")
        assert (
            vib_heavy["asset_health_score"].iloc[-1]
            < equal["asset_health_score"].iloc[-1]
        )

    def test_missing_signal_lowers_confidence_but_still_scores(self):
        df = pd.concat([_stable("temp"), _degrading("vib")])
        det = AssetHealthIndexEvents(
            df,
            asset_id="press-01",
            signals={"temp": "thermal", "vib": "bearing", "curr": "electrical"},
        )
        out = det.health_index(window="4h", baseline_window="8h")
        assert not out.empty
        assert (out["signals_available"] == 2).all()
        assert (out["signals_expected"] == 3).all()
        assert out["confidence_level"].iloc[0] == pytest.approx(2 / 3, abs=1e-4)

    def test_empty_input_returns_typed_empty_frame(self):
        det = AssetHealthIndexEvents(
            _empty_df(), asset_id="press-01", signals={"temp": "thermal"}
        )
        out = det.health_index()
        assert out.empty
        assert "asset_health_score" in out.columns

    def test_unknown_signal_uuid_returns_empty_frame(self):
        det = AssetHealthIndexEvents(
            _stable("temp"), asset_id="press-01", signals={"nope": "thermal"}
        )
        assert det.health_index(window="4h").empty


class TestHealthEvents:

    def test_crossing_emits_one_event_per_band(self):
        df = pd.concat([_stable("temp"), _degrading("vib", per_sample=0.5)])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        events = det.health_events(
            window="4h",
            baseline_window="8h",
            warning_threshold=95.0,
            critical_threshold=80.0,
        )
        assert not events.empty
        assert set(events["severity"]) <= {"warning", "critical"}
        # One row per entry into a band, not one per window below it.
        assert events["severity"].value_counts().max() == 1
        assert (events["direction"] == "below").all()

    def test_repeat_mode_emits_every_window_below_threshold(self):
        df = pd.concat([_stable("temp"), _degrading("vib", per_sample=0.5)])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        crossings = det.health_events(
            window="4h",
            baseline_window="8h",
            warning_threshold=95.0,
            critical_threshold=80.0,
        )
        every = det.health_events(
            window="4h",
            baseline_window="8h",
            warning_threshold=95.0,
            critical_threshold=80.0,
            on_crossing_only=False,
        )
        assert len(every) > len(crossings)

    def test_healthy_asset_emits_nothing(self):
        df = pd.concat([_stable("temp"), _stable("curr", level=12.0)])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "curr": "electrical"}
        )
        assert det.health_events(window="4h", baseline_window="8h").empty

    def test_inverted_thresholds_rejected(self):
        det = AssetHealthIndexEvents(
            _stable(), asset_id="press-01", signals={"temp": "thermal"}
        )
        with pytest.raises(ValueError, match="must not be above"):
            det.health_events(warning_threshold=40.0, critical_threshold=60.0)


class TestSignalContributions:

    def test_degrading_signal_ranks_first(self):
        df = pd.concat([_stable("temp"), _degrading("vib")])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        out = det.signal_contributions(window="4h", baseline_window="8h")
        assert list(out["rank"]) == [1, 2]
        assert out["signal_uuid"].iloc[0] == "vib"
        assert out["contribution_pct"].iloc[0] > out["contribution_pct"].iloc[1]
        assert out["contribution_pct"].sum() == pytest.approx(100.0, abs=0.1)

    def test_healthy_asset_has_no_deficit_to_distribute(self):
        df = pd.concat([_stable("temp"), _stable("curr", level=12.0)])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "curr": "electrical"}
        )
        out = det.signal_contributions(window="4h", baseline_window="8h")
        assert not out.empty
        assert out["mean_health"].min() > 90.0

    def test_empty_input_returns_typed_empty_frame(self):
        det = AssetHealthIndexEvents(
            _empty_df(), asset_id="press-01", signals={"temp": "thermal"}
        )
        out = det.signal_contributions()
        assert out.empty
        assert "contribution_pct" in out.columns


class TestMaintenanceForecast:

    def test_declining_health_yields_a_finite_horizon(self):
        df = pd.concat(
            [_stable("temp"), _degrading("vib", start=100.0, per_sample=0.1)]
        )
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        out = det.maintenance_forecast(
            window="2h", baseline_window="6h", critical_threshold=40.0
        )
        assert len(out) == 1
        row = out.iloc[0]
        assert row["decline_per_day"] > 0
        assert row["confidence"] >= 0.5
        assert row["recommended_maintenance_days"] > 0
        assert pd.notna(row["forecast_time"])
        assert row["forecast_time"] > row["end"]

    def test_low_confidence_trend_reports_no_horizon(self):
        """A saturating index declines on average but fits badly — no date."""
        df = pd.concat([_stable("temp"), _degrading("vib", per_sample=0.2)])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        row = det.maintenance_forecast(
            window="2h", baseline_window="6h", critical_threshold=40.0
        ).iloc[0]
        assert row["decline_per_day"] > 0
        assert row["confidence"] < 0.5
        assert pd.isna(row["recommended_maintenance_days"])

        forced = det.maintenance_forecast(
            window="2h",
            baseline_window="6h",
            critical_threshold=40.0,
            min_confidence=0.0,
        ).iloc[0]
        assert forced["recommended_maintenance_days"] > 0

    def test_stable_health_has_no_horizon(self):
        df = pd.concat([_stable("temp"), _stable("curr", level=12.0)])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "curr": "electrical"}
        )
        out = det.maintenance_forecast(window="4h", baseline_window="8h")
        assert pd.isna(out["recommended_maintenance_days"].iloc[0])
        assert pd.isna(out["forecast_time"].iloc[0])

    def test_already_critical_asset_needs_maintenance_now(self):
        df = pd.concat([_stable("temp"), _degrading("vib", per_sample=1.0)])
        det = AssetHealthIndexEvents(
            df, asset_id="press-01", signals={"temp": "thermal", "vib": "bearing"}
        )
        out = det.maintenance_forecast(
            window="2h", baseline_window="4h", critical_threshold=95.0
        )
        assert out["recommended_maintenance_days"].iloc[0] == 0.0

    def test_too_little_data_returns_typed_empty_frame(self):
        det = AssetHealthIndexEvents(
            _stable("temp", times=pd.date_range("2024-03-01", periods=5, freq="1min")),
            asset_id="press-01",
            signals={"temp": "thermal"},
        )
        out = det.maintenance_forecast(window="1D")
        assert out.empty
        assert "recommended_maintenance_days" in out.columns
