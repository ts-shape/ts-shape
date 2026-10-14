import numpy as np
import pandas as pd
import pytest

from ts_shape.events.maintenance.reliability import ReliabilityEvents
from ts_shape.events.production.oee_calculator import OEEEvents

T0 = pd.Timestamp("2024-03-30 12:00", tz="Europe/Berlin")


def _fault_frame(points, uuid="fault", start=T0):
    """(hours_from_start, failed) -> change-encoded fault-bit rows."""
    return pd.DataFrame(
        {
            "systime": [start + pd.Timedelta(hours=h) for h, _ in points],
            "uuid": uuid,
            "value_bool": [v for _, v in points],
        }
    )


# up 10 h, down 1 h, up 20 h, down 2 h, up 5 h (still running at data end)
HAND = [(0, False), (10, True), (11, False), (31, True), (33, False), (38, False)]


def test_failure_events_hand_example():
    out = ReliabilityEvents(_fault_frame(HAND), "fault").failure_events()
    assert list(out["repair_hours"]) == [1.0, 2.0]
    assert list(out["operating_hours_before"]) == [10.0, 20.0]
    assert list(out["first_observed"]) == [True, False]
    assert out["start"].iloc[1] == T0 + pd.Timedelta(hours=31)
    assert str(out["start"].dt.tz) == "Europe/Berlin"


def test_mtbf_mttr_hand_example():
    row = ReliabilityEvents(_fault_frame(HAND), "fault").mtbf_mttr().iloc[0]
    assert row["operating_hours"] == pytest.approx(35)
    assert row["repair_hours"] == pytest.approx(3)
    assert row["failures"] == 2
    assert row["mtbf_hours"] == pytest.approx(17.5)
    # ISO 22400-2: averaged over FE + 1 operating periods
    assert row["mtbf_iso22400_hours"] == pytest.approx(35 / 3)
    assert row["mttr_hours"] == pytest.approx(1.5)
    assert row["availability_pct"] == pytest.approx(100 * 35 / 38)
    assert row["failure_rate_per_hour"] == pytest.approx(2 / 35)


def test_windows_split_time_exactly_and_follow_dst():
    out = ReliabilityEvents(_fault_frame(HAND), "fault").mtbf_mttr(window="1D")
    # 2024-03-31 is the 23 h spring-forward day in Berlin
    assert list(out["duration_seconds"]) == [86400, 82800, 86400]
    assert out["operating_hours"].sum() == pytest.approx(35)
    assert out["repair_hours"].sum() == pytest.approx(3)
    assert list(out["failures"]) == [1, 1, 0]
    assert np.isnan(out["mtbf_hours"].iloc[2])


def test_data_gap_is_neither_operating_nor_repair():
    # 12 h logging outage while "up": without max_gap it is booked as
    # operating time, with max_gap it is no data.
    pts = [(0, False), (2, False), (14, False), (15, True), (16, False), (18, False)]
    df = _fault_frame(pts)
    naive = ReliabilityEvents(df, "fault").mtbf_mttr().iloc[0]
    capped = ReliabilityEvents(df, "fault", max_gap="3h").mtbf_mttr().iloc[0]
    assert naive["operating_hours"] == pytest.approx(17)
    assert capped["operating_hours"] == pytest.approx(5)
    assert capped["no_data_hours"] == pytest.approx(12)
    assert capped["repair_hours"] == pytest.approx(1)


def test_failure_split_by_gap_counts_once():
    pts = [(0, False), (5, True), (6, True), (20, True), (21, False), (25, False)]
    rel = ReliabilityEvents(_fault_frame(pts), "fault", max_gap="2h")
    ev = rel.failure_events()
    assert len(ev) == 1
    assert ev["repair_hours"].iloc[0] == pytest.approx(2)  # 5-6 and 20-21


def test_status_codes_other_uuids_and_data_end():
    df = pd.DataFrame(
        {
            "systime": [T0 + pd.Timedelta(hours=h) for h in (0, 4, 5, 9)],
            "uuid": "status",
            "value_string": ["RUN", "BREAKDOWN", "RUN", "IDLE"],
        }
    )
    noise = _fault_frame([(0, True), (50, False)], uuid="other")
    rel = ReliabilityEvents(
        pd.concat([df, noise], ignore_index=True),
        "status",
        failure_states=("BREAKDOWN",),
        value_column="value_string",
        data_end=T0 + pd.Timedelta(hours=12),
    )
    row = rel.mtbf_mttr().iloc[0]
    assert row["failures"] == 1
    assert row["operating_hours"] == pytest.approx(11)  # idle is not a failure
    assert row["repair_hours"] == pytest.approx(1)


def _weibull_history(beta, eta, n, seed):
    rng = np.random.default_rng(seed)
    tbf = eta * rng.weibull(beta, n)
    pts, t = [(0.0, False)], 0.0
    for life in tbf:
        t += life
        pts.append((t, True))
        t += 0.5
        pts.append((t, False))
    pts.append((t + 0.3 * eta, False))  # still running: right-censored
    return _fault_frame(pts, start=pd.Timestamp("2020-01-01", tz="UTC"))


@pytest.mark.parametrize("beta,pattern", [(0.6, "infant_mortality"), (2.5, "wear_out")])
def test_weibull_fit_recovers_parameters(beta, pattern):
    eta = 200.0
    rel = ReliabilityEvents(_weibull_history(beta, eta, 400, seed=1), "fault")
    fit = rel.weibull_fit().iloc[0]
    assert fit["beta"] == pytest.approx(beta, rel=0.12)
    assert fit["eta_hours"] == pytest.approx(eta, rel=0.12)
    assert fit["n_censored"] == 1
    assert fit["failure_pattern"] == pattern
    from math import gamma, log

    assert fit["mttf_hours"] == pytest.approx(
        fit["eta_hours"] * gamma(1 + 1 / fit["beta"])
    )
    assert fit["b10_hours"] == pytest.approx(
        fit["eta_hours"] * (-log(0.9)) ** (1 / fit["beta"])
    )


def test_reliability_curve_and_conditional_reliability():
    rel = ReliabilityEvents(_weibull_history(2.0, 100.0, 300, seed=2), "fault")
    fit = rel.weibull_fit().iloc[0]
    b, e = fit["beta"], fit["eta_hours"]
    curve = rel.reliability_at([0, e])
    assert curve["reliability"].iloc[0] == pytest.approx(1.0)
    assert curve["reliability"].iloc[1] == pytest.approx(np.exp(-1))
    cond = rel.reliability_at([50], age_hours=80).iloc[0]
    expected = np.exp(-((130 / e) ** b)) / np.exp(-((80 / e) ** b))
    assert cond["reliability"] == pytest.approx(expected)
    assert cond["hazard_per_hour"] == pytest.approx(b / e * (130 / e) ** (b - 1))


def test_empty_and_too_few_failures():
    empty = pd.DataFrame(columns=["systime", "uuid", "value_bool"])
    rel = ReliabilityEvents(empty, "fault")
    assert rel.failure_events().empty
    assert rel.mtbf_mttr().empty
    assert rel.weibull_fit().empty
    assert rel.reliability_at([10]).empty
    one = ReliabilityEvents(_fault_frame([(0, False), (5, True), (6, False)]), "fault")
    assert one.weibull_fit().empty
    assert np.isnan(
        ReliabilityEvents(_fault_frame([(0, False), (5, False)]), "fault")
        .mtbf_mttr()["mtbf_hours"]
        .iloc[0]
    )


def test_oee_iso22400_reference_example():
    """ISO 22400-2 OEE = availability x effectiveness x quality rate.

    8 h planned shift: run 06-08 and 09-13 (6 h), 3000 parts at an ideal
    cycle of 6 s, 150 rejects:
    A = 6/8 = 75 %, P = 3000 * 6 s / 6 h = 83.33 %, Q = 2850/3000 = 95 %,
    OEE = 59.375 %.
    """
    day = pd.Timestamp("2024-05-06")
    h = lambda x: day + pd.Timedelta(hours=x)  # noqa: E731
    rows = [
        *[
            (h(x), "run", v, None)
            for x, v in [(6, True), (8, False), (9, True), (13, False), (14, False)]
        ],
        (h(6), "parts", None, 0),
        (h(14), "parts", None, 3000),
        (h(6), "total", None, 0),
        (h(14), "total", None, 3000),
        (h(6), "reject", None, 0),
        (h(14), "reject", None, 150),
    ]
    df = pd.DataFrame(rows, columns=["systime", "uuid", "value_bool", "value_integer"])
    row = (
        OEEEvents(df)
        .oee(
            "run",
            "parts",
            6.0,
            total_uuid="total",
            reject_uuid="reject",
            planned_time_hours=8,
        )
        .iloc[0]
    )
    assert row["availability_pct"] == pytest.approx(75.0, abs=0.01)
    assert row["performance_pct"] == pytest.approx(83.33, abs=0.01)
    assert row["quality_pct"] == pytest.approx(95.0, abs=0.01)
    assert row["oee_pct"] == pytest.approx(59.375, abs=0.01)
