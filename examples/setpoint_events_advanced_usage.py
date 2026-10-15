"""Setpoint response analysis with SetpointChangeEvents.

Simulates control loops on ``ts_shape.make_timeseries`` data (a setpoint that
steps, and a process value that follows it as a second-order system plus
sensor noise), then walks through the response KPIs: step detection with
noise filtering, settling time (absolute, percentage and derivative based),
overshoot, rise time, decay and oscillation, the combined control-quality
table, a before/after tuning comparison and a multi-loop report.

Run: python examples/setpoint_events_advanced_usage.py
"""

import numpy as np
import pandas as pd

import ts_shape
from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents

FREQ = "1s"
N_POINTS = 3600  # one hour per loop
STEPS = [(0, 50.0), (600, 80.0), (1800, 60.0), (2700, 90.0)]  # (second, setpoint)


def make_loop(
    name: str, *, zeta: float, omega: float, noise: float, seed: int
) -> pd.DataFrame:
    """Setpoint ``SP_<name>`` and process value ``PV_<name>`` for one loop.

    The process value is a second-order response with damping ``zeta`` and
    natural frequency ``omega`` (rad/s) to the setpoint steps; the sensor
    noise comes from ``make_timeseries``.
    """
    pv = ts_shape.make_timeseries(
        [f"PV_{name}"],
        n_points=N_POINTS,
        freq=FREQ,
        baseline=0.0,
        noise=noise,
        seed=seed,
    )
    setpoint = np.empty(N_POINTS)
    for second, value in STEPS:
        setpoint[second:] = value

    # x'' = omega^2 (sp - x) - 2 zeta omega x', integrated once per second
    x, v = STEPS[0][1], 0.0
    response = np.empty(N_POINTS)
    for i in range(N_POINTS):
        for _ in range(10):  # 0.1 s sub-steps keep the integration stable
            a = omega**2 * (setpoint[i] - x) - 2 * zeta * omega * v
            v += a * 0.1
            x += v * 0.1
        response[i] = x
    pv["value_double"] = response + pv["value_double"]

    sp = pv.assign(uuid=f"SP_{name}", value_double=setpoint)
    return pd.concat([sp, pv], ignore_index=True)


def example_basic_enhancements(df):
    """Step detection with noise filtering; absolute vs percentage tolerance."""
    spe = SetpointChangeEvents(df, setpoint_uuid="SP_TEMP_001")

    steps_all = spe.detect_setpoint_steps(min_delta=0.5)
    steps_clean = spe.detect_setpoint_steps(
        min_delta=0.5, filter_noise=True, noise_threshold=0.1
    )
    settle_abs = spe.time_to_settle("PV_TEMP_001", tol=2.0, hold="30s")
    settle_pct = spe.time_to_settle(
        "PV_TEMP_001", settle_pct=0.02, hold="30s", lookahead="15min"
    )

    print("Setpoint steps:")
    print(steps_clean[["start", "prev_level", "new_level", "magnitude"]])
    print(
        f"Steps without / with noise filtering: {len(steps_all)} / {len(steps_clean)}"
    )
    settling = pd.DataFrame(
        {
            "start": settle_abs["start"],
            "abs_2_units_s": settle_abs["t_settle_seconds"],
            "pct_2_s": settle_pct["t_settle_seconds"],
        }
    )
    print("\nSettling time, 2 units vs 2 % of the step:")
    print(settling)
    return steps_clean, settle_pct


def example_derivative_settling(df):
    """Settled when the rate of change stays low, vs settled within a band."""
    spe = SetpointChangeEvents(df, setpoint_uuid="SP_PRESSURE_001")

    settle_deriv = spe.time_to_settle_derivative(
        "PV_PRESSURE_001", rate_threshold=0.05, lookahead="20min", hold="1min"
    )
    settle_error = spe.time_to_settle(
        "PV_PRESSURE_001", settle_pct=0.01, lookahead="20min"
    )

    comparison = pd.DataFrame(
        {
            "start": settle_deriv["start"],
            "deriv_s": settle_deriv["t_settle_seconds"],
            "error_s": settle_error["t_settle_seconds"],
            "deriv_settled": settle_deriv["settled"],
            "error_settled": settle_error["settled"],
        }
    )
    print("\nDerivative-based vs error-based settling:")
    print(comparison)
    return settle_deriv, settle_error


def example_enhanced_overshoot(df):
    """Overshoot, undershoot and oscillation of each step response."""
    spe = SetpointChangeEvents(df, setpoint_uuid="SP_FLOW_001")
    overshoot = spe.overshoot_metrics("PV_FLOW_001", window="10min")

    # overshoot_pct is in percent of the step; oscillation_count counts every
    # crossing of the setpoint, so on a noisy signal it counts the noise too.
    numbers = ["overshoot_pct", "undershoot_pct", "oscillation_count"]
    print("\nTransient response of the underdamped flow loop:")
    print(overshoot[["start", *numbers]].round(dict.fromkeys(numbers, 1)))
    excessive = overshoot[overshoot["overshoot_pct"] > 15]
    print(f"Steps with more than 15 % overshoot: {len(excessive)}")
    return overshoot, excessive


def example_settling_characteristics(df):
    """Rise time, exponential decay and oscillation frequency."""
    spe = SetpointChangeEvents(df, setpoint_uuid="SP_LEVEL_001")

    rise = spe.rise_time("PV_LEVEL_001", start_pct=0.1, end_pct=0.9, lookahead="15min")
    decay = spe.decay_rate("PV_LEVEL_001", lookahead="15min", min_points=10)
    freq = spe.oscillation_frequency("PV_LEVEL_001", window="15min", min_oscillations=2)

    characteristics = pd.DataFrame(
        {
            "start": rise["start"],
            "rise_s": rise["rise_time_seconds"],
            "decay_lambda": decay["decay_rate_lambda"].round(4),
            "decay_r2": decay["fit_quality_r2"].round(3),
            "period_s": freq["period_seconds"].round(1),
        }
    )
    print("\nSettling characteristics of the level loop:")
    print(characteristics)
    return characteristics


def quality_score(row) -> float:
    """A simple 0-100 score: slow settling, overshoot and oscillation cost points."""
    score = 100.0
    if pd.isna(row["t_settle_seconds"]) or row["t_settle_seconds"] > 300:
        score -= 20
    if row["overshoot_pct"] > 10:
        score -= 1.5 * row["overshoot_pct"]
    if row["oscillation_count"] > 3:
        score -= 5 * row["oscillation_count"]
    if abs(row["steady_state_error"]) > 1.0:
        score -= 10
    return max(0.0, score)


def example_control_quality_analysis(df):
    """Every KPI of every setpoint change in one table, and a score."""
    spe = SetpointChangeEvents(df, setpoint_uuid="SP_REACTOR_TEMP")
    quality = spe.control_quality_metrics(
        "PV_REACTOR_TEMP",
        settle_pct=0.02,
        hold="30s",
        lookahead="20min",
        rate_threshold=0.05,
    )
    quality["quality_score"] = quality.apply(quality_score, axis=1)

    print("\nControl quality of the reactor temperature loop:")
    numbers = ["t_settle_seconds", "overshoot_pct", "quality_score"]
    print(quality[["start", *numbers]].round(dict.fromkeys(numbers, 1)))
    poor = quality[quality["quality_score"] < 70]
    print(f"Setpoint changes needing attention: {len(poor)}")
    return quality


def example_tuning_comparison(df_before, df_after):
    """Mean KPIs of the same loop before and after retuning."""

    def kpis(df):
        spe = SetpointChangeEvents(df, setpoint_uuid="SP_001")
        q = spe.control_quality_metrics("PV_001", settle_pct=0.02, lookahead="15min")
        return [
            q["t_settle_seconds"].mean(),
            q["overshoot_pct"].mean(),
            q["oscillation_count"].mean(),
            q["steady_state_error"].abs().mean(),
        ]

    comparison = pd.DataFrame(
        {
            "metric": [
                "settling time (s)",
                "overshoot (%)",
                "oscillations",
                "|steady-state error|",
            ],
            "before": kpis(df_before),
            "after": kpis(df_after),
        }
    )
    comparison["improvement_pct"] = (
        (comparison["before"] - comparison["after"]) / comparison["before"] * 100
    )
    print("\nTuning improvement:")
    print(comparison.round(2))
    return comparison


def example_batch_analysis(df, setpoint_uuids):
    """One summary row per loop, worst settling first."""
    rows = []
    for sp_uuid in setpoint_uuids:
        pv_uuid = sp_uuid.replace("SP_", "PV_")
        quality = SetpointChangeEvents(
            df, setpoint_uuid=sp_uuid
        ).control_quality_metrics(pv_uuid, settle_pct=0.02, lookahead="15min")
        if quality.empty:
            continue
        rows.append(
            {
                "loop": sp_uuid,
                "changes": len(quality),
                "avg_settle_s": quality["t_settle_seconds"].mean(),
                "max_overshoot_pct": quality["overshoot_pct"].max(),
                "avg_rise_s": quality["rise_time_seconds"].mean(),
            }
        )
    report = pd.DataFrame(rows).sort_values("avg_settle_s", ascending=False)
    print("\nControl loop report:")
    print(report.round(3).to_string(index=False))

    problematic = report[
        (report["avg_settle_s"] > 300) | (report["max_overshoot_pct"] > 20)
    ]
    print(f"Problematic loops: {', '.join(problematic['loop']) or 'none'}")
    return report


if __name__ == "__main__":
    loops = {
        "TEMP_001": dict(zeta=1.0, omega=0.05, noise=0.05),  # critically damped
        "PRESSURE_001": dict(zeta=1.5, omega=0.03, noise=0.02),  # slow, overdamped
        # noise-free, so the oscillation counts show the response, not the noise
        "FLOW_001": dict(zeta=0.25, omega=0.1, noise=0.0),  # underdamped, overshoots
        "LEVEL_001": dict(zeta=0.4, omega=0.05, noise=0.0),
        "REACTOR_TEMP": dict(zeta=0.6, omega=0.04, noise=0.0),
    }
    df = pd.concat(
        [make_loop(name, seed=i, **p) for i, (name, p) in enumerate(loops.items())],
        ignore_index=True,
    )

    example_basic_enhancements(df)
    example_derivative_settling(df)
    example_enhanced_overshoot(df)
    example_settling_characteristics(df)
    example_control_quality_analysis(df)

    before = make_loop("001", zeta=0.2, omega=0.08, noise=0.0, seed=10)
    after = make_loop("001", zeta=0.8, omega=0.08, noise=0.0, seed=11)
    example_tuning_comparison(before, after)

    example_batch_analysis(df, [f"SP_{name}" for name in loops])
