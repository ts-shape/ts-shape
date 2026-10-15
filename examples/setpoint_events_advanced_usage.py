"""
Advanced usage examples for SetpointChangeEvents enhancements.

This file demonstrates the new features added to the SetpointChangeEvents class
including noise filtering, percentage-based tolerance, derivative-based settling,
comprehensive overshoot metrics, and control quality analysis.
"""

import pandas as pd

from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents

# Assuming you have a DataFrame 'df' with your time-series data
# with columns: uuid, sequence_number, systime, plctime, is_delta,
# value_integer, value_string, value_double, value_bool, value_bytes


def example_basic_enhancements(df):
    """Examples of basic enhancements to existing methods."""

    # Initialize
    spe = SetpointChangeEvents(df, setpoint_uuid="SP_TEMP_001")

    # 1. Noise filtering in step detection
    # Original behavior - may detect false steps from noise
    steps_noisy = spe.detect_setpoint_steps(min_delta=0.5)

    # With noise filtering - cleaner detection
    steps_clean = spe.detect_setpoint_steps(
        min_delta=0.5,
        filter_noise=True,
        noise_threshold=0.1,  # Ignore changes smaller than 0.1
    )

    # 2. Percentage-based tolerance for settling
    # Original - absolute tolerance (same for all step sizes)
    settle_abs = spe.time_to_settle("PV_TEMP_001", tol=2.0, hold="30s")

    # New - percentage-based tolerance (adapts to step size)
    settle_pct = spe.time_to_settle(
        "PV_TEMP_001",
        settle_pct=0.02,  # 2% of step magnitude
        hold="30s",
        lookahead="15m",
    )

    print(
        f"Steps without / with noise filtering: {len(steps_noisy)} / {len(steps_clean)}"
    )
    print(
        f"Settling rows (absolute / percentage tolerance): {len(settle_abs)} / {len(settle_pct)}"
    )
    return steps_clean, settle_pct


def example_derivative_settling(df):
    """Example of derivative-based settling detection."""

    spe = SetpointChangeEvents(df, setpoint_uuid="SP_PRESSURE_001")

    # Detect settling based on rate of change
    # Useful for processes with slow asymptotic approach
    settle_deriv = spe.time_to_settle_derivative(
        "PV_PRESSURE_001",
        rate_threshold=0.05,  # Consider settled when |dv/dt| < 0.05
        lookahead="20m",
        hold="1m",  # Rate must stay low for 1 minute
    )

    # Compare with error-based settling
    settle_error = spe.time_to_settle(
        "PV_PRESSURE_001", settle_pct=0.01, lookahead="20m"
    )

    print("Derivative-based vs Error-based settling:")
    comparison = pd.DataFrame(
        {
            "time": settle_deriv["start"],
            "deriv_settle_time": settle_deriv["t_settle_seconds"],
            "error_settle_time": settle_error["t_settle_seconds"],
            "deriv_settled": settle_deriv["settled"],
            "error_settled": settle_error["settled"],
        }
    )
    print(comparison)

    return settle_deriv, settle_error


def example_enhanced_overshoot(df):
    """Example of enhanced overshoot metrics."""

    spe = SetpointChangeEvents(df, setpoint_uuid="SP_FLOW_001")

    # Get comprehensive transient response metrics
    overshoot = spe.overshoot_metrics("PV_FLOW_001", window="10m")

    print("Transient Response Metrics:")
    print(f"Overshoot: {overshoot['overshoot_pct'].iloc[0]:.1%}")
    print(f"Undershoot: {overshoot['undershoot_pct'].iloc[0]:.1%}")
    print(f"Oscillations: {overshoot['oscillation_count'].iloc[0]}")
    print(
        f"Avg Oscillation Amplitude: {overshoot['oscillation_amplitude'].iloc[0]:.2f}"
    )

    # Identify problematic responses
    excessive_overshoot = overshoot[overshoot["overshoot_pct"] > 0.15]  # >15%
    oscillatory = overshoot[overshoot["oscillation_count"] > 5]

    return overshoot, excessive_overshoot, oscillatory


def example_settling_characteristics(df):
    """Example of detailed settling characteristic analysis."""

    spe = SetpointChangeEvents(df, setpoint_uuid="SP_LEVEL_001")

    # 1. Rise Time (10-90%)
    rise = spe.rise_time(
        "PV_LEVEL_001", start_pct=0.1, end_pct=0.9, lookahead="15m"  # 10%  # 90%
    )

    # 2. Decay Rate (exponential decay constant)
    decay = spe.decay_rate("PV_LEVEL_001", lookahead="15m", min_points=10)

    # 3. Oscillation Frequency
    freq = spe.oscillation_frequency("PV_LEVEL_001", window="15m", min_oscillations=2)

    # Combine results
    characteristics = pd.DataFrame(
        {
            "time": rise["start"],
            "rise_time_sec": rise["rise_time_seconds"],
            "decay_lambda": decay["decay_rate_lambda"],
            "decay_fit_r2": decay["fit_quality_r2"],
            "osc_freq_hz": freq["oscillation_freq_hz"],
            "osc_period_sec": freq["period_seconds"],
        }
    )

    print("Settling Characteristics:")
    print(characteristics)

    return characteristics


def example_control_quality_analysis(df):
    """Example of comprehensive control quality metrics."""

    spe = SetpointChangeEvents(df, setpoint_uuid="SP_REACTOR_TEMP")

    # Get all quality metrics in one call
    quality = spe.control_quality_metrics(
        "PV_REACTOR_TEMP",
        tol=2.0,  # Absolute tolerance (fallback)
        settle_pct=0.02,  # 2% tolerance (takes priority)
        hold="30s",  # Must hold for 30 seconds
        lookahead="20m",  # Analyze 20 minutes after change
        rate_threshold=0.05,  # For derivative-based settling
    )

    print("Control Quality Metrics Summary:")
    print(quality)

    # Quality scoring example
    def calculate_quality_score(row):
        """Simple quality scoring function."""
        score = 100.0

        # Penalize slow settling
        if row["t_settle_seconds"] and row["t_settle_seconds"] > 300:
            score -= 20

        # Penalize overshoot
        if row["overshoot_pct"] and row["overshoot_pct"] > 0.1:
            score -= 15 * row["overshoot_pct"] * 100

        # Penalize oscillations
        if row["oscillation_count"] and row["oscillation_count"] > 3:
            score -= 5 * row["oscillation_count"]

        # Penalize steady-state error
        if row["steady_state_error"] and row["steady_state_error"] > 1.0:
            score -= 10

        return max(0, score)

    quality["quality_score"] = quality.apply(calculate_quality_score, axis=1)

    print("\nQuality Scores:")
    print(quality[["start", "quality_score"]].sort_values("quality_score"))

    # Identify loops needing attention
    poor_quality = quality[quality["quality_score"] < 70]
    print(f"\nLoops needing attention: {len(poor_quality)}")

    return quality


def example_tuning_comparison(df_before, df_after):
    """Compare control quality before and after tuning."""

    # Before tuning
    spe_before = SetpointChangeEvents(df_before, setpoint_uuid="SP_001")
    quality_before = spe_before.control_quality_metrics(
        "PV_001", settle_pct=0.02, lookahead="15m"
    )

    # After tuning
    spe_after = SetpointChangeEvents(df_after, setpoint_uuid="SP_001")
    quality_after = spe_after.control_quality_metrics(
        "PV_001", settle_pct=0.02, lookahead="15m"
    )

    # Calculate improvements
    comparison = pd.DataFrame(
        {
            "metric": [
                "Settling Time (s)",
                "Overshoot (%)",
                "Undershoot (%)",
                "Oscillation Count",
                "Steady State Error",
            ],
            "before": [
                quality_before["t_settle_seconds"].mean(),
                quality_before["overshoot_pct"].mean() * 100,
                quality_before["undershoot_pct"].mean() * 100,
                quality_before["oscillation_count"].mean(),
                quality_before["steady_state_error"].mean(),
            ],
            "after": [
                quality_after["t_settle_seconds"].mean(),
                quality_after["overshoot_pct"].mean() * 100,
                quality_after["undershoot_pct"].mean() * 100,
                quality_after["oscillation_count"].mean(),
                quality_after["steady_state_error"].mean(),
            ],
        }
    )

    comparison["improvement_%"] = (
        (comparison["before"] - comparison["after"]) / comparison["before"] * 100
    )

    print("Tuning Improvement Analysis:")
    print(comparison)

    return comparison


def example_batch_analysis(df, setpoint_uuid_list):
    """Analyze multiple control loops in batch."""

    results = []

    for sp_uuid in setpoint_uuid_list:
        pv_uuid = sp_uuid.replace("SP_", "PV_")

        spe = SetpointChangeEvents(df, setpoint_uuid=sp_uuid)

        # Get quality metrics
        quality = spe.control_quality_metrics(pv_uuid, settle_pct=0.02, lookahead="15m")

        if not quality.empty:
            # Aggregate metrics per loop
            summary = {
                "loop": sp_uuid,
                "num_changes": len(quality),
                "avg_settle_time": quality["t_settle_seconds"].mean(),
                "max_overshoot": quality["overshoot_pct"].max(),
                "avg_oscillations": quality["oscillation_count"].mean(),
                "avg_steady_error": quality["steady_state_error"].mean(),
            }
            results.append(summary)

    # Create summary report
    report = pd.DataFrame(results)
    report = report.sort_values("avg_settle_time", ascending=False)

    print("Control Loop Performance Report:")
    print(report)

    # Flag problematic loops
    problematic = report[
        (report["avg_settle_time"] > 300)
        | (report["max_overshoot"] > 0.2)
        | (report["avg_oscillations"] > 5)
    ]

    print(f"\nProblematic Loops: {len(problematic)}")
    print(problematic)

    return report


# Example usage patterns
if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_timeseries_data.csv')

    # Run examples
    # example_basic_enhancements(df)
    # example_derivative_settling(df)
    # example_enhanced_overshoot(df)
    # example_settling_characteristics(df)
    # example_control_quality_analysis(df)

    print("See function definitions above for usage examples.")
    print("Uncomment the example calls in __main__ to run with your data.")
