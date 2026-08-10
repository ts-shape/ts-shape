"""Counter-reset handling for ShiftHandoverReport.generate_report."""

import pandas as pd

from ts_shape.events.production.shift_handover import ShiftHandoverReport


def _make_reset_handover_df():
    """A single shift_1 (06:00-14:00) whose production counter resets mid-shift.

    Production: 100, 200, 300, [reset] 50, 150.
    Naive production: max(0, 150 - 100) = 50.
    Reset-aware production: 100 + 100 + 50 + 100 = 350, with 1 reset.
    OK and NOK counters stay monotonic.
    """
    times = pd.to_datetime(
        [
            "2024-01-01 07:00:00",
            "2024-01-01 08:00:00",
            "2024-01-01 09:00:00",
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
        ]
    )
    prod = [100, 200, 300, 50, 150]
    ok = [90, 180, 270, 360, 450]
    nok = [2, 4, 6, 8, 10]

    rows = []
    for t, p, o, n in zip(times, prod, ok, nok):
        rows.append({"systime": t, "uuid": "prod_counter", "value_integer": p})
        rows.append({"systime": t, "uuid": "ok_counter", "value_integer": o})
        rows.append({"systime": t, "uuid": "nok_counter", "value_integer": n})
        rows.append({"systime": t, "uuid": "machine_state", "value_string": "Running"})
    return pd.DataFrame(rows)


def test_generate_report_under_reports_without_reset_handling():
    report = ShiftHandoverReport(_make_reset_handover_df())
    result = report.generate_report(
        counter_uuid="prod_counter",
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="nok_counter",
        state_uuid="machine_state",
    )
    assert list(result.columns) == ShiftHandoverReport.OUTPUT_COLUMNS
    assert "resets" not in result.columns

    row = result[result["shift"] == "shift_1"].iloc[0]
    assert row["production"] == 50  # under-reports: real production is 350


def test_generate_report_handles_resets():
    report = ShiftHandoverReport(_make_reset_handover_df())
    result = report.generate_report(
        counter_uuid="prod_counter",
        ok_counter_uuid="ok_counter",
        nok_counter_uuid="nok_counter",
        state_uuid="machine_state",
        handle_resets=True,
    )
    assert "resets" in result.columns

    row = result[result["shift"] == "shift_1"].iloc[0]
    assert row["production"] == 350
    assert row["ok_parts"] == 360  # monotonic ok counter unaffected
    assert row["nok_parts"] == 8
    assert row["resets"] == 1
