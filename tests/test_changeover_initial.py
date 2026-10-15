"""Test the include_initial option on ChangeoverEvents.detect_changeover."""

import pandas as pd

from ts_shape.events.production.changeover import ChangeoverEvents


def _frame():
    rows = [
        ("2024-01-01 08:00:00", "PART_A"),
        ("2024-01-01 09:00:00", "PART_A"),
        ("2024-01-01 10:00:00", "PART_B"),
        ("2024-01-01 11:00:00", "PART_B"),
    ]
    return pd.DataFrame(
        [
            {
                "uuid": "prod",
                "systime": pd.Timestamp(t),
                "value_string": v,
                "is_delta": True,
            }
            for t, v in rows
        ]
    )


def test_include_initial_default_emits_first_reading():
    co = ChangeoverEvents(_frame())
    changes = co.detect_changeover("prod")
    # Default: the initial PART_A reading plus the real A->B change = 2 events.
    assert list(changes["new_value"]) == ["PART_A", "PART_B"]


def test_include_initial_false_only_real_changes():
    co = ChangeoverEvents(_frame())
    changes = co.detect_changeover("prod", include_initial=False)
    # Only the real changeover to PART_B.
    assert list(changes["new_value"]) == ["PART_B"]


def test_changeover_quality_metrics_docstring_matches_output():
    """The docstring must not promise duration patterns it does not return."""
    co = ChangeoverEvents(_frame())
    out = co.changeover_quality_metrics("prod")
    doc = ChangeoverEvents.changeover_quality_metrics.__doc__ or ""
    assert "duration patterns" not in doc
    assert not any("duration" in c for c in out.columns)
    for col in ("product", "changeover_count"):
        assert col in out.columns and col in doc
