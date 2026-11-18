"""Cycle time profiling of one PLC without any signal mapping.

The same call works on the raw frame of ``AzureBlobParquetLoader`` and the
metadata table of ``MetadataJsonLoader``; here a simulated PLC stands in.
Dashboard: ``streamlit run examples/cycle_profiling/dashboard.py``.
"""

import pandas as pd

import ts_shape
from ts_shape.cycle_profile import profile_cycles

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)

frame, _ = ts_shape.make_plc_cycles(n_cycles=300, seed=7)
result = profile_cycles(frame)
s = result.summary
print(
    f"anchor {s['anchor_event']}, {s['cycle_count']} cycles: "
    f"{s['long_count']} long, {s['short_count']} short, {s['pause_count']} pauses, {s['reason']}"
)
for v in s["variants"]:
    print(
        f"variant {v['variant']}: median {v['median_seconds']:.1f} s, target (P10) "
        f"{v['target_seconds']:.1f} s, gain {v['throughput_gain_pct']:.1f} %"
    )

register = result.tables["contribution_register"]
print("\nWhat to optimize first (variant A):")
print(
    register[register["variant"] == "A"]
    .sort_values("potential_rank")
    .head(3)[["segment", "median_seconds", "potential_seconds"]]
    .to_string(index=False)
)
print("\nWhat makes the cycle unstable (variant A):")
print(
    register[register["variant"] == "A"]
    .nlargest(3, "variance_contribution_factor")[
        ["segment", "variance_contribution_factor", "spread_seconds"]
    ]
    .to_string(index=False)
)
