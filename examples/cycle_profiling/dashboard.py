"""Cycle time profile dashboard (Streamlit): the numbers of one PLC profile.

Run::

    pip install streamlit
    streamlit run examples/cycle_profiling/dashboard.py

Sources (sidebar):

- **Result folder**: a folder written by ``python -m ts_shape.cycle_profile``
  (or ``CycleProfileResult.save``).
- **Azure Blob**: raw parquet files read with ``AzureBlobParquetLoader`` as it
  is, plus an optional metadata JSON (``MetadataJsonLoader``) for labels.
  Credentials from ``AZURE_STORAGE_SAS_URL`` or
  ``AZURE_STORAGE_CONNECTION_STRING``.
- **Simulated PLC**: ``ts_shape.make_plc_cycles`` to try the page.

Everything shown is a value of :func:`ts_shape.cycle_profile.profile_cycles`.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import ts_shape
from ts_shape.cycle_profile import (
    CycleProfileResult,
    load_azure_timeseries,
    profile_cycles,
)

st.set_page_config(page_title="Cycle time profile", layout="wide")


@st.cache_data(show_spinner="Profiling the simulated PLC")
def _simulated(n_cycles: int, seed: int) -> CycleProfileResult:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=n_cycles, seed=seed)
    return profile_cycles(frame)


@st.cache_data(show_spinner="Loading from Azure and profiling")
def _azure(
    container: str, prefix: str, start: str, end: str, metadata_json: str
) -> CycleProfileResult:
    from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader
    from ts_shape.loader.timeseries.azure_blob_loader import AzureBlobParquetLoader

    sas_url = os.environ.get("AZURE_STORAGE_SAS_URL")
    if sas_url:
        loader = AzureBlobParquetLoader(sas_url=sas_url, prefix=prefix)
    else:
        loader = AzureBlobParquetLoader(
            container_name=container,
            connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
            prefix=prefix,
        )
    metadata = (
        MetadataJsonLoader.from_file(metadata_json, strict=False).to_df()
        if metadata_json
        else None
    )
    uuids = None if metadata is None else list(metadata.index)
    raw = load_azure_timeseries(loader, start, end, uuids)
    return profile_cycles(raw, metadata)


@st.cache_data
def _folder(path: str) -> CycleProfileResult:
    return CycleProfileResult.load(path)


def _f(value: object, digits: int = 1, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}{unit}"
    return str(value)


# ---------------------------------------------------------------- source
with st.sidebar:
    st.header("Source")
    source = st.radio("Data", ["Result folder", "Azure Blob", "Simulated PLC"], index=2)
    result: CycleProfileResult | None = None
    if source == "Result folder":
        path = st.text_input("Folder", "cycle_profile")
        if path and os.path.isfile(os.path.join(path, "summary.json")):
            result = _folder(path)
        else:
            st.write("No summary.json in this folder.")
    elif source == "Azure Blob":
        container = st.text_input("Container", os.environ.get("AZURE_CONTAINER", ""))
        prefix = st.text_input("Prefix", "")
        start = st.text_input("Start (UTC)", "2026-11-17T00:00:00Z")
        end = st.text_input("End (UTC)", "2026-11-17T23:59:59Z")
        metadata_json = st.text_input("Metadata JSON (optional)", "")
        if st.button("Load and profile"):
            st.session_state["azure"] = (container, prefix, start, end, metadata_json)
        if "azure" in st.session_state:
            result = _azure(*st.session_state["azure"])
    else:
        n_cycles = st.number_input("Cycles", 60, 3000, 600, step=60)
        seed = st.number_input("Seed", 0, 10_000, 7)
        result = _simulated(int(n_cycles), int(seed))

if result is None:
    st.stop()

s = result.summary
t = result.tables

# ---------------------------------------------------------------- summary
st.title("Cycle time profile")
st.caption(
    f"{s['signal_count']} signals · {s['row_count']:,} rows · "
    f"{s['data_start']} – {s['data_end']} · anchor {s['anchor_event']} · "
    f"variant signal {s['variant_uuid'] or 'none'} · "
    f"time resolution {_f(s['time_resolution_seconds'], 1, ' s')}"
)

c = st.columns(6)
c[0].metric("Cycles", s["cycle_count"])
c[1].metric("Normal", s["non_pause_cycle_count"] - s["long_count"] - s["short_count"])
c[2].metric("Long", s["long_count"])
c[3].metric("Short", s["short_count"])
c[4].metric(
    "Pauses",
    s["pause_count"],
    _f((s["pause_seconds"] or 0) / 3600, 2, " h"),
    delta_color="off",
)
c[5].metric(
    "Readiness",
    "evaluable" if s["evaluable"] else "not evaluable",
    s["reason"],
    delta_color="off",
)

for v in s["variants"]:
    st.subheader(f"Variant {v['variant']}")
    c = st.columns(7)
    c[0].metric("Median", _f(v["median_seconds"], 1, " s"))
    c[1].metric("Target (P10)", _f(v["target_seconds"], 1, " s"))
    c[2].metric("P90", _f(v["p90_seconds"], 1, " s"))
    c[3].metric("Gain at target", _f(v["throughput_gain_pct"], 2, " %"))
    c[4].metric(
        "Cycles/h median → target",
        f"{_f(v['cycles_per_hour_median'], 2)} → {_f(v['cycles_per_hour_target'], 2)}",
    )
    c[5].metric("Short limit", _f(v["lower_limit_seconds"], 1, " s"))
    c[6].metric("Long limit", _f(v["upper_limit_seconds"], 1, " s"))
    c = st.columns(2)
    c[0].metric(
        "Largest potential",
        _f(v["top_potential_seconds"], 1, " s"),
        v["top_potential_segment"],
        delta_color="off",
    )
    c[1].metric(
        "Largest variance share",
        _f(v["top_variance_share_pct"], 1, " %"),
        v["top_variance_segment"],
        delta_color="off",
    )

# ---------------------------------------------------------------- tables
tabs = st.tabs(
    [
        "Contribution register",
        "Variance",
        "Influencers",
        "Unusual cycles",
        "Pauses",
        "Readiness",
        "Standard view",
        "Anchor candidates",
        "Cycle records",
        "Signals",
    ]
)
register = t["contribution_register"]
with tabs[0]:
    st.dataframe(
        register.sort_values(["variant", "potential_rank"])[
            [
                "variant",
                "potential_rank",
                "segment",
                "median_seconds",
                "p10_seconds",
                "p90_seconds",
                "potential_seconds",
                "cycle_count",
                "deviation_cycle_count",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
with tabs[1]:
    st.dataframe(
        register.sort_values(
            ["variant", "variance_contribution_factor"], ascending=[True, False]
        )[
            [
                "variant",
                "segment",
                "variance_contribution_factor",
                "spread_seconds",
                "mad_seconds",
                "std_seconds",
                "skewness",
                "bimodality_coefficient",
                "is_bimodal",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
with tabs[2]:
    st.dataframe(t["influencers"], hide_index=True, width="stretch")
with tabs[3]:
    st.dataframe(
        t["unusual_cycles"][
            [
                "systime",
                "variant",
                "cycle_class",
                "duration_seconds",
                "median_seconds",
                "deviation_seconds",
                "dominant_segment",
                "dominant_excess_seconds",
                "sequence_deviation",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
with tabs[4]:
    st.dataframe(t["pauses"], hide_index=True, width="stretch")
with tabs[5]:
    st.dataframe(t["evaluation_window"], hide_index=True, width="stretch")
    st.dataframe(t["cycle_time_statistics"], hide_index=True, width="stretch")
with tabs[6]:
    st.dataframe(t["standard_view"], hide_index=True, width="stretch")
with tabs[7]:
    st.dataframe(t["anchors"], hide_index=True, width="stretch")
with tabs[8]:
    st.dataframe(t["cycle_records"], hide_index=True, width="stretch")
with tabs[9]:
    st.dataframe(t["signals"], hide_index=True, width="stretch")
