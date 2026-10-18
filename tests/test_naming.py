"""Naming lint: every detector signature follows docs/naming.md.

* no retired parameter names (they live on only as deprecated keyword
  aliases, which are not part of the signature) -- in the detectors and in
  every public class of ``ts_shape.transform`` and ``ts_shape.features``,
  and the loaders' time ranges (``start`` / ``end``);
* optional arguments of detectors are keyword-only;
* ``event_uuid`` defaults look like ``"<pack>:<name>"`` and are unique;
* no public signature anywhere in ts_shape uses a name from the "Not"
  column of the Parameters table in docs/naming.md (parsed from the page),
  except the documented exceptions in :data:`NOT_COLUMN_EXCEPTIONS`;
* detector output columns follow the "Output columns" rules: a legacy
  spelling (``window_start``, ``n_samples``, ``pct_<x>``, ``<x>_ms``,
  ``<x>_s``, ...) appears only as an explicitly listed deprecated twin, next
  to its canonical column.
"""

from __future__ import annotations

import fnmatch
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import ts_shape
import ts_shape.features
import ts_shape.loader
import ts_shape.transform
from ts_shape.eventlog.taxonomy import CLASS_ALIASES, REGISTRY

#: retired name -> canonical name (docs/naming.md)
RETIRED = {
    "column_name": "time_column",
    "timestamp_column": "time_column",
    "systime_col": "time_column",
    "time_col": "time_column",
    "value_col": "value_column",
    "uuid_col": "uuid_column",
    "sensor_uuids": "signal_uuids",
    "uuid_list": "signal_uuids",
    "run_uuid": "run_state_uuid",
    "freq": "window",
    "period": "window",
    "resample": "window",
    "z_threshold": "n_sigma",
    "threshold_sigma": "n_sigma",
    "low": "lower_limit",
    "high": "upper_limit",
    "lower": "lower_limit",
    "upper": "upper_limit",
    "lower_bound": "lower_limit",
    "upper_bound": "upper_limit",
    "low_limit": "lower_limit",
    "high_limit": "upper_limit",
    "minor_threshold": "warning_threshold",
    "moderate_threshold": "critical_threshold",
    "include_first": "include_initial",
    "count_initial_active": "include_initial",
    "significant_only": "only_significant",
    "on_crossing_only": "only_crossings",
    "counter_column": "value_column_counter",
    "energy_column": "value_column_energy",
    "state_column": "value_column_state",
    "value_cols": "value_columns",
    "cycle_uuid_col": "cycle_uuid_column",
    "group_col": "group_column",
    "df": "dataframe",
    "state_uuid": "run_state_uuid",
    "expected_freq": "expected_interval",
    "uuid_a": "signal_uuid_a",
    "uuid_b": "signal_uuid_b",
    "period1": "period_a",
    "period2": "period_b",
}

#: Retired in transform/features (the same names as the detectors).
RETIRED_TRANSFORM_FEATURES = dict(RETIRED)

#: Retired time-range names in the loaders; a range is ``start`` / ``end``.
RETIRED_LOADER = {
    "start_timestamp": "start",
    "end_timestamp": "end",
    "start_time": "start",
    "end_time": "end",
}

CLASSES = sorted(
    {c for c, _ in REGISTRY if c in dir(ts_shape) and c not in CLASS_ALIASES}
)


def _public_classes(package):
    """Public classes defined in ``package`` and its submodules."""
    found = {}
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and not name.startswith("_")
            ):
                found[f"{module.__name__}.{name}"] = obj
    return found


TRANSFORM_FEATURE_CLASSES = {
    **_public_classes(ts_shape.transform),
    **_public_classes(ts_shape.features),
}


def _callables(cls):
    yield "__init__", cls.__init__
    for name, member in vars(cls).items():
        if name.startswith("_") or not callable(getattr(cls, name, None)):
            continue
        if type(member).__name__ == "_Forward":  # deprecated method alias
            continue
        yield name, getattr(cls, name)


def _params(fn):
    try:
        return list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return []


@pytest.mark.parametrize("cls_name", CLASSES)
def test_no_retired_parameter_names(cls_name):
    cls = getattr(ts_shape, cls_name)
    bad = [
        f"{cls_name}.{name}({p.name}) -> {RETIRED[p.name]}"
        for name, fn in _callables(cls)
        for p in _params(fn)
        if p.name in RETIRED
    ]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("qualname", sorted(TRANSFORM_FEATURE_CLASSES))
def test_no_retired_parameter_names_in_transform_and_features(qualname):
    cls = TRANSFORM_FEATURE_CLASSES[qualname]
    bad = [
        f"{qualname}.{name}({p.name}) -> {RETIRED_TRANSFORM_FEATURES[p.name]}"
        for name, fn in _callables(cls)
        for p in _params(fn)
        if p.name in RETIRED_TRANSFORM_FEATURES
    ]
    assert not bad, "\n".join(bad)


LOADER_CLASSES = _public_classes(ts_shape.loader)


@pytest.mark.parametrize("qualname", sorted(LOADER_CLASSES))
def test_loader_time_ranges_are_start_and_end(qualname):
    cls = LOADER_CLASSES[qualname]
    bad = [
        f"{qualname}.{name}({p.name}) -> {RETIRED_LOADER[p.name]}"
        for name, fn in _callables(cls)
        for p in _params(fn)
        if p.name in RETIRED_LOADER
    ]
    assert not bad, "\n".join(bad)


def test_transform_and_features_classes_are_linted():
    names = {qualname.rsplit(".", 1)[1] for qualname in TRANSFORM_FEATURE_CLASSES}
    assert {"DateTimeFilter", "NumericStatistics", "CycleDataProcessor"} <= names


@pytest.mark.parametrize("cls_name", CLASSES)
def test_optional_arguments_are_keyword_only(cls_name):
    cls = getattr(ts_shape, cls_name)
    bad = []
    for name, fn in _callables(cls):
        for p in _params(fn):
            # The input frame is always the first, positional argument, even
            # where it is optional.
            if p.name == "dataframe":
                continue
            if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is not p.empty:
                bad.append(f"{cls_name}.{name}({p.name}=...)")
    assert not bad, "pass these by keyword (add `*`):\n" + "\n".join(bad)


_EVENT_UUID = re.compile(r"^[a-z]+:[a-z0-9_]+$")


def test_event_uuid_defaults_are_namespaced_and_unique():
    seen: dict[str, str] = {}
    problems = []
    classes = {getattr(ts_shape, c): c for c in reversed(CLASSES)}  # dedupe aliases
    for cls, cls_name in sorted(classes.items(), key=lambda kv: kv[1]):
        p = inspect.signature(cls).parameters.get("event_uuid")
        if p is None or p.default is p.empty or p.default is None:
            continue
        if not _EVENT_UUID.match(str(p.default)):
            problems.append(f"{cls_name}: {p.default!r} is not '<pack>:<name>'")
        if p.default in seen:
            problems.append(
                f"{cls_name} and {seen[p.default]} share the default {p.default!r}"
            )
        seen[p.default] = cls_name
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# docs/naming.md "Not" column, over every public signature
# ---------------------------------------------------------------------------

NAMING_MD = Path(__file__).resolve().parents[1] / "docs" / "naming.md"


def _not_column() -> dict[str, str]:
    """Retired parameter names (glob patterns) from the Parameters table.

    ``<role>`` / ``<what>`` placeholders become ``*``. Rows whose "Not" cell
    describes values rather than names ("an integer ``window``", "free-form
    defaults such as ...") are skipped.
    """
    text = NAMING_MD.read_text(encoding="utf-8")
    table = text.split("## Parameters", 1)[1].split("\n## ", 1)[0]
    found: dict[str, str] = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in ("Concept", "") or set(cells[0]) <= {"-"}:
            continue
        concept, canonical, retired = cells
        if re.sub(r"`[^`]*`|[,/ ]", "", retired):
            continue  # prose, not a list of names
        for token in re.findall(r"`([^`]+)`", retired):
            found[re.sub(r"<\w+>", "*", token)] = concept
    return found


NOT_COLUMN = _not_column()

#: ``(qualified callable, parameter)`` -> why it keeps a "Not" name.
NOT_COLUMN_EXCEPTIONS: dict[tuple[str, str], str] = {
    # Not a frame argument.
    ("ts_shape.pipeline:PipelineResult.__init__", "data"): "dataclass field",
    ("ts_shape.signals.spec:SignalSpec.from_dict", "data"): "a dict, not a frame",
    ("ts_shape.signals.spec:SignalSpecs.from_dict", "data"): "a dict, not a frame",
    # Not a value column of the input frame.
    (
        "ts_shape.context.value_mapping:ValueMapper.__init__",
        "mapping_value_column",
    ): "a column of the mapping table, paired with mapping_key_column",
    ("ts_shape.utils.signal_checks:remember_uuids", "column"): "internal helper",
    ("ts_shape.utils.signal_checks:remember_uuids", "uuids"): "internal helper",
    (
        "ts_shape.events.quality.sensor_drift:SensorDriftEvents.drift_trend",
        "metric",
    ): "the statistic to trend ('mean' / 'std'), not a column",
    (
        "ts_shape.features.pattern_recognition:PatternRecognition"
        ".compute_distance_profile",
        "metric",
    ): "a distance metric, not a column",
    # Sampling interval of generated or partitioned data, not a time bucket.
    ("ts_shape.datasets:make_timeseries", "freq"): "sampling interval (datasets)",
    ("ts_shape.datasets:make_id_signal", "freq"): "sampling interval (datasets)",
    ("ts_shape.signals.primitives:windows", "freq"): "deferred: rename to window",
    **{
        (
            "ts_shape.loader.timeseries.databricks_spark_parquet_loader"
            f":DatabricksSparkParquetLoader.{m}",
            p,
        ): "deferred: loader partition arguments"
        for m in ("build_paths", "load_by_time_range_and_uuids")
        for p in ("freq", "uuids")
    },
    # A signal's own uuid (a field), not "the signal a detector analyzes".
    ("ts_shape.signals.spec:SignalSpec.__init__", "uuid"): "dataclass field",
    ("ts_shape.eventlog.objects:ObjectSpec.__init__", "uuid"): "dataclass field",
    ("ts_shape.signals.infer:infer_spec", "uuid"): "deferred: rename to signal_uuid",
    ("ts_shape.datasets:make_id_signal", "uuid"): "deferred: rename to signal_uuid",
    ("ts_shape.datasets:make_timeseries", "uuids"): "deferred: signal_uuids",
    (
        "ts_shape.loader.metadata.metadata_json_loader:MetadataJsonLoader.get_by_uuid",
        "uuid",
    ): "deferred: loader lookup argument",
    (
        "ts_shape.loader.metadata.metadata_json_loader"
        ":MetadataJsonLoader.filter_by_uuid",
        "uuids",
    ): "deferred: loader lookup argument",
    # Loader uuid lists.
    **{
        (f"ts_shape.loader.timeseries.{cls}.{m}", "uuid_list"): (
            "deferred: loader uuid lists"
        )
        for cls in (
            "azure_blob_loader:AzureBlobParquetLoader",
            "databricks_unity_parquet_loader:DatabricksUnityParquetLoader",
        )
        for m in (
            "load_files_by_time_range_and_uuids",
            "stream_files_by_time_range_and_uuids",
        )
    },
    (
        "ts_shape.loader.timeseries.parquet_loader:ParquetLoader.load_by_uuid_list",
        "uuid_list",
    ): "deferred: loader uuid lists",
    (
        "ts_shape.loader.timeseries.parquet_loader"
        ":ParquetLoader.load_files_by_time_range_and_uuids",
        "uuid_list",
    ): "deferred: loader uuid lists",
    (
        "ts_shape.loader.timeseries.databricks_spark_parquet_loader"
        ":DatabricksSparkParquetLoader.fetch_data_as_dataframe",
        "uuids",
    ): "deferred: loader uuid lists",
    (
        "ts_shape.loader.timeseries.s3proxy_parquet_loader:S3ProxyDataAccess.__init__",
        "uuids",
    ): "deferred: loader uuid lists",
    (
        "ts_shape.loader.timeseries.timescale_loader:TimescaleDBDataAccess.__init__",
        "uuids",
    ): "deferred: loader uuid lists",
    (
        "ts_shape.loader.combine.integrator:DataIntegratorHybrid.combine_data",
        "uuids",
    ): "deferred: loader uuid lists",
    (
        "ts_shape.transform.harmonization:DataHarmonizer.merge_multi_signals",
        "uuids",
    ): "deferred: rename to signal_uuids",
}


def _all_public_callables():
    for info in pkgutil.walk_packages(ts_shape.__path__, "ts_shape."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # loader backends with optional dependencies
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != info.name:
                continue
            if inspect.isfunction(obj):
                yield f"{info.name}:{name}", obj
            elif inspect.isclass(obj):
                if getattr(obj, "__deprecated_alias_of__", None) is not None:
                    continue  # deprecated class alias
                for member, fn in _callables(obj):
                    yield f"{info.name}:{name}.{member}", fn


PUBLIC_CALLABLES = dict(_all_public_callables())


def _retired_hits():
    for qualname, fn in PUBLIC_CALLABLES.items():
        for p in _params(fn):
            if p.name in ("self", "cls"):
                continue
            for pattern, concept in NOT_COLUMN.items():
                if fnmatch.fnmatchcase(p.name, pattern):
                    yield qualname, p.name, concept


def test_not_column_is_parsed():
    assert {"df", "state_uuid", "*_value_column", "uuids", "freq"} <= set(NOT_COLUMN)
    assert "window" not in NOT_COLUMN  # "an integer `window`" is prose
    assert len(PUBLIC_CALLABLES) > 500


def test_public_signatures_avoid_the_not_column():
    bad = [
        f"{q}({p}): naming.md says not this for {concept!r}"
        for q, p, concept in _retired_hits()
        if (q, p) not in NOT_COLUMN_EXCEPTIONS
    ]
    assert not bad, "\n".join(bad)


def test_not_column_exceptions_are_current():
    hits = {(q, p) for q, p, _ in _retired_hits()}
    importable = {q.split(":")[0] for q in PUBLIC_CALLABLES}
    stale = [
        key
        for key in NOT_COLUMN_EXCEPTIONS
        if key not in hits and key[0].split(":")[0] in importable
    ]
    assert not stale, f"renamed or gone, drop the exception: {stale}"


# ---------------------------------------------------------------------------
# Output columns (docs/naming.md, "Output columns")
# ---------------------------------------------------------------------------

#: Legacy output column -> its canonical twin. A legacy spelling is allowed
#: only here, and only next to its twin (removed in the next major release).
DEPRECATED_TWINS = {
    "window_start": "start",
    "window_end": "end",
    "time": "systime",
    "n_samples": "sample_count",
    "mean_interval_ms": "mean_interval_seconds",
    "std_interval_ms": "std_interval_seconds",
    "max_gap_ms": "max_gap_seconds",
    "avg_cycle_time_s": "avg_cycle_time_seconds",
    "target_cycle_time_s": "target_cycle_time_seconds",
    "period1_value": "period_a_value",
    "period2_value": "period_b_value",
    "pct_GRR": "grr_pct",
    "pct_PV": "pv_pct",
    "pct_GRR_tolerance": "grr_tolerance_pct",
    "pct_above": "time_above_pct",
    "pct_contribution": "contribution_pct",
    "pct_of_available_time": "share_of_available_time_pct",
    "pct_of_best": "share_of_best_pct",
    "pct_of_total": "share_of_total_pct",
    "pct_of_window": "share_of_window_pct",
    "pct_saturated": "saturated_pct",
    "pct_time_at_high": "time_at_high_pct",
    "pct_time_at_low": "time_at_low_pct",
    "pct_time_disturbed": "time_disturbed_pct",
    "pct_time_saturated": "time_saturated_pct",
}

#: Legacy columns of one method (bare ratios, seconds without a unit,
#: upper-case indices) and their canonical twins.
METHOD_TWINS = {
    ("LineThroughputEvents", "throughput_oee"): {
        "availability": "availability_ratio",
        "performance": "performance_ratio",
        "oee_score": "oee_ratio",
    },
    # Seconds without a unit in the name, twinned by ``*_seconds``.
    ("MicroStopEvents", "micro_stop_frequency"): {
        "total_lost_time": "total_lost_time_seconds",
    },
    ("MicroStopEvents", "micro_stop_impact"): {
        "total_run_time": "total_run_time_seconds",
        "total_micro_stop_time": "total_micro_stop_time_seconds",
    },
    ("MicroStopEvents", "micro_stop_patterns"): {
        "avg_lost_time": "avg_lost_time_seconds",
    },
    ("OperatingStateEvents", "state_duration_statistics"): {
        "mean_duration": "mean_duration_seconds",
        "std_duration": "std_duration_seconds",
        "min_duration": "min_duration_seconds",
        "max_duration": "max_duration_seconds",
        "total_duration": "total_duration_seconds",
    },
    # Capability indices in lower case.
    ("ToleranceDeviationEvents", "capability_indices"): {"Cp": "cp", "Cpk": "cpk"},
}

#: Names that look legacy but are something else (a rate, a speed unit).
NOT_LEGACY = {"observed_ramp_per_s", "mean_speed_m_s"}

_LEGACY = re.compile(r"^(pct_.+|.+_ms|.+_s|n_samples|window_(start|end)|time)$")


def _output_problems(key, frame):
    cols = set(map(str, frame.columns))
    twins = {**DEPRECATED_TWINS, **METHOD_TWINS.get(key, {})}
    for col in cols:
        if col in twins:
            if twins[col] not in cols:
                yield f"{col!r} without its twin {twins[col]!r}"
        elif _LEGACY.match(col) and col not in NOT_LEGACY:
            yield f"{col!r} is a legacy spelling (see docs/naming.md)"


@pytest.mark.parametrize("key", sorted(REGISTRY), ids=lambda k: ".".join(k))
def test_output_columns_follow_naming(key):
    from invariants import harness

    status, plain, empty = harness.sample_results(*key)
    if status != "ok":
        pytest.skip(status)
    problems = [
        f"{name}: {p}"
        for name, result in (("result", plain), ("empty result", empty))
        for frame in harness._frames(result)
        for p in _output_problems(key, frame)
    ]
    assert not problems, "\n".join(problems)


def test_deprecated_twins_sit_next_to_their_canonical_column():
    from invariants import harness

    for key, twins in {
        ("SamplingResolutionEvents", "sampling_jitter"): {
            "mean_interval_ms": "mean_interval_seconds"
        },
        ("DowntimeEvents", "downtime_by_reason"): {
            "pct_of_total": "share_of_total_pct"
        },
        **METHOD_TWINS,
    }.items():
        _, plain, _ = harness.sample_results(*key)
        cols = list(plain.columns)
        for old, new in twins.items():
            assert cols.index(new) == cols.index(old) + 1, (key, old, cols)
