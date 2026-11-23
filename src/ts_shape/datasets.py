"""Synthetic timeseries generators for trying out ts-shape.

These helpers produce DataFrames in the standard ts-shape schema
(``systime``, ``uuid``, ``value_*``, ``is_delta``) so any detector,
transform, or statistic can be exercised without real data or loaders::

    import ts_shape
    df = ts_shape.make_timeseries(["sensor:temp"], n_outliers=4)
    ts_shape.OutlierDetectionEvents(df, value_column="value_double") \\
        .detect_outliers_zscore()
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.loader._utils import empty_signal_frame
from ts_shape.utils.deprecation import caller_stacklevel

_VALUE_COLUMNS = ("value_double", "value_integer", "value_bool")


def _unused_columns(n: int) -> dict[str, pd.Series]:
    """The value columns a generator leaves empty, in nullable dtypes."""
    return {
        "value_bool": pd.Series(pd.array([pd.NA] * n, dtype="boolean")),
        "value_integer": pd.Series(pd.array([pd.NA] * n, dtype="Int64")),
        "value_double": pd.Series(np.full(n, np.nan)),
        "value_string": pd.Series([pd.NA] * n, dtype=object),
    }


def make_timeseries(
    uuids: str | Sequence[str] = ("sensor:signal",),
    *,
    n_points: int = 1000,
    freq: str = "30s",
    start: str = "2025-01-01 00:00:00",
    baseline: float = 100.0,
    noise: float = 1.0,
    drift: float = 0.0,
    n_outliers: int = 0,
    value_column: str = "value_double",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a standard-schema synthetic timeseries DataFrame.

    Args:
        uuids: Signal identifiers (or a single one as a string); one block of
            ``n_points`` rows per uuid. An empty list gives an empty frame with
            the standard columns.
        n_points: Number of samples generated per uuid.
        freq: Pandas offset alias for the sampling interval (e.g. ``"30s"``).
        start: Timestamp of the first sample.
        baseline: Mean level of the generated signal.
        noise: Standard deviation of the Gaussian noise added to the signal.
        drift: Total linear drift applied across the series (tool-wear style).
        n_outliers: Number of large spike outliers injected per uuid. Spikes
            are ``12 * noise`` high, so with ``noise=0`` none are visible (a
            :class:`~ts_shape.errors.DataQualityWarning` says so).
        value_column: Which value column to populate -- one of
            ``value_double``, ``value_integer`` or ``value_bool``.
        seed: Seed for reproducibility; pass ``None`` for fresh randomness.

    Returns:
        DataFrame with columns ``systime``, ``uuid``, ``value_bool``,
        ``value_integer``, ``value_double``, ``value_string`` and ``is_delta``.
        The populated column is ``float64`` / ``int64`` / ``bool``; the unused
        ones are missing, as ``boolean``, ``Int64``, ``float64`` and
        ``object``.

    Raises:
        ValueError: If ``value_column`` is not a supported value column, or
            ``n_points`` is not positive.
    """
    if value_column not in _VALUE_COLUMNS:
        raise ValueError(
            f"value_column must be one of {_VALUE_COLUMNS}; got {value_column!r}"
        )
    if n_points <= 0:
        raise ValueError(f"n_points must be positive; got {n_points}")

    uuid_list = [uuids] if isinstance(uuids, str) else list(uuids)
    if not uuid_list:
        return empty_signal_frame()
    if n_outliers > 0 and noise == 0:
        warnings.warn(
            "n_outliers has no effect with noise=0: outliers are 12 * noise high",
            DataQualityWarning,
            stacklevel=caller_stacklevel(),
        )

    rng = np.random.default_rng(seed)
    frames = []
    for uuid in uuid_list:
        times = pd.date_range(start=start, periods=n_points, freq=freq)
        signal = baseline + rng.normal(0.0, noise, n_points)
        if drift:
            signal = signal + np.linspace(0.0, drift, n_points)
        if n_outliers > 0:
            idx = rng.choice(n_points, size=min(n_outliers, n_points), replace=False)
            signal[idx] += rng.choice([-1.0, 1.0], size=len(idx)) * noise * 12.0

        frame = pd.DataFrame(
            {
                "systime": times,
                "uuid": uuid,
                **_unused_columns(n_points),
                "is_delta": True,
            }
        )
        if value_column == "value_double":
            frame["value_double"] = signal
        elif value_column == "value_integer":
            frame["value_integer"] = signal.round().astype("int64")
        else:  # value_bool
            frame["value_bool"] = signal > baseline
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def make_id_signal(
    uuid: str = "object:id",
    values: Sequence[str] = ("A", "B", "C"),
    *,
    hold: int = 10,
    freq: str = "30s",
    start: str = "2025-01-01 00:00:00",
    value_column: str = "value_string",
    source_uuid: str | None = None,
) -> pd.DataFrame:
    """Generate a categorical *identifier* signal in the standard schema.

    Each value in ``values`` is held for ``hold`` consecutive samples, so the
    signal looks like a real batch / serial / coil / recipe id stream that
    changes over time. Feed it to :mod:`ts_shape.eventlog.objects` to extract
    object instances.

    Args:
        uuid: Signal identifier.
        values: Ordered id values; each held for ``hold`` samples.
        hold: Samples each value persists before the next one starts.
        freq: Pandas offset alias for the sampling interval.
        start: Timestamp of the first sample.
        value_column: Which column carries the id (``value_string`` or
            ``value_integer``).
        source_uuid: Optional ``source_uuid`` to stamp on every row.

    Returns:
        DataFrame with the standard ts-shape columns, ``value_column`` filled.

    Raises:
        ValueError: For an unsupported ``value_column``, ``hold <= 0``, or
            ``value_column="value_integer"`` with ids that are not whole
            numbers.
    """
    if value_column not in ("value_string", "value_integer"):
        raise ValueError("value_column must be 'value_string' or 'value_integer'")
    if hold <= 0:
        raise ValueError(f"hold must be positive; got {hold}")
    if value_column == "value_integer":
        bad = [v for v in values if not _is_integral(v)]
        if bad:
            raise ValueError(
                f"value_column='value_integer' needs whole-number ids; got "
                f"{bad[:5]}. Use value_column='value_string' for text ids."
            )

    n = len(values) * hold
    times = pd.date_range(start=start, periods=n, freq=freq)
    seq = [v for v in values for _ in range(hold)]
    frame = pd.DataFrame(
        {
            "systime": times,
            "uuid": uuid,
            **_unused_columns(n),
            "is_delta": False,
        }
    )
    if value_column == "value_integer":
        frame["value_integer"] = pd.array([int(v) for v in seq], dtype="Int64")
    else:
        frame["value_string"] = pd.array([str(v) for v in seq], dtype="string")
    if source_uuid is not None:
        frame["source_uuid"] = source_uuid
    return frame


def make_object_signals(
    *,
    seed: int = 0,
    days: float = 3,
    include: Iterable[str] | None = None,
    noise: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a labelled benchmark of object-identifier signals.

    Simulates one or two production lines and the plant around them (work
    orders with setups, stops and cycles; batches inside the material; one
    serial per cycle with a companion part counter; carriers circulating in a
    loop; operators logging in at the 06:00 / 14:00 / 22:00 shift changes;
    orders, customers and suppliers stamped at shipping and goods receipt;
    SSCC-labelled boxes put away into storage bins) and samples every signal
    the way a historian stores it: on change, with keep-alive re-sends, or
    periodically, with gaps. Identifiers use real formats (SAP numbers,
    GTIN-13/14 and SSCC with check digits, GS1 application identifiers,
    EPC GRAI-96 hex, UUIDs, date-coded batches); several formats per object
    type are drawn per seed. Negative controls (states, alarm texts,
    measurements, counters, timestamps, constants, plus hard cases such as a
    1/2/3 mode code) are mixed in.

    The uuids are opaque (``"sig:0007"``) and shuffled, so only the values and
    their behaviour over time tell what a signal carries.

    Seeds 0-4 are the dev split; seeds 100-119 the held-out split
    (``ts_shape._object_signals.DEV_SEEDS`` / ``HELDOUT_SEEDS``). Seeds from
    100 on also draw formats that dev seeds never use, plus wider and
    differently shaped hold-time distributions; the ``variant`` label names
    the format.

    Args:
        seed: Seed of the whole simulation; the output is deterministic per
            seed.
        days: Length of the recording in days.
        include: Object types (``"batch"``), roles (``"state"``) or negative
            kinds (``"alarm_text"``) to keep; ``None`` keeps every signal.
        noise: Add realistic read noise: missing readings, empty strings,
            garbage reads (``"NOREAD"``, truncated ids, ``65535``), lower-case
            and stripped leading zeros, and extra per-signal gaps.

    Returns:
        ``(frame, labels)``. ``frame`` is the long-format frame with the
        standard columns. ``labels`` has one row per uuid with ``uuid``,
        ``role`` (identifier, state, free_text, measurement, counter,
        timestamp or constant), ``object_type`` (``None`` unless an
        identifier), ``value_column``, ``parent_uuid`` (the signal it nests
        in: batch in material, serial in batch; else ``None``) and
        ``variant`` (the format or negative kind).

    Raises:
        ValueError: If ``days`` is not positive or ``include`` names an
            unknown object type or role.

    Example:
        >>> import ts_shape
        >>> frame, labels = ts_shape.make_object_signals(seed=0, days=1)
        >>> sorted(labels["role"].unique())
        ['constant', 'counter', 'free_text', 'identifier', 'measurement',
         'state', 'timestamp']
        >>> batch = labels.loc[labels["object_type"] == "batch"].iloc[0]
        >>> labels.set_index("uuid").loc[batch["parent_uuid"], "object_type"]
        'material'
    """
    from ts_shape._object_signals import generate

    return generate(seed=seed, days=days, include=include, noise=noise)


def _is_integral(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | np.integer):
        return True
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number.is_integer()


def make_plc_cycles(
    *,
    n_cycles: int = 300,
    seed: int = 0,
    start: str | pd.Timestamp = "2026-01-05 06:00:00",
    articles: Sequence[str] = ("A", "B"),
    article_block: int = 60,
    retry_probability: float = 0.06,
    abort_probability: float = 0.02,
    pause_every: int = 100,
    pause_minutes: float = 15.0,
    sample_period: str = "1s",
    heartbeat: bool = True,
    prefix: str = "plc",
    parts_per_cycle: int = 1,
    reported_cycle_time: bool = False,
    alt_sequence_probability: float = 0.0,
    shape_fault_probability: float = 0.0,
    parallel_lane: bool = False,
    poll_period: str | None = None,
    stop_causes: int = 0,
    stop_probability: float = 0.0,
    changeover_steps: bool = False,
    rampup_cycles: int = 0,
    handshake_partner: bool = False,
    order_size: int = 0,
    scrap_probability: float = 0.0,
    setting_changes: Sequence[tuple[int, float]] = (),
    energy: bool = False,
    curve_faults: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the raw signals of one cyclic machine PLC, with ground truth.

    A step chain ``10 .. 80`` (about 45 s per cycle) drives the signals a
    historian records for such a machine, with no mapping or naming
    convention a profiler could rely on:

    ======================  =====================================================
    ``<prefix>:step``        step chain (``value_integer``, one row per change)
    ``<prefix>:clamp_closed``  handshake, True from step 20 to step 70
    ``<prefix>:robot_ready``   handshake, True in step 30
    ``<prefix>:camera_done``   pulse at the end of each camera inspection
    ``<prefix>:camera_nok``    pulse when the camera rejects (then re-inspects)
    ``<prefix>:part_counter``  part counter, +1 in step 80 (the cycle boundary)
    ``<prefix>:article``       article (``value_string``), changes in blocks
    ``<prefix>:pressure``      analog clamp pressure, sampled every ``sample_period``
    ``<prefix>:heartbeat``     alive bit toggling every ``sample_period``
    ======================  =====================================================

    Injected disturbances (the ground truth a cycle profiler should find):
    camera NOK with one or two re-inspections (+12 s each: long cycles),
    aborted cycles that skip steps 40-70 (short cycles), breaks of about
    ``pause_minutes`` waiting in step 10 (pauses), every second article 6 s
    slower in step 40 (a variant) and a conveyor wait of 2.5 s in 40 % of the
    cycles in step 60 (a bimodal segment).

    Args:
        n_cycles: Number of complete cycles (counter increment to increment).
        seed: Random seed; the output is deterministic per seed.
        start: Timestamp of the first row.
        articles: Article names, produced in blocks of ``article_block``.
        article_block: Cycles per article block.
        retry_probability: Share of cycles with a camera NOK.
        abort_probability: Share of the other cycles that are aborted.
        pause_every: A break every this many cycles (0: none).
        pause_minutes: Typical length of a break.
        sample_period: Sample period of the pressure and the heartbeat.
        heartbeat: Whether to add the heartbeat signal.
        prefix: Prefix of the signal uuids.
        parts_per_cycle: Parts counted per cycle (the counter's step).
        reported_cycle_time: Add ``<prefix>:last_cycle_time``, the duration
            of the cycle just ended, published 1 s after the cycle boundary
            (as many controllers do).
        alt_sequence_probability: Share of cycles that run step 60 before
            the camera step 50 (an alternative route).
        shape_fault_probability: Share of cycles whose clamp pressure only
            reaches 130 instead of 180 bar (a curve-shape fault).
        parallel_lane: Add a second, independent machine on the same
            controller (``<prefix>:lane2_counter``, ``<prefix>:lane2_clamp``)
            at 1/1.13 of the cycle rate, so its phase drifts.
        poll_period: Read every signal at this fixed poll (``"13s"``), as a
            polling OPC UA client does; ``None`` (default) keeps the
            change-encoded rows.
        stop_causes: Number of stop causes (0-4) injected at the start of a
            cycle with ``stop_probability``: a jam (``alarm_1`` rises just
            before, about 2 min), material low (``alarm_2``, about 5 min), a
            manual fix (``mode`` 2 and ``door_open`` held, about 3 min) and an
            unknown cause that leaves no trace (about 1.5 min).
        stop_probability: Share of cycles that start with a stop.
        changeover_steps: At an article change, run a changeover in step 10
            of the first new cycle: ``setup_mode`` held, ``unlock_tool`` and
            ``lock_tool`` pulses and a new ``fixture_id``; in about 30 % of the
            changeovers the fixture is prepared while the old article still
            runs (an external step).
        rampup_cycles: After a pause, stop or changeover, this many cycles
            run step 40 slower (up to +60 %), and ``mold_temp`` (sampled)
            cools during the stop and heats up again.
        handshake_partner: Run a four-phase handshake with the conveyor in
            step 60 (``req_conveyor`` / ``ack_conveyor``); the acknowledge
            delay includes the conveyor wait.
        order_size: Cycles per production order (``order_no``, text; 0: no
            order signal).
        scrap_probability: Share of bad parts: ``good_counter`` and
            ``bad_counter`` add up to the part counter; the rate is 5x in a
            ramp-up, 10x on a curve fault and 3x after an alarm stop.
        setting_changes: ``(cycle, percent)`` changes of a held
            ``speed_override`` (100 at the start); step 40 runs at
            100/percent of its duration.
        energy: Add ``energy_kwh``, a cumulative meter (30 kW while
            producing, 4 kW in stops).
        curve_faults: ``{fault: probability}`` pressure-curve faults:
            ``low_plateau`` (130 instead of 180 bar), ``late_rise`` (+2 s),
            ``spike_in_hold`` (260 bar for 1 s).

    The optional disturbances draw random numbers only when enabled, so the
    default data do not change.

    Returns:
        ``(frame, truth)``. ``frame`` is the long-format frame with the
        standard columns. ``truth`` has one row per cycle: ``start``,
        ``end``, ``duration_seconds``, ``kind`` (``normal``, ``long``,
        ``short`` or ``pause``), ``article``, ``retry_count``,
        ``pause_seconds``, ``parts``, ``sequence`` (``standard`` /
        ``swapped``) and ``shape_fault``; with the options above also
        ``stop_cause``, ``stop_seconds``, ``changeover_seconds``, ``startup``,
        ``nok``, ``order_id``, ``setting_value``, ``curve_fault`` and
        ``handshake_wait_seconds``.

    Raises:
        ValueError: If ``n_cycles`` is below 1, ``articles`` is empty or
            ``sample_period`` is not positive.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_plc_cycles(n_cycles=120, seed=1)
        >>> frame["uuid"].nunique(), "plc:part_counter" in set(frame["uuid"])
        (9, True)
        >>> truth["kind"].value_counts().sort_index().to_dict()
        {'long': 2, 'normal': 113, 'pause': 1, 'short': 4}
    """
    from ts_shape._plc_cycles import generate

    return generate(
        n_cycles=n_cycles,
        seed=seed,
        start=start,
        articles=articles,
        article_block=article_block,
        retry_probability=retry_probability,
        abort_probability=abort_probability,
        pause_every=pause_every,
        pause_minutes=pause_minutes,
        sample_period=sample_period,
        heartbeat=heartbeat,
        prefix=prefix,
        parts_per_cycle=parts_per_cycle,
        reported_cycle_time=reported_cycle_time,
        alt_sequence_probability=alt_sequence_probability,
        shape_fault_probability=shape_fault_probability,
        parallel_lane=parallel_lane,
        poll_period=poll_period,
        stop_causes=stop_causes,
        stop_probability=stop_probability,
        changeover_steps=changeover_steps,
        rampup_cycles=rampup_cycles,
        handshake_partner=handshake_partner,
        order_size=order_size,
        scrap_probability=scrap_probability,
        setting_changes=setting_changes,
        energy=energy,
        curve_faults=curve_faults,
    )


def make_line_stops(
    *,
    duration: str = "48h",
    seed: int = 0,
    start: str | pd.Timestamp = "2026-01-05 06:00:00",
    mean_run_minutes: float = 90.0,
    count_period: str = "10s",
    prefix: str = "line",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a continuous line (no machine cycle) that stops now and then.

    The line counts produced units every ``count_period`` while it runs
    (``<prefix>:produced_count``) and holds ``<prefix>:running``. It stops
    after an exponential run time for one of three causes: a web break
    (``<prefix>:alarm_break`` rises just before, about 12 min), die pressure
    (``<prefix>:alarm_pressure``, about 25 min) or an unknown cause without a
    trace (about 6 min). All signals are logged on change.

    Args:
        duration: Length of the recording.
        seed: Random seed; the output is deterministic per seed.
        start: Timestamp of the first row.
        mean_run_minutes: Mean run time between two stops.
        count_period: The counter counts one unit per this period.
        prefix: Prefix of the signal uuids.

    Returns:
        ``(frame, truth)``: the long-format frame, and one row per stop with
        ``start``, ``end``, ``cause`` and ``duration_seconds``.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_line_stops(duration="24h", seed=1)
        >>> sorted(truth["cause"].unique())
        ['die_pressure', 'unknown', 'web_break']
    """
    from ts_shape._line_stops import generate

    return generate(
        duration=duration,
        seed=seed,
        start=start,
        mean_run_minutes=mean_run_minutes,
        count_period=count_period,
        prefix=prefix,
    )


def make_control_loops(
    *,
    loops: Sequence[str] = ("good", "oscillating", "sluggish"),
    duration: str = "8h",
    seed: int = 0,
    start: str | pd.Timestamp = "2026-01-05 06:00:00",
    sample_period: str = "2s",
    mean_step_minutes: float = 20.0,
    distractors: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate PI control loops with setpoint steps, and look-alike signals.

    Each loop is a first-order plant with dead time under a PI controller
    tuned ``good`` (lambda tuning), ``oscillating`` (gain far too high) or
    ``sluggish`` (gain far too low). Loop ``i`` writes ``plc:tic10<i>_sp``,
    ``_pv`` and ``_op`` (output in %, 0..100). The setpoint steps by 5..20
    units after an exponential hold (at least 5 min). With ``distractors``,
    a held setting nobody follows (``plc:line_speed_set``), a drifting
    ``plc:ambient_temp`` and a randomly moving ``plc:bypass_valve`` are
    added. All signals are sampled every ``sample_period``.

    Args:
        loops: Tuning quality of each loop: ``good``, ``oscillating`` or
            ``sluggish``.
        duration: Length of the recording.
        seed: Random seed; the output is deterministic per seed.
        start: Timestamp of the first row.
        sample_period: Sampling period of every signal.
        mean_step_minutes: Mean hold of a setpoint (plus 5 min).
        distractors: Add the look-alike signals.

    Returns:
        ``(frame, truth)``: the long-format frame, and one row per loop
        with ``loop``, ``quality``, ``setpoint_uuid``, ``actual_uuid``,
        ``output_uuid``, ``kp``, ``ti_seconds``, ``plant_gain``,
        ``plant_tau_seconds``, ``dead_time_seconds`` and ``step_count``.

    Example:
        >>> import ts_shape
        >>> frame, truth = ts_shape.make_control_loops(duration="1h", seed=1)
        >>> list(truth["quality"])
        ['good', 'oscillating', 'sluggish']
    """
    from ts_shape._control_loops import generate

    return generate(
        loops=loops,
        duration=duration,
        seed=seed,
        start=start,
        sample_period=sample_period,
        mean_step_minutes=mean_step_minutes,
        distractors=distractors,
    )
