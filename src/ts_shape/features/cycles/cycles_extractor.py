"""Cut a signal into production cycles.

Every ``process_*`` method returns one row per cycle with the columns
``cycle_start``, ``cycle_end``, ``cycle_uuid``, ``is_complete`` and the
canonical interval columns ``start``, ``end``, ``duration_seconds``, ``uuid``
(``"cycles:<method>"``) and ``source_uuid`` (the start signal). Rules shared by
all methods:

- Only the rows of ``start_uuid`` (and ``end_uuid``) are read; other signals in
  the frame are ignored. A frame without a ``uuid`` column is taken as one
  pre-filtered signal.
- Time columns keep the input's dtype (a tz-aware ``systime`` stays tz-aware).
- An end closes the cycle that is open at that moment. At one timestamp, an
  end is applied before a start, so back-to-back cycles may share a timestamp
  (the end of one is the start of the next).
- A second start while a cycle is open aborts the open cycle: it stays in the
  output with ``cycle_end = NaT`` and ``is_complete = False``, and the end goes
  to the newer start. Cycles never overlap.
- An end with no open cycle (for example before the first start) is ignored
  and counted in ``get_extraction_stats()["unmatched_ends"]``.
- ``is_complete`` is True only when both boundaries were seen. A cycle with no
  end (the last one, or an aborted one) has ``cycle_end = NaT``. A cycle that
  was already running at the first non-missing sample (its real start lies
  before the data) keeps its observed end but has ``is_complete = False``;
  leading rows with a missing value do not count as data.
- Several start rows at the same timestamp are one start.
- ``cycle_uuid`` is deterministic: the same signal, method and start time
  always give the same id, so ids are stable across runs and unique within a
  result.
- No cycles: an empty frame with the full column set.
"""

import logging
import uuid
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import caller_stacklevel
from ts_shape.utils.frames import ffill_values, require_columns, to_ns

logger = logging.getLogger(__name__)

_CYCLE_NAMESPACE = uuid.UUID("6f1f2a4e-6d8c-5b7e-9a51-3c0c1e2f7a10")

_OUTPUT_COLUMNS = [
    "cycle_start",
    "cycle_end",
    "cycle_uuid",
    "is_complete",
    "start",
    "end",
    "duration_seconds",
    "uuid",
    "source_uuid",
]


def _empty_stats() -> dict[str, Any]:
    return {
        "total_cycles": 0,
        "complete_cycles": 0,
        "incomplete_cycles": 0,
        "unmatched_starts": 0,
        "unmatched_ends": 0,
        "partial_first_cycles": 0,
        "overlapping_cycles": 0,
        "warnings": [],
    }


class CycleExtractor(Base):
    """Class for processing cycles based on different criteria."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_uuid: str,
        end_uuid: str | None = None,
        value_change_threshold: float = 0.0,
        *,
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        """Initializes the class with the data and the UUIDs for cycle start and end.

        Args:
            dataframe: Input DataFrame with cycle data
            start_uuid: UUID for cycle start variable
            end_uuid: UUID for cycle end variable (defaults to start_uuid)
            value_change_threshold: Minimum threshold for considering a value change significant (default: 0.0)
            time_column: Timestamp column. Default ``"systime"``.
            uuid_column: Signal identifier column. Default ``"uuid"``. A frame
                without it is taken as one pre-filtered signal.

        Raises:
            ColumnNotFoundError: If ``dataframe`` has no ``time_column``.
            ValueError: If ``start_uuid`` is not a string.
        """
        require_columns(dataframe, [time_column], where="CycleExtractor")
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.uuid_column = uuid_column

        # Validate input types (Base already guards the DataFrame type).
        if not isinstance(start_uuid, str):
            raise ValueError("start_uuid must be a string")

        # Use Base's processed frame: a sorted, defensive copy. This avoids
        # mutating the caller's DataFrame and guarantees chronological order.
        self.df = self.dataframe
        self.start_uuid = start_uuid
        self.end_uuid = end_uuid if end_uuid else start_uuid
        self.value_change_threshold = abs(value_change_threshold)

        # Statistics tracking
        self._stats: dict[str, Any] = _empty_stats()

        logger.info(
            f"CycleExtractor initialized with start_uuid: {self.start_uuid} and end_uuid: {self.end_uuid}"
        )

    # ------------------------------------------------------------------
    # Signal selection and boundary events
    # ------------------------------------------------------------------

    def _signal_frame(self, signal_uuid: str | None = None) -> pd.DataFrame:
        """Time-ordered rows of one signal, NaT timestamps dropped.

        A frame without a ``uuid`` column is one pre-filtered signal. A frame
        whose ``uuid`` column holds a single other uuid is used as well (with a
        warning). With several signals and none matching, the result is empty.
        """
        signal_uuid = self.start_uuid if signal_uuid is None else signal_uuid
        df = self.df
        if self.time_column not in df.columns:
            return df.iloc[0:0]
        if self.uuid_column in df.columns and not df.empty:
            match = df[self.uuid_column] == signal_uuid
            if match.any():
                df = df[match]
            else:
                present = sorted(str(u) for u in df[self.uuid_column].dropna().unique())
                if len(present) == 1:
                    warnings.warn(
                        f"CycleExtractor: uuid {signal_uuid!r} not found; the frame "
                        f"holds one signal {present[0]!r}, which is used.",
                        DataQualityWarning,
                        stacklevel=caller_stacklevel(),
                    )
                else:
                    warnings.warn(
                        f"CycleExtractor: uuid {signal_uuid!r} not found. "
                        f"Available uuids: {present}",
                        DataQualityWarning,
                        stacklevel=caller_stacklevel(),
                    )
                    return df.iloc[0:0]
        return df[df[self.time_column].notna()]

    @staticmethod
    def _bool_state(frame: pd.DataFrame) -> pd.Series:
        """``value_bool`` as a nullable boolean Series (NA where missing)."""
        if "value_bool" not in frame.columns:
            return pd.Series(pd.NA, index=frame.index, dtype="boolean")
        return frame["value_bool"].astype("boolean")

    def _bool_edges(
        self, frame: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Timestamp | None]:
        """Rising and falling edge times of ``value_bool``.

        A missing sample holds the previous state; leading missing samples
        count as False. Returns the rising-edge times, the falling-edge times,
        and the time of the first non-missing sample when that sample is
        True (else None): the signal was already True when it was first seen,
        so the first rising edge (at that time) is not a real start.
        """
        raw = self._bool_state(frame)
        known_at = np.flatnonzero(raw.notna().to_numpy(dtype=bool))
        times = frame[self.time_column]
        partial = (
            times.iloc[known_at[0]]
            if len(known_at) and bool(raw.iloc[known_at[0]])
            else None
        )
        state = raw.ffill().fillna(False).astype(bool)
        prev = state.shift(fill_value=False)
        return times[state & ~prev], times[~state & prev], partial

    def _bool_events(
        self, frame: pd.DataFrame
    ) -> tuple[pd.Series, pd.Timestamp | None]:
        """Times of the True events of a pulse signal, and the partial start.

        A signal that is never False is an event log: every True row is an
        event (and none is partial). Otherwise a held True counts once, at
        its rising edge; see :meth:`_bool_edges` for the partial start.
        """
        raw = self._bool_state(frame)
        if not raw.eq(False).fillna(False).any():
            return frame[self.time_column][raw.eq(True).fillna(False)], None
        rising, _, partial = self._bool_edges(frame)
        return rising, partial

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def process_persistent_cycle(self) -> pd.DataFrame:
        """Processes cycles where the value of the variable stays true during the cycle.

        A cycle starts at a rising edge (False -> True) and ends at the next
        falling edge (True -> False), i.e. at the first False sample. A missing
        sample holds the previous state. A signal already True at its first
        non-missing sample gives a first cycle with ``is_complete = False``
        (leading missing samples are not data).

        Returns:
            One row per cycle (see the module docstring); empty, with the
                full column set, when no cycle is found.
        """
        frame = self._signal_frame()
        starts, ends, partial = self._bool_edges(frame)
        return self._generate_cycle_dataframe(
            starts, ends, method="persistent", partial_start=partial
        )

    def process_trigger_cycle(self) -> pd.DataFrame:
        """Processes cycles triggered by a true pulse (true, then back to false).

        Each trigger starts a cycle that ends at the next trigger. A trigger is
        a rising edge (False -> True); for a signal that is never False (an
        event log of True rows) every row is a trigger. The last trigger gives
        an incomplete cycle.

        Returns:
            One row per cycle (see the module docstring); empty, with the
                full column set, when no cycle is found.
        """
        frame = self._signal_frame()
        starts, partial = self._bool_events(frame)
        return self._generate_cycle_dataframe(
            starts, starts, method="trigger", same_rows=True, partial_start=partial
        )

    def process_separate_start_end_cycle(self) -> pd.DataFrame:
        """Processes cycles where different variables indicate cycle start and end.

        Start events are the True events of ``start_uuid``, end events the True
        events of ``end_uuid`` (a held True counts once, at its rising edge; a
        signal that is never False is an event log where every True row
        counts). An end closes the open cycle; a second start aborts it. When
        ``end_uuid`` equals ``start_uuid`` (or the frame has no ``uuid``
        column), each event closes the previous cycle and opens the next.

        Returns:
            One row per cycle (see the module docstring); empty, with the
                full column set, when no cycle is found.
        """
        start_frame = self._signal_frame(self.start_uuid)
        starts, partial = self._bool_events(start_frame)
        if self.uuid_column not in self.df.columns or self.start_uuid == self.end_uuid:
            return self._generate_cycle_dataframe(
                starts,
                starts,
                method="separate_start_end",
                same_rows=True,
                partial_start=partial,
            )
        ends, _ = self._bool_events(self._signal_frame(self.end_uuid))
        return self._generate_cycle_dataframe(
            starts, ends, method="separate_start_end", partial_start=partial
        )

    def process_step_sequence(self, start_step: int, end_step: int) -> pd.DataFrame:
        """Processes cycles based on a step sequence, where specific integer values denote cycle start and end.

        Args:
            start_step: ``value_integer`` step that opens a cycle when the signal enters
                it.
            end_step: ``value_integer`` step that closes the open cycle when the signal
                enters it.

        Returns:
            One row per cycle (see the module docstring); ``cycle_end`` is NaT
                for a cycle that never closes. A step held over several samples is
                entered once; a missing sample holds the previous step. With
                ``start_step == end_step`` each entry closes the previous cycle and
                opens the next.
        """
        frame = self._signal_frame()
        if "value_integer" in frame.columns:
            raw = to_numeric_checked(
                frame["value_integer"], what="step column 'value_integer'"
            ).astype("float64")
        else:
            raw = pd.Series(np.nan, index=frame.index, dtype="float64")
        step = raw.ffill()
        entered = step.notna() & step.ne(step.shift())
        times = frame[self.time_column]
        starts = times[entered & (step == start_step)]
        ends = times[entered & (step == end_step)]
        # Time of the first non-missing sample: leading missing samples are
        # not part of the step sequence.
        known_at = np.flatnonzero(raw.notna().to_numpy(dtype=bool))
        partial = (
            times.iloc[known_at[0]]
            if len(known_at) and raw.iloc[known_at[0]] == start_step
            else None
        )
        return self._generate_cycle_dataframe(
            starts,
            ends,
            method="step_sequence",
            same_rows=start_step == end_step,
            partial_start=partial,
        )

    def process_state_change_cycle(self) -> pd.DataFrame:
        """Processes cycles where the start of a new cycle is the end of the previous cycle.

        Every sample of the signal is a boundary: a cycle runs from one sample
        to the next later one. The first cycle starts at the first sample,
        whose state began before the data, so it is marked incomplete.

        Returns:
            One row per cycle (see the module docstring); empty, with the
                full column set, when no cycle is found.
        """
        frame = self._signal_frame()
        times = frame[self.time_column]
        return self._generate_cycle_dataframe(
            times,
            times,
            method="state_change",
            same_rows=True,
            partial_start=times.iloc[0] if len(times) else None,
        )

    def process_value_change_cycle(self) -> pd.DataFrame:
        """Processes cycles where a change in the value indicates a new cycle.

        A sample is a boundary when one of its value columns differs from the
        previous sample: numbers by more than ``value_change_threshold``
        (compared sample to sample), bools and strings by any change. A missing
        value holds the previous one. The first sample is a boundary too; its
        cycle began before the data and is marked incomplete.

        Returns:
            One row per cycle (see the module docstring); empty, with the
                full column set, when no cycle is found.
        """
        frame = self._signal_frame()
        change = pd.Series(False, index=frame.index)
        for column in ("value_double", "value_integer"):
            if column in frame.columns:
                values = to_numeric_checked(
                    frame[column], what=f"column {column!r}"
                ).astype("float64")
                values = values.ffill()
                change |= (
                    values.diff().abs().gt(self.value_change_threshold).fillna(False)
                )
        for column in ("value_bool", "value_string"):
            if column in frame.columns:
                held = ffill_values(frame[column].astype(object))
                prev = held.shift()
                change |= (held.ne(prev) & held.notna() & prev.notna()).astype(bool)
        if len(change):
            change.iloc[0] = True
        times = frame[self.time_column][change.to_numpy(dtype=bool)]
        return self._generate_cycle_dataframe(
            times,
            times,
            method="value_change",
            same_rows=True,
            partial_start=times.iloc[0] if len(times) else None,
        )

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    def _time_dtype(self) -> Any:
        if self.time_column in self.df.columns:
            return self.df[self.time_column].dtype
        return pd.Series(dtype="datetime64[ns]").dtype

    def _empty_cycles(self) -> pd.DataFrame:
        time_dtype = self._time_dtype()
        return pd.DataFrame(
            {
                "cycle_start": pd.Series(dtype=time_dtype),
                "cycle_end": pd.Series(dtype=time_dtype),
                "cycle_uuid": pd.Series(dtype=object),
                "is_complete": pd.Series(dtype=bool),
                "start": pd.Series(dtype=time_dtype),
                "end": pd.Series(dtype=time_dtype),
                "duration_seconds": pd.Series(dtype="float64"),
                "uuid": pd.Series(dtype=object),
                "source_uuid": pd.Series(dtype=object),
            }
        )

    @staticmethod
    def _pair_events(
        starts: pd.Series, ends: pd.Series, same_rows: bool
    ) -> tuple[pd.Series, pd.Series, int]:
        """Pair start and end times; returns (cycle_start, cycle_end, unmatched_ends).

        Ends are applied before starts at the same timestamp. An end closes
        the cycle opened by the immediately preceding start event; a start
        followed by another start (or nothing) has no end.
        """
        start_times = (
            pd.Series(starts.dropna().unique(), dtype=starts.dtype)
            .sort_values(kind="stable")
            .reset_index(drop=True)
        )
        if same_rows:
            end_times = start_times.shift(-1)
            return start_times, end_times, 0

        end_times = (
            pd.Series(ends.dropna().unique(), dtype=starts.dtype)
            .sort_values(kind="stable")
            .reset_index(drop=True)
        )
        events = pd.DataFrame(
            {
                "time": pd.concat([end_times, start_times], ignore_index=True),
                "is_start": np.r_[
                    np.zeros(len(end_times), dtype=bool),
                    np.ones(len(start_times), dtype=bool),
                ],
            }
        ).sort_values(["time", "is_start"], kind="stable", ignore_index=True)
        is_start = events["is_start"].to_numpy()
        prev_is_start = np.r_[False, is_start[:-1]]
        closes = ~is_start & prev_is_start
        # Start i is closed by event i + 1 when that event is a closing end.
        next_closes = np.r_[closes[1:], False]
        end_of_start = events["time"].shift(-1).where(next_closes)
        cycle_start = events["time"][is_start].reset_index(drop=True)
        cycle_end = end_of_start[is_start].reset_index(drop=True)
        unmatched_ends = int((~is_start & ~closes).sum())
        return cycle_start, cycle_end, unmatched_ends

    def _cycle_ids(self, method: str, starts: pd.Series) -> list[str]:
        key = f"{self.start_uuid}|{self.end_uuid}|{method}"
        if starts.empty:
            return []
        ns = starts.astype("datetime64[ns, UTC]") if starts.dt.tz else starts
        stamps = ns.dt.as_unit("ns").astype("int64").tolist()
        return [str(uuid.uuid5(_CYCLE_NAMESPACE, f"{key}|{t}")) for t in stamps]

    def _generate_cycle_dataframe(
        self,
        cycle_starts: pd.Series,
        cycle_ends: pd.Series,
        method: str = "cycle",
        same_rows: bool = False,
        partial_start: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Pair boundary times into the cycle table.

        Args:
            cycle_starts: Times that open a cycle.
            cycle_ends: Times that close the open cycle.
            method: Short method name, used in ``uuid`` and ``cycle_uuid``.
            same_rows: Starts and ends are the same events: each one closes
                the previous cycle and opens the next.
            partial_start: Start time of a cycle that was already running at
                the first sample; that cycle is marked incomplete.
        """
        self._stats.update({k: v for k, v in _empty_stats().items() if k != "warnings"})
        if cycle_starts.empty:
            return self._empty_cycles()

        start, end, unmatched_ends = self._pair_events(
            cycle_starts, cycle_ends, same_rows
        )
        has_end = end.notna()
        is_complete = has_end.copy()
        partial_count = 0
        if partial_start is not None and not pd.isna(partial_start):
            partial = start == partial_start
            partial_count = int(partial.sum())
            is_complete &= ~partial

        cycle_df = pd.DataFrame(
            {
                "cycle_start": start,
                "cycle_end": end,
                "cycle_uuid": pd.Series(self._cycle_ids(method, start), dtype=object),
                "is_complete": is_complete.astype(bool),
            }
        )
        cycle_df["start"] = cycle_df["cycle_start"]
        cycle_df["end"] = cycle_df["cycle_end"]
        cycle_df["duration_seconds"] = (
            cycle_df["cycle_end"] - cycle_df["cycle_start"]
        ).dt.total_seconds()
        cycle_df["uuid"] = f"cycles:{method}"
        cycle_df["source_uuid"] = self.start_uuid
        cycle_df = cycle_df[_OUTPUT_COLUMNS]

        complete_cycles = int(cycle_df["is_complete"].sum())
        incomplete_cycles = len(cycle_df) - complete_cycles
        unmatched_starts = int((~has_end).sum())

        if unmatched_starts:
            warning_msg = (
                f"{unmatched_starts} cycle(s) have no matching end and are "
                "marked incomplete."
            )
            # A cycle still open when the data ends is expected, not a data
            # quality problem: it is marked incomplete and only logged.
            logger.warning("%s", warning_msg)
            self._stats["warnings"].append(warning_msg)

        self._stats.update(
            {
                "total_cycles": len(cycle_df),
                "complete_cycles": complete_cycles,
                "incomplete_cycles": incomplete_cycles,
                "unmatched_starts": unmatched_starts,
                "unmatched_ends": unmatched_ends,
                "partial_first_cycles": partial_count,
            }
        )

        logger.info(
            f"Generated {len(cycle_df)} cycles ({complete_cycles} complete, {incomplete_cycles} incomplete)."
        )
        return cycle_df

    # ------------------------------------------------------------------
    # Validation and diagnostics
    # ------------------------------------------------------------------

    def _parse_duration(self, duration_str: str) -> pd.Timedelta:
        """Parse a duration string like '1s', '5m', '1h' to a pandas Timedelta.

        Args:
            duration_str: Duration string (e.g., '1s', '5m', '1h', '2d')

        Returns:
            pd.Timedelta object

        Raises:
            ValueError: If duration string format is invalid
        """
        import re

        match = re.match(r"^(\d+(?:\.\d+)?)\s*([smhd])$", duration_str.lower().strip())
        if not match:
            # Anything pandas understands ('5min', '90s', '1h30min', ...).
            try:
                return pd.Timedelta(duration_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid duration format: '{duration_str}'. Expected a pandas "
                    "duration ('5min', '90s', '1h30min') or a number followed by "
                    "s/m/h/d (e.g., '1s', '5m', '1h')"
                ) from exc

        number, unit = match.groups()
        unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}

        return pd.Timedelta(**{unit_map[unit]: float(number)})

    def validate_cycles(
        self,
        cycle_df: pd.DataFrame,
        min_duration: str = "1s",
        max_duration: str = "1h",
        warn: bool = True,
    ) -> pd.DataFrame:
        """Validate cycles based on duration constraints.

        Args:
            cycle_df: DataFrame with cycle data (output from process_* methods)
            min_duration: Minimum acceptable cycle duration (default: '1s')
            max_duration: Maximum acceptable cycle duration (default: '1h')
            warn: Whether to emit a
                :class:`~ts_shape.errors.DataQualityWarning` for invalid
                cycles (default: True)

        Returns:
            A copy of ``cycle_df`` with ``cycle_duration`` (Timedelta),
                ``duration_seconds``, ``is_valid`` and ``validation_issue``
                (``""``, ``"incomplete_cycle"``, ``"too_short;"`` or
                ``"too_long;"``). Incomplete cycles are invalid; the duration
                bounds are inclusive. An empty input gives an empty frame with
                these columns.
        """
        # Parse duration constraints
        min_td = self._parse_duration(min_duration)
        max_td = self._parse_duration(max_duration)

        # Create a copy to avoid modifying the original
        validated_df = cycle_df.copy()

        # Calculate cycle durations
        validated_df["cycle_duration"] = pd.to_timedelta(
            validated_df["cycle_end"] - validated_df["cycle_start"]
        )
        if "duration_seconds" not in validated_df.columns:
            validated_df["duration_seconds"] = validated_df[
                "cycle_duration"
            ].dt.total_seconds()

        # Initialize validation columns
        validated_df["is_valid"] = pd.Series(True, index=validated_df.index, dtype=bool)
        validated_df["validation_issue"] = pd.Series(
            "", index=validated_df.index, dtype=object
        )
        if validated_df.empty:
            logger.warning("Empty cycle DataFrame provided for validation.")
            return validated_df

        if "is_complete" in validated_df.columns:
            complete_mask = validated_df["is_complete"].astype(bool)
        else:
            complete_mask = validated_df["cycle_end"].notna()

        # Check for incomplete cycles
        incomplete_mask = ~complete_mask
        if incomplete_mask.any():
            validated_df.loc[incomplete_mask, "is_valid"] = False
            validated_df.loc[incomplete_mask, "validation_issue"] = "incomplete_cycle"
            if warn:
                warnings.warn(
                    f"validate_cycles: found {int(incomplete_mask.sum())} "
                    "incomplete cycles.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )

        # Check duration constraints (only for complete cycles)
        too_short_mask = complete_mask & (validated_df["cycle_duration"] < min_td)
        too_long_mask = complete_mask & (validated_df["cycle_duration"] > max_td)

        if too_short_mask.any():
            validated_df.loc[too_short_mask, "is_valid"] = False
            validated_df.loc[too_short_mask, "validation_issue"] = (
                validated_df.loc[too_short_mask, "validation_issue"] + "too_short;"
            )
            if warn:
                warnings.warn(
                    f"validate_cycles: found {int(too_short_mask.sum())} cycles "
                    f"shorter than {min_duration}.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )

        if too_long_mask.any():
            validated_df.loc[too_long_mask, "is_valid"] = False
            validated_df.loc[too_long_mask, "validation_issue"] = (
                validated_df.loc[too_long_mask, "validation_issue"] + "too_long;"
            )
            if warn:
                warnings.warn(
                    f"validate_cycles: found {int(too_long_mask.sum())} cycles "
                    f"longer than {max_duration}.",
                    DataQualityWarning,
                    stacklevel=caller_stacklevel(),
                )

        valid_count = validated_df["is_valid"].sum()
        invalid_count = (~validated_df["is_valid"]).sum()

        logger.info(
            "Validation complete: %d valid cycles, %d invalid cycles.",
            valid_count,
            invalid_count,
        )

        return validated_df

    @staticmethod
    def _as_ns(times: pd.Series) -> np.ndarray:
        """Datetimes as int64 nanoseconds (UTC for tz-aware input)."""
        return to_ns(pd.to_datetime(times))

    def detect_overlapping_cycles(
        self, cycle_df: pd.DataFrame, resolve: str = "flag"
    ) -> pd.DataFrame:
        """Detect and optionally resolve overlapping cycles.

        Two cycles overlap when one starts strictly before the other ends, so
        back-to-back cycles sharing a timestamp do not overlap. Only cycles
        with a known end are compared.

        Args:
            cycle_df: DataFrame with cycle data
            resolve: How to handle overlaps - 'flag' (mark only), 'keep_first'
                (earliest start wins), 'keep_last' (latest start wins),
                'keep_longest' (longest wins). Resolution is greedy, so the
                kept cycles never overlap and a cycle is dropped only when it
                overlaps a kept one.

        Returns:
            ``cycle_df`` sorted by ``cycle_start`` with a ``has_overlap`` column;
                for a ``keep_*`` strategy the dropped cycles are removed.

        Raises:
            ValueError: For an unknown ``resolve``.
        """
        if resolve not in ("flag", "keep_first", "keep_last", "keep_longest"):
            raise ValueError(
                f"Unknown resolve {resolve!r}; use 'flag', 'keep_first', "
                "'keep_last' or 'keep_longest'."
            )
        result_df = cycle_df.sort_values("cycle_start", kind="stable").reset_index(
            drop=True
        )
        result_df["has_overlap"] = pd.Series(False, index=result_df.index, dtype=bool)
        if result_df.empty:
            logger.warning("Empty cycle DataFrame provided for overlap detection.")
            return result_df

        # A partial first cycle still has a known extent and takes part; only
        # a missing end (or start) excludes a cycle.
        known = result_df["cycle_end"].notna() & result_df["cycle_start"].notna()
        positions = np.flatnonzero(known.to_numpy())
        starts = self._as_ns(result_df["cycle_start"][known])
        ends = self._as_ns(result_df["cycle_end"][known])
        n = len(positions)
        if n == 0:
            self._stats["overlapping_cycles"] = 0
            return result_df

        prev_max_end = np.r_[np.iinfo(np.int64).min, np.maximum.accumulate(ends)[:-1]]
        next_start = np.r_[starts[1:], np.iinfo(np.int64).max]
        overlap = (starts < prev_max_end) | (ends > next_start)
        result_df.loc[positions[overlap], "has_overlap"] = True

        later_overlaps = np.searchsorted(starts, ends, side="left") - (np.arange(n) + 1)
        overlap_count = int(np.clip(later_overlaps, 0, None).sum())
        self._stats["overlapping_cycles"] = overlap_count
        if overlap_count > 0:
            warnings.warn(
                f"detect_overlapping_cycles: detected {overlap_count} "
                "overlapping cycle pairs.",
                DataQualityWarning,
                stacklevel=caller_stacklevel(),
            )

        if resolve == "flag" or overlap_count == 0:
            return result_df

        keep = np.ones(n, dtype=bool)
        s_list, e_list = starts.tolist(), ends.tolist()
        if resolve == "keep_first":
            last_end = None
            for i in range(n):
                if last_end is not None and s_list[i] < last_end:
                    keep[i] = False
                else:
                    last_end = e_list[i]
        elif resolve == "keep_last":
            first_start = None
            for i in range(n - 1, -1, -1):
                if first_start is not None and e_list[i] > first_start:
                    keep[i] = False
                else:
                    first_start = s_list[i]
        else:  # keep_longest
            import bisect

            order = sorted(range(n), key=lambda i: (s_list[i] - e_list[i], i))
            kept_s: list[int] = []
            kept_e: list[int] = []
            for i in order:
                s, e = s_list[i], e_list[i]
                pos = bisect.bisect_left(kept_s, s)
                clash = (pos > 0 and kept_e[pos - 1] > s) or (
                    pos < len(kept_s) and kept_s[pos] < e
                )
                if clash:
                    keep[i] = False
                else:
                    kept_s.insert(pos, s)
                    kept_e.insert(pos, e)

        drop = positions[~keep]
        if len(drop):
            result_df = result_df.drop(index=drop).reset_index(drop=True)
            logger.info(
                f"Resolved overlaps: removed {len(drop)} cycles using '{resolve}' strategy."
            )
        return result_df

    def suggest_method(self) -> dict[str, Any]:
        """Suggest the best cycle extraction method based on data characteristics.

        Analyzes the start signal's rows to recommend extraction method(s).

        Returns:
            Dictionary with method suggestions and reasoning
        """
        suggestions: dict[str, Any] = {
            "recommended_methods": [],
            "reasoning": [],
            "data_characteristics": {},
        }
        frame = self._signal_frame()

        def has(column: str) -> bool:
            return column in frame.columns and bool(frame[column].notna().any())

        has_bool = has("value_bool")
        has_integer = has("value_integer")
        has_double = has("value_double")
        has_string = has("value_string")

        suggestions["data_characteristics"] = {
            "has_boolean_values": has_bool,
            "has_integer_values": has_integer,
            "has_double_values": has_double,
            "has_string_values": has_string,
            "row_count": len(frame),
            "separate_start_end": self.start_uuid != self.end_uuid,
        }

        # Suggest methods based on characteristics
        if suggestions["data_characteristics"]["separate_start_end"]:
            suggestions["recommended_methods"].append(
                "process_separate_start_end_cycle"
            )
            suggestions["reasoning"].append("Separate start and end UUIDs detected")

        if has_bool:
            # Analyze boolean patterns
            bool_data = frame["value_bool"].astype("boolean").dropna()
            if len(bool_data) > 1 and bool(
                bool_data.ne(bool_data.shift()).iloc[1:].any()
            ):
                suggestions["recommended_methods"].append("process_persistent_cycle")
                suggestions["reasoning"].append(
                    "Boolean transitions detected, suitable for persistent cycles"
                )

                suggestions["recommended_methods"].append("process_trigger_cycle")
                suggestions["reasoning"].append(
                    "Boolean data present, can use trigger-based extraction"
                )

        if has_integer:
            # Check if integers represent steps
            unique_values = frame["value_integer"].dropna().nunique()
            if 2 <= unique_values <= 20:  # Reasonable step count
                suggestions["recommended_methods"].append("process_step_sequence")
                suggestions["reasoning"].append(
                    f"Integer data with {unique_values} unique values suggests step sequence"
                )

        if has_double or has_integer or has_string:
            # Value change method is versatile
            suggestions["recommended_methods"].append("process_value_change_cycle")
            suggestions["reasoning"].append(
                "Data shows changing values, suitable for value change detection"
            )

        # Check for state-like patterns
        if has_integer or has_string:
            suggestions["recommended_methods"].append("process_state_change_cycle")
            suggestions["reasoning"].append(
                "State-based data detected, consider state change cycles"
            )

        # If no specific method suggested, recommend value_change as fallback
        if not suggestions["recommended_methods"]:
            suggestions["recommended_methods"].append("process_value_change_cycle")
            suggestions["reasoning"].append(
                "Default method: works with any value changes"
            )

        logger.info(f"Method suggestion: {suggestions['recommended_methods']}")
        return suggestions

    def get_extraction_stats(self) -> dict[str, Any]:
        """Get statistics about the last cycle extraction.

        Returns:
            Dictionary with ``total_cycles``, ``complete_cycles``,
                ``incomplete_cycles``, ``unmatched_starts`` (cycles without an
                end), ``unmatched_ends`` (ends with no open cycle),
                ``partial_first_cycles``, ``overlapping_cycles`` (from the last
                ``detect_overlapping_cycles``), ``warnings``, ``success_rate``
                and ``configuration``.
        """
        stats = self._stats.copy()
        stats["warnings"] = list(self._stats["warnings"])

        # Calculate success rate
        if stats["total_cycles"] > 0:
            stats["success_rate"] = stats["complete_cycles"] / stats["total_cycles"]
        else:
            stats["success_rate"] = 0.0

        # Add configuration info
        stats["configuration"] = {
            "start_uuid": self.start_uuid,
            "end_uuid": self.end_uuid,
            "value_change_threshold": self.value_change_threshold,
        }

        return stats

    def reset_stats(self) -> None:
        """Reset extraction statistics."""
        self._stats = _empty_stats()
        logger.info("Statistics reset.")
