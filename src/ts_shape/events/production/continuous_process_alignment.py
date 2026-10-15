"""Continuous-process alignment: transport-lag compensation for multi-station lines."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_increments
from ts_shape.utils.deprecation import REMOVAL, caller_stacklevel, keyword_only

_SPEED_FACTORS: dict[str, float] = {
    "m/min": 1.0 / 60.0,
    "m/s": 1.0,
    "mm/s": 1.0 / 1000.0,
}

_ALIGN_COLS = [
    "material_ref_time",
    "systime",
    "uuid",
    "component",
    "position_offset_m",
    "lag_seconds",
]

_LAG_COLS = [
    "start",
    "uuid",
    "component",
    "position_offset_m",
    "mean_speed_m_s",
    "lag_seconds",
]


def _time_dtype(aligned_df: pd.DataFrame) -> object:
    """dtype of ``material_ref_time`` (datetime64[ns] when absent or empty)."""
    if (
        "material_ref_time" in aligned_df.columns
        and pd.api.types.is_datetime64_any_dtype(aligned_df["material_ref_time"])
    ):
        return aligned_df["material_ref_time"].dtype
    return np.dtype("datetime64[ns]")


class ContinuousProcessAlignmentEvents(Base):
    """Align multi-station readings on a continuous production line.

    Readings are shifted to a common material reference time via speed-based
    transport lag compensation.

    Args:
        dataframe: Long-format DataFrame with ``time_column``, ``uuid_column``,
            and signal value columns. All UUIDs (speed, stations, cut signals)
            share the same DataFrame.
        speed_uuid: UUID of the line-speed signal.
        line_config: Physical layout description — list of dicts with keys
            ``name`` (str), ``offset`` (float, meters from reference), and
            ``uuids`` (list[str]).
        ref_uuid: Deprecated; has no effect and will be removed in the next major release
            (passing it emits a DeprecationWarning). Put a reference-point
            signal into ``line_config`` with ``offset`` 0 instead.
        cut_uuid: Optional UUID whose values carry the cut piece length in
            meters. Used by `segment_by_cut` as the default
            ``cut_length_uuid``.
        time_column: Name of the timestamp column (default ``"systime"``).
        uuid_column: Name of the UUID/identifier column (default ``"uuid"``).
        value_column: Value column of the speed signal, and the default
            station value column when no override is supplied to a method
            (default ``"value_double"``).
        speed_unit: Unit of the speed signal. One of ``"m/min"``, ``"m/s"``,
            ``"mm/s"``.
        min_speed: Minimum speed in m/s used to clamp the speed before
            computing lag, preventing infinite lag at standstill (default
            ``0.01``).

    Raises:
        ValueError: If ``speed_unit`` is not one of the supported units.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        speed_uuid: str,
        line_config: list[dict],
        *,
        ref_uuid: str | None = None,
        cut_uuid: str | None = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        speed_unit: str = "m/min",
        min_speed: float = 0.01,
    ) -> None:
        super().__init__(dataframe, time_column)

        self.speed_uuid = speed_uuid
        self.line_config = line_config
        if ref_uuid is not None:
            warnings.warn(
                f"ref_uuid has no effect and will be removed in {REMOVAL}",
                DeprecationWarning,
                stacklevel=caller_stacklevel(),
            )
        self.ref_uuid = ref_uuid
        self.cut_uuid = cut_uuid
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.value_column = value_column
        self.min_speed = min_speed

        if speed_unit not in _SPEED_FACTORS:
            raise ValueError(
                f"speed_unit must be one of {list(_SPEED_FACTORS)}; got {speed_unit!r}"
            )
        self._speed_factor: float = _SPEED_FACTORS[speed_unit]

        # Build flat lookup: uuid -> (component_name, offset_m)
        self._uuid_meta: dict[str, tuple] = {}
        for component in line_config:
            name = component["name"]
            offset = float(component["offset"])
            for uid in component.get("uuids", []):
                self._uuid_meta[uid] = (name, offset)

        self._all_station_uuids: list[str] = list(self._uuid_meta.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_speed_df(self) -> pd.DataFrame:
        df = self.dataframe
        speed_df = (
            df[df[self.uuid_column] == self.speed_uuid]
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )
        return speed_df

    def _get_station_df(self, uuid: str) -> pd.DataFrame:
        df = self.dataframe
        return (
            df[df[self.uuid_column] == uuid]
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )

    def _valid_speed(self, speed_df: pd.DataFrame, speed_col: str) -> pd.DataFrame:
        """Speed readings with a usable value, as ``[time, _speed_raw]``."""
        speed = pd.DataFrame(
            {
                self.time_column: speed_df[self.time_column],
                "_speed_raw": pd.to_numeric(speed_df[speed_col], errors="coerce")
                .astype("float64")
                .to_numpy(),
            }
        )
        return speed.dropna().reset_index(drop=True)

    def _attach_speed(
        self,
        station_df: pd.DataFrame,
        speed_df: pd.DataFrame,
        speed_col: str = "value_double",
    ) -> pd.DataFrame:
        """Attach the line speed in effect at each station row.

        The most recent valid speed reading is used (missing speed values are
        skipped, so the last known speed is held). Rows before the first speed
        reading take the first reading.
        """
        speed = self._valid_speed(speed_df, speed_col)
        if speed.empty:
            merged = station_df.copy()
            merged["_speed_raw"] = np.nan
            return merged
        merged = pd.merge_asof(
            station_df,
            speed,
            on=self.time_column,
            direction="backward",
        )
        merged["_speed_raw"] = merged["_speed_raw"].fillna(speed["_speed_raw"].iloc[0])
        return merged

    def _lag_from_speed(self, merged: pd.DataFrame, offset_m: float) -> pd.Series:
        speed_m_s = (
            merged["_speed_raw"].fillna(self.min_speed) * self._speed_factor
        ).clip(lower=self.min_speed)
        return offset_m / speed_m_s

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @keyword_only(1)
    def align_to_reference(
        self,
        station_uuids: list[str] | None = None,
        *,
        value_column: str | None = None,
    ) -> pd.DataFrame:
        """Shift each station reading backward by its transport lag.

        This produces a common ``material_ref_time`` for all stations:
        ``systime - offset / speed``, with the most recent valid speed reading
        at each station sample (missing speed values are skipped; samples
        before the first speed reading use the first one), clamped to at
        least ``min_speed``.

        Args:
            station_uuids: Subset of UUIDs to process. Defaults to all UUIDs in
                ``line_config``.
            value_column: Value column to carry through. Defaults to the
                constructor ``value_column``.

        Returns:
            DataFrame with columns ``material_ref_time``, ``systime``, ``uuid``,
                ``component``, ``position_offset_m``, ``lag_seconds``, and
                ``<value_column>``.
        """
        val_col = value_column or self.value_column
        uuids = station_uuids if station_uuids is not None else self._all_station_uuids

        if self.dataframe.empty:
            return pd.DataFrame(columns=_ALIGN_COLS + [val_col])

        speed_df = self._get_speed_df()

        parts: list[pd.DataFrame] = []
        for uid in uuids:
            if uid not in self._uuid_meta:
                continue
            component_name, offset_m = self._uuid_meta[uid]
            station_df = self._get_station_df(uid)
            if station_df.empty or speed_df.empty:
                continue

            merged = self._attach_speed(
                station_df, speed_df, speed_col=self.value_column
            )
            lag = self._lag_from_speed(merged, offset_m)

            out = merged[[self.time_column, self.uuid_column]].copy()
            if val_col in merged.columns:
                out[val_col] = merged[val_col]
            else:
                out[val_col] = np.nan

            out["component"] = component_name
            out["position_offset_m"] = offset_m
            out["lag_seconds"] = lag.values
            out["material_ref_time"] = merged[self.time_column] - pd.to_timedelta(
                lag.values, unit="s"
            )
            out = out.rename(columns={self.time_column: "systime"})
            parts.append(out)

        if not parts:
            return pd.DataFrame(columns=_ALIGN_COLS + [val_col])

        result = pd.concat(parts, ignore_index=True)
        front = _ALIGN_COLS + [val_col]
        extra = [c for c in result.columns if c not in front]
        return result[front + extra]

    def segment_by_cut(
        self,
        aligned_df: pd.DataFrame,
        *,
        cut_length_uuid: str | None = None,
        part_counter_uuid: str | None = None,
        cut_value_column: str = "value_double",
        handle_resets: bool = True,
    ) -> pd.DataFrame:
        """Add piece columns to the output of `align_to_reference`.

        Adds ``piece_id``, ``piece_length_m``, and ``piece_cut_ref_time`` to
        ``aligned_df``.

        Cut-event detection strategies (in priority order):

        1. ``part_counter_uuid`` (boolean): each True row = 1 cut.
        2. ``part_counter_uuid`` (integer/float): each step where the counter
           increases by ``delta`` counts as ``delta`` cuts, all attributed to
           the same timestamp when the signal resolution is coarser than the
           cut rate. The first reading is the baseline; missing readings are
           gaps.
        3. ``cut_length_uuid`` only: each row of the length signal = 1 cut.

        Each cut time is shifted back to the reference point by the cutter's
        own transport lag (when the cutter's uuid is in ``line_config``), the
        same way `align_to_reference` shifts station readings.

        Piece assignment: a cut finishes the piece whose material passed the
        reference point since the previous cut. Rows with ``material_ref_time``
        in ``(previous cut, this cut]`` get this cut's ``piece_id``; rows up to
        and including the first cut belong to piece 1; rows after the last
        cut are not cut yet and get ``piece_id`` <NA>, ``piece_length_m`` NaN
        and ``piece_cut_ref_time`` NaT. Pieces cut at the same instant (a
        counter step > 1) share one boundary, so the material goes to the
        first of them and the others have no rows.

        Args:
            aligned_df: Output of `align_to_reference`.
            cut_length_uuid: UUID carrying the cut piece length value.
                Defaults to the constructor ``cut_uuid`` when None.
            part_counter_uuid: UUID of a part-counter signal (boolean or
                monotonically increasing integer).
            cut_value_column: Column holding the length value in
                ``cut_length_uuid`` rows. With a counter, each cut takes the
                latest length reported at or before it.
            handle_resets: When True (default) and a numeric ``part_counter_uuid`` is
                used, a decreasing counter (a reset at a shift/part change or
                PLC restart) is treated as new production: the post-reset
                counter value is emitted as that many cut events instead of
                being skipped. When False, a reset's negative delta is
                ignored and its post-reset pieces are not counted.

        Returns:
            ``aligned_df`` augmented with ``piece_id`` (nullable ``Int64``,
                1-based, numbered in cut order), ``piece_length_m`` (float or
                NaN), and ``piece_cut_ref_time`` (the piece's cut at the
                reference point, in the data's time zone).

        Raises:
            ValueError: If neither ``cut_length_uuid`` (nor the constructor
                ``cut_uuid``) nor ``part_counter_uuid`` is given.
        """
        if cut_length_uuid is None:
            cut_length_uuid = self.cut_uuid
        if cut_length_uuid is None and part_counter_uuid is None:
            raise ValueError(
                "Provide at least one of cut_length_uuid or part_counter_uuid."
            )

        raw_df = self.dataframe
        tcol = self.time_column

        def _signal(uuid: str) -> pd.DataFrame:
            rows = raw_df[raw_df[self.uuid_column] == uuid]
            rows = rows[pd.to_datetime(rows[tcol]).notna()]
            order = np.argsort(rows[tcol].to_numpy(), kind="stable")
            return rows.iloc[order].reset_index(drop=True)

        # ---- detect cut events: (time, piece_length_m, n_pieces) ----
        cut_times = pd.Series(raw_df[tcol].iloc[:0])
        cut_lengths = np.empty(0, dtype="float64")
        cut_counts = np.empty(0, dtype=np.int64)

        if part_counter_uuid is not None:
            counter_df = _signal(part_counter_uuid)
            if not counter_df.empty:
                if (
                    "value_bool" in counter_df.columns
                    and counter_df["value_bool"].notna().any()
                ):
                    fired = (
                        counter_df["value_bool"]
                        .astype("boolean")
                        .fillna(False)
                        .to_numpy(dtype=bool)
                    )
                    cut_times = counter_df[tcol][fired]
                    cut_counts = np.ones(int(fired.sum()), dtype=np.int64)
                else:
                    num_col = (
                        "value_integer"
                        if "value_integer" in counter_df.columns
                        and counter_df["value_integer"].notna().any()
                        else "value_double"
                    )
                    if num_col in counter_df.columns:
                        counts = pd.to_numeric(
                            counter_df[num_col], errors="coerce"
                        ).astype("float64")
                        # Missing readings are gaps, not a counter at 0.
                        present = counts.notna().to_numpy()
                        counts = counts[present].reset_index(drop=True)
                        times = counter_df[tcol][present].reset_index(drop=True)
                        if handle_resets:
                            # Reset-aware: a decreasing counter is a reset, so
                            # the post-reset value is new production.
                            increments = counter_increments(counts)
                        else:
                            increments = counts.diff().fillna(0.0)
                        n = np.rint(increments.to_numpy()).astype(np.int64)
                        hit = n > 0
                        cut_times = times[hit]
                        cut_counts = n[hit]
            cut_lengths = np.full(len(cut_counts), np.nan)
        else:
            assert cut_length_uuid is not None
            length_df = _signal(cut_length_uuid)
            cut_times = length_df[tcol]
            cut_counts = np.ones(len(length_df), dtype=np.int64)
            if cut_value_column in length_df.columns:
                cut_lengths = (
                    pd.to_numeric(length_df[cut_value_column], errors="coerce")
                    .astype("float64")
                    .to_numpy()
                )
            else:
                cut_lengths = np.full(len(length_df), np.nan)

        cut_times = pd.Series(pd.to_datetime(cut_times)).reset_index(drop=True)

        # ---- attach lengths when a counter and a length signal are given ----
        if (
            cut_length_uuid is not None
            and part_counter_uuid is not None
            and len(cut_counts)
        ):
            length_df = _signal(cut_length_uuid)
            if not length_df.empty and cut_value_column in length_df.columns:
                # The length reported at or before each cut.
                length_times = pd.DatetimeIndex(pd.to_datetime(length_df[tcol]))
                idx = length_times.searchsorted(cut_times, side="right") - 1
                vals = (
                    pd.to_numeric(length_df[cut_value_column], errors="coerce")
                    .astype("float64")
                    .to_numpy()
                )
                cut_lengths = np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)

        result = aligned_df.copy()
        if not len(cut_counts):
            result["piece_id"] = pd.array([pd.NA] * len(result), dtype="Int64")
            result["piece_length_m"] = np.nan
            result["piece_cut_ref_time"] = pd.Series(
                pd.NaT, index=result.index, dtype=_time_dtype(aligned_df)
            )
            return result

        # ---- cut times at the reference point (shift back by cutter lag) ----
        cutter_uuid = part_counter_uuid or cut_length_uuid
        offset_m = 0.0
        if cutter_uuid in self._uuid_meta:
            offset_m = self._uuid_meta[cutter_uuid][1]
        elif cut_length_uuid in self._uuid_meta:
            offset_m = self._uuid_meta[cut_length_uuid][1]
        cut_ref = cut_times
        speed_df = self._get_speed_df()
        if offset_m != 0 and not speed_df.empty:
            merged = self._attach_speed(
                pd.DataFrame({tcol: cut_times}), speed_df, speed_col=self.value_column
            )
            lag = self._lag_from_speed(merged, offset_m)
            cut_ref = cut_times - pd.to_timedelta(lag.to_numpy(), unit="s")

        # ---- one row per piece; pieces of one counter step share a cut ----
        piece_ref = pd.Series(
            pd.DatetimeIndex(np.repeat(pd.DatetimeIndex(cut_ref), cut_counts))
        )
        piece_len = np.repeat(cut_lengths, cut_counts)
        piece_ids = np.arange(1, len(piece_ref) + 1, dtype=np.int64)

        # Material that passed the reference point in (previous cut, this cut]
        # belongs to the piece this cut finishes; material before the first
        # cut belongs to the first piece; material after the last cut is not
        # cut yet (piece_id <NA>). Pieces cut at the same instant (a counter
        # step > 1) share one boundary: the material goes to the first of them.
        bounds = pd.DatetimeIndex(piece_ref)
        mat = pd.DatetimeIndex(pd.to_datetime(aligned_df["material_ref_time"]))
        if (bounds.tz is None) != (mat.tz is None):
            raise ValueError(
                "material_ref_time and the cut signal must both be tz-aware or "
                "both be naive."
            )
        b_i8 = np.maximum.accumulate(bounds.as_unit("ns").asi8)
        first_of_bound = np.ones(len(b_i8), dtype=bool)
        first_of_bound[1:] = b_i8[1:] != b_i8[:-1]
        uniq_pos = np.flatnonzero(first_of_bound)
        k = np.searchsorted(b_i8[uniq_pos], mat.as_unit("ns").asi8, side="left")
        assigned = (k < len(uniq_pos)) & ~np.asarray(mat.isna())
        piece_pos = uniq_pos[np.clip(k, 0, len(uniq_pos) - 1)]

        ids = pd.array(piece_ids[piece_pos], dtype="Int64")
        ids[~assigned] = pd.NA
        result["piece_id"] = ids
        result["piece_length_m"] = np.where(assigned, piece_len[piece_pos], np.nan)
        ref_out = pd.Series(bounds[piece_pos], index=result.index)
        result["piece_cut_ref_time"] = ref_out.where(assigned, pd.NaT)
        return result

    @keyword_only(1)
    def lag_profile(
        self,
        station_uuids: list[str] | None = None,
        *,
        window: str = "1min",
    ) -> pd.DataFrame:
        """Compute per-window mean lag for each station.

        Args:
            station_uuids: Subset of UUIDs to process. Defaults to all UUIDs in
                ``line_config``.
            window: Pandas resample frequency for the speed averaging window
                (default ``"1min"``).

        Returns:
            DataFrame with columns ``start``, ``uuid``, ``component``,
                ``position_offset_m``, ``mean_speed_m_s``, and ``lag_seconds``.
                Windows without speed readings have NaN speed and lag.
        """
        uuids = station_uuids if station_uuids is not None else self._all_station_uuids
        speed_df = self._get_speed_df()

        if speed_df.empty:
            return pd.DataFrame(columns=_LAG_COLS)

        speed_resampled = (
            speed_df.set_index(self.time_column)[self.value_column]
            .resample(window)
            .mean()
            .rename("mean_speed_raw")
            .reset_index()
            .rename(columns={self.time_column: "start"})
        )
        # A window without speed readings has no lag (NaN), not the lag of
        # min_speed (which would be offset / 0.01 m/s, thousands of seconds).
        speed_resampled["mean_speed_m_s"] = (
            speed_resampled["mean_speed_raw"] * self._speed_factor
        ).clip(lower=self.min_speed)

        parts: list[pd.DataFrame] = []
        for uid in uuids:
            if uid not in self._uuid_meta:
                continue
            component_name, offset_m = self._uuid_meta[uid]
            row = speed_resampled.copy()
            row["uuid"] = uid
            row["component"] = component_name
            row["position_offset_m"] = offset_m
            row["lag_seconds"] = offset_m / row["mean_speed_m_s"]
            parts.append(row[_LAG_COLS])

        if not parts:
            return pd.DataFrame(columns=_LAG_COLS)

        return pd.concat(parts, ignore_index=True).sort_values(
            ["start", "position_offset_m"]
        )

    @keyword_only(1)
    def alignment_quality(
        self,
        station_uuids: list[str] | None = None,
        *,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Coverage check: sample counts per window for speed and each station.

        Args:
            station_uuids: Subset of UUIDs to check. Defaults to all UUIDs in
                ``line_config``.
            window: Pandas resample frequency for the counting window (default
                ``"1h"``).

        Returns:
            DataFrame with columns ``start``, ``speed_sample_count``,
                ``has_speed_data``, ``has_full_coverage``, and
                ``per_uuid_counts``.
        """
        uuids = station_uuids if station_uuids is not None else self._all_station_uuids
        df = self.dataframe

        speed_counts = (
            df[df[self.uuid_column] == self.speed_uuid]
            .set_index(self.time_column)[self.value_column]
            .resample(window)
            .count()
            .rename("speed_sample_count")
        )

        station_counts: dict[str, pd.Series] = {}
        for uid in uuids:
            if uid not in self._uuid_meta:
                continue
            station_counts[uid] = (
                df[df[self.uuid_column] == uid]
                .set_index(self.time_column)[self.value_column]
                .resample(window)
                .count()
                .rename(uid)
            )

        all_series = [speed_counts] + list(station_counts.values())
        combined = pd.concat(all_series, axis=1).fillna(0).astype(int)
        combined.index.name = "start"
        combined = combined.reset_index()

        combined["has_speed_data"] = combined["speed_sample_count"] > 0

        if station_counts:
            uid_cols = list(station_counts.keys())
            combined["has_full_coverage"] = combined["has_speed_data"] & combined[
                uid_cols
            ].gt(0).all(axis=1)
            combined["per_uuid_counts"] = combined[uid_cols].apply(
                lambda row: row.to_dict(), axis=1
            )
            combined = combined.drop(columns=uid_cols)
        else:
            combined["has_full_coverage"] = combined["has_speed_data"]
            combined["per_uuid_counts"] = [{} for _ in range(len(combined))]

        return combined[
            [
                "start",
                "speed_sample_count",
                "has_speed_data",
                "has_full_coverage",
                "per_uuid_counts",
            ]
        ]
