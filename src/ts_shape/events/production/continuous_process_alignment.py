"""Continuous-process alignment: transport-lag compensation for multi-station lines."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base

_SPEED_FACTORS: Dict[str, float] = {
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
    "window_start",
    "uuid",
    "component",
    "position_offset_m",
    "mean_speed_m_s",
    "lag_seconds",
]


class ContinuousProcessAlignmentEvents(Base):
    """Align multi-station readings on a continuous production line to a common
    material reference time via speed-based transport lag compensation.

    Parameters
    ----------
    dataframe:
        Long-format DataFrame with ``time_column``, ``uuid_column``, and signal
        value columns.  All UUIDs (speed, stations, cut signals) share the same
        DataFrame.
    speed_uuid:
        UUID of the line-speed signal.
    line_config:
        Physical layout description — list of dicts with keys:
        ``name`` (str), ``offset`` (float, metres from reference), ``uuids`` (list[str]).
    ref_uuid:
        Optional UUID of a signal located at the reference point (offset = 0).
    cut_uuid:
        Optional UUID whose values carry the cut piece length in metres.
    time_column:
        Name of the timestamp column (default ``"systime"``).
    uuid_column:
        Name of the UUID/identifier column (default ``"uuid"``).
    value_column:
        Default value column used when no override is supplied to a method
        (default ``"value_double"``).
    speed_unit:
        Unit of the speed signal.  One of ``"m/min"``, ``"m/s"``, ``"mm/s"``.
    min_speed:
        Minimum speed in m/s used to clamp the speed before computing lag,
        preventing infinite lag at standstill (default ``0.01``).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        speed_uuid: str,
        line_config: List[Dict],
        *,
        ref_uuid: Optional[str] = None,
        cut_uuid: Optional[str] = None,
        time_column: str = "systime",
        uuid_column: str = "uuid",
        value_column: str = "value_double",
        speed_unit: str = "m/min",
        min_speed: float = 0.01,
    ) -> None:
        super().__init__(dataframe, time_column)

        self.speed_uuid = speed_uuid
        self.line_config = line_config
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
        self._uuid_meta: Dict[str, tuple] = {}
        for component in line_config:
            name = component["name"]
            offset = float(component["offset"])
            for uid in component.get("uuids", []):
                self._uuid_meta[uid] = (name, offset)

        self._all_station_uuids: List[str] = list(self._uuid_meta.keys())

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

    def _attach_speed(
        self,
        station_df: pd.DataFrame,
        speed_df: pd.DataFrame,
        speed_col: str = "value_double",
    ) -> pd.DataFrame:
        """Backward-fill merge: attach most-recent speed reading to each station row."""
        speed_subset = speed_df[[self.time_column, speed_col]].rename(
            columns={speed_col: "_speed_raw"}
        )
        merged = pd.merge_asof(
            station_df,
            speed_subset,
            on=self.time_column,
            direction="backward",
        )
        return merged

    def _lag_from_speed(self, merged: pd.DataFrame, offset_m: float) -> pd.Series:
        speed_m_s = (
            merged["_speed_raw"].fillna(self.min_speed) * self._speed_factor
        ).clip(lower=self.min_speed)
        return offset_m / speed_m_s

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def align_to_reference(
        self,
        station_uuids: Optional[List[str]] = None,
        *,
        value_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Shift each station reading backward by its transport lag to produce
        a common ``material_ref_time``.

        Parameters
        ----------
        station_uuids:
            Subset of UUIDs to process.  Defaults to all UUIDs in
            ``line_config``.
        value_column:
            Value column to carry through.  Defaults to the constructor
            ``value_column``.

        Returns
        -------
        DataFrame with columns:
        ``material_ref_time``, ``systime``, ``uuid``, ``component``,
        ``position_offset_m``, ``lag_seconds``, ``<value_column>``.
        """
        val_col = value_column or self.value_column
        uuids = station_uuids if station_uuids is not None else self._all_station_uuids

        if self.dataframe.empty:
            return pd.DataFrame(columns=_ALIGN_COLS + [val_col])

        speed_df = self._get_speed_df()

        parts: List[pd.DataFrame] = []
        for uid in uuids:
            if uid not in self._uuid_meta:
                continue
            component_name, offset_m = self._uuid_meta[uid]
            station_df = self._get_station_df(uid)
            if station_df.empty or speed_df.empty:
                continue

            merged = self._attach_speed(station_df, speed_df)
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
        cut_length_uuid: Optional[str] = None,
        part_counter_uuid: Optional[str] = None,
        cut_value_column: str = "value_double",
    ) -> pd.DataFrame:
        """Add ``piece_id``, ``piece_length_m``, and ``piece_cut_ref_time`` to
        ``aligned_df`` (output of :meth:`align_to_reference`).

        Cut-event detection strategies (in priority order):

        1. ``part_counter_uuid`` (boolean): each True row = 1 cut.
        2. ``part_counter_uuid`` (integer/float): each step where the counter
           increases by ``delta`` counts as ``delta`` cuts, all attributed to
           the same timestamp when the signal resolution is coarser than the
           cut rate.
        3. ``cut_length_uuid`` only: each row of the length signal = 1 cut.

        Parameters
        ----------
        aligned_df:
            Output of :meth:`align_to_reference`.
        cut_length_uuid:
            UUID carrying the cut piece length value.
        part_counter_uuid:
            UUID of a part-counter signal (boolean or monotonically increasing
            integer).
        cut_value_column:
            Column holding the length value in ``cut_length_uuid`` rows.

        Returns
        -------
        ``aligned_df`` augmented with ``piece_id`` (int, 1-based),
        ``piece_length_m`` (float or NaN), ``piece_cut_ref_time`` (Timestamp).
        """
        if cut_length_uuid is None and part_counter_uuid is None:
            raise ValueError(
                "Provide at least one of cut_length_uuid or part_counter_uuid."
            )

        raw_df = self.dataframe

        # ---- detect cut events ----
        # Each entry: (systime, piece_length_m or NaN, n_pieces)
        cut_events: List[tuple] = []

        if part_counter_uuid is not None:
            counter_df = (
                raw_df[raw_df[self.uuid_column] == part_counter_uuid]
                .sort_values(self.time_column)
                .reset_index(drop=True)
            )
            if not counter_df.empty:
                # Detect boolean or numeric
                bool_col = "value_bool"
                int_col = "value_integer"
                dbl_col = "value_double"

                if (
                    bool_col in counter_df.columns
                    and counter_df[bool_col].notna().any()
                ):
                    vals = counter_df[bool_col].fillna(False).astype(bool)
                    for i, row in counter_df.iterrows():
                        if vals.iloc[
                            i if isinstance(i, int) else counter_df.index.get_loc(i)
                        ]:
                            cut_events.append((row[self.time_column], np.nan, 1))
                else:
                    # Use integer or double counter
                    num_col = (
                        int_col
                        if int_col in counter_df.columns
                        and counter_df[int_col].notna().any()
                        else dbl_col
                    )
                    if num_col in counter_df.columns:
                        counts = counter_df[num_col].ffill().fillna(0)
                        prev = counts.iloc[0]
                        for idx in range(1, len(counter_df)):
                            curr = counts.iloc[idx]
                            delta = int(round(curr - prev))
                            if delta > 0:
                                ts = counter_df.iloc[idx][self.time_column]
                                cut_events.append((ts, np.nan, delta))
                            prev = curr
        else:
            # No counter — each row of cut_length_uuid is a cut
            length_df = (
                raw_df[raw_df[self.uuid_column] == cut_length_uuid]
                .sort_values(self.time_column)
                .reset_index(drop=True)
            )
            for _, row in length_df.iterrows():
                length_val = row.get(cut_value_column, np.nan)
                cut_events.append((row[self.time_column], length_val, 1))

        # ---- attach lengths when cut_length_uuid provided and counter used ----
        if cut_length_uuid is not None and part_counter_uuid is not None and cut_events:
            length_df = (
                raw_df[raw_df[self.uuid_column] == cut_length_uuid]
                .sort_values(self.time_column)
                .reset_index(drop=True)
            )
            if not length_df.empty:
                cut_times = pd.Series([e[0] for e in cut_events])
                length_times = length_df[self.time_column].reset_index(drop=True)
                length_vals = length_df[cut_value_column].reset_index(drop=True)
                # backward-fill: for each cut event, find most recent length
                indices = (
                    np.searchsorted(length_times.values, cut_times.values, side="right")
                    - 1
                )
                new_events = []
                for i, (ts, _, n) in enumerate(cut_events):
                    idx = indices[i]
                    length = (
                        length_vals.iloc[idx] if 0 <= idx < len(length_vals) else np.nan
                    )
                    new_events.append((ts, length, n))
                cut_events = new_events

        if not cut_events:
            result = aligned_df.copy()
            result["piece_id"] = np.nan
            result["piece_length_m"] = np.nan
            result["piece_cut_ref_time"] = pd.NaT
            return result

        # ---- expand cut_events to per-piece rows, align to material_ref_time ----
        piece_rows: List[Dict] = []
        piece_id = 1
        for ts, length_m, n_pieces in cut_events:
            # Determine material_ref_time of this cut event
            # Look up the cutter's offset via cut_length_uuid or part_counter_uuid
            cutter_uuid = part_counter_uuid or cut_length_uuid
            offset_m = 0.0
            if cutter_uuid in self._uuid_meta:
                offset_m = self._uuid_meta[cutter_uuid][1]
            elif cut_length_uuid in self._uuid_meta:
                offset_m = self._uuid_meta[cut_length_uuid][1]

            # Find speed at cut time
            speed_df = self._get_speed_df()
            if not speed_df.empty and offset_m > 0:
                idx = (
                    np.searchsorted(speed_df[self.time_column].values, ts, side="right")
                    - 1
                )
                idx = max(0, min(idx, len(speed_df) - 1))
                raw_speed = speed_df.iloc[idx][self.value_column]
                speed_m_s = max(float(raw_speed) * self._speed_factor, self.min_speed)
                lag_s = offset_m / speed_m_s
                cut_ref_time = ts - pd.Timedelta(seconds=lag_s)
            else:
                cut_ref_time = ts

            for _ in range(n_pieces):
                piece_rows.append(
                    {
                        "_piece_id": piece_id,
                        "_piece_length_m": length_m,
                        "_piece_cut_ref_time": cut_ref_time,
                    }
                )
                piece_id += 1

        # Build cut boundary array (material_ref_time of each cut)
        _cut_ref_times = np.array(
            [r["_piece_cut_ref_time"] for r in piece_rows], dtype="datetime64[ns]"
        )
        # Unique boundaries for searchsorted (first occurrence per piece)
        _unique_cut_boundaries = np.array(
            [piece_rows[0]["_piece_cut_ref_time"]]
            + [
                piece_rows[i]["_piece_cut_ref_time"]
                for i in range(1, len(piece_rows))
                if piece_rows[i]["_piece_cut_ref_time"]
                != piece_rows[i - 1]["_piece_cut_ref_time"]
            ],
            dtype="datetime64[ns]",
        )

        # Build a piece lookup: boundary index -> piece_id, length, ref_time
        # We need to map material_ref_time -> piece_id
        # Each piece_row represents one piece; pieces with same cut_ref_time share boundary
        # Build piece table indexed by piece_id (1-based)
        piece_table = pd.DataFrame(piece_rows).rename(
            columns={
                "_piece_id": "piece_id",
                "_piece_length_m": "piece_length_m",
                "_piece_cut_ref_time": "piece_cut_ref_time",
            }
        )
        # Build cut_ref_time -> first piece_id mapping (for assigning piece via boundary)
        # Strategy: material is assigned to the piece cut after it passes
        # i.e. material_ref_time in [cut[i-1], cut[i]) -> piece_id = i
        # We use the unique cut_ref_times as boundaries
        unique_cut_ref = piece_table.drop_duplicates("piece_cut_ref_time")[
            ["piece_id", "piece_length_m", "piece_cut_ref_time"]
        ].reset_index(drop=True)

        cut_boundaries = unique_cut_ref["piece_cut_ref_time"].values.astype(
            "datetime64[ns]"
        )

        mat_ref = aligned_df["material_ref_time"].values.astype("datetime64[ns]")
        # searchsorted: index 0 = before first cut → piece_id = first piece
        # index k = between cut[k-1] and cut[k] → piece_id row k-1 (0-based)
        indices = np.searchsorted(cut_boundaries, mat_ref, side="right") - 1
        indices = np.clip(indices, 0, len(unique_cut_ref) - 1)

        result = aligned_df.copy()
        result["piece_id"] = unique_cut_ref["piece_id"].iloc[indices].values
        result["piece_length_m"] = unique_cut_ref["piece_length_m"].iloc[indices].values
        result["piece_cut_ref_time"] = (
            unique_cut_ref["piece_cut_ref_time"].iloc[indices].values
        )
        return result

    def lag_profile(
        self,
        station_uuids: Optional[List[str]] = None,
        *,
        window: str = "1min",
    ) -> pd.DataFrame:
        """Compute per-window mean lag for each station.

        Returns
        -------
        DataFrame with columns:
        ``window_start``, ``uuid``, ``component``, ``position_offset_m``,
        ``mean_speed_m_s``, ``lag_seconds``.
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
            .rename(columns={self.time_column: "window_start"})
        )
        speed_resampled["mean_speed_m_s"] = (
            speed_resampled["mean_speed_raw"].fillna(self.min_speed)
            * self._speed_factor
        ).clip(lower=self.min_speed)

        parts: List[pd.DataFrame] = []
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
            ["window_start", "position_offset_m"]
        )

    def alignment_quality(
        self,
        station_uuids: Optional[List[str]] = None,
        *,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Coverage check: sample counts per window for speed and each station.

        Returns
        -------
        DataFrame with columns:
        ``window_start``, ``speed_sample_count``, ``has_speed_data``,
        ``has_full_coverage``, ``per_uuid_counts``.
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

        station_counts: Dict[str, pd.Series] = {}
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
        combined.index.name = "window_start"
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
                "window_start",
                "speed_sample_count",
                "has_speed_data",
                "has_full_coverage",
                "per_uuid_counts",
            ]
        ]
