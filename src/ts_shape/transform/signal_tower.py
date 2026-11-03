"""Decode a PLC signal / stack / tower light into a canonical state signal.

Industrial machines often expose their state as several boolean lamp signals
on a stack light — a red lamp for *alarm*, yellow for *warning*, green for
*running*, plus (optionally) a buzzer. Downstream analytics want a single
state series like ``alarm / warning / running / stopped`` instead of four
boolean signals.

:class:`SignalTowerDecoder` composes the lamps into one canonical
``value_string`` signal using a configurable priority (default: alarm > warning
> running > stopped). The output is a change-encoded long-format frame that
feeds every state-based detector (:class:`MachineStateEvents`,
:class:`StateTransitionEvents`, :class:`AlarmManagementEvents`, ...)
without further massaging.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.deprecation import caller_stacklevel
from ts_shape.utils.frames import require_columns

logger = logging.getLogger(__name__)

#: Default priority order (first wins when more than one lamp is on).
DEFAULT_PRIORITY: tuple[str, ...] = ("alarm", "warning", "running", "stopped")

#: Default label used when no lamp is on (every mapped signal is False).
DEFAULT_OFF_STATE = "stopped"

__all__: list[str] = [
    "DEFAULT_OFF_STATE",
    "DEFAULT_PRIORITY",
    "SignalTowerDecoder",
]


class SignalTowerDecoder(Base):
    """Collapse a lamp stack into a single canonical state signal.

    Args:
        dataframe: Long-format frame (``systime``, ``uuid``, ``value_bool`` / ``value_integer``).
        lamp_map: ``{signal_uuid: semantic_state}``. Each listed uuid is a
            boolean lamp whose "on" rows contribute the mapped state. Several
            uuids may map to the same state (two "warning" lamps).
        event_uuid: UUID stamped on the decoded output rows.
        priority: Ordered sequence from highest-severity state to lowest; when
            several lamps are on at the same instant, the state earliest in
            ``priority`` wins. Default: :data:`DEFAULT_PRIORITY`. Any state
            you map through ``lamp_map`` that is not in ``priority`` ranks
            below every listed state, in the order you listed them.
        off_state: Label emitted when every lamp is off. Default
            :data:`DEFAULT_OFF_STATE` (``"stopped"``). Pass ``None`` to skip
            emitting an "all off" row (the previous winning state then
            persists until the next change).
        value_column: Column holding each lamp's boolean reading. Defaults to
            ``"value_bool"``; a boolean cast is applied to any dtype so an
            integer 0/1 lamp works too.
        time_column: Timestamp column.
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        lamp_map: Mapping[str, str],
        *,
        event_uuid: str = "transform:signal_tower",
        priority: Sequence[str] = DEFAULT_PRIORITY,
        off_state: str | None = DEFAULT_OFF_STATE,
        value_column: str = "value_bool",
        time_column: str = "systime",
    ) -> None:
        if not lamp_map:
            raise ValueError("lamp_map must list at least one signal uuid")
        require_columns(
            dataframe,
            [time_column, "uuid", value_column],
            where="SignalTowerDecoder",
        )
        super().__init__(dataframe, time_column=time_column)
        self.lamp_map: dict[str, str] = dict(lamp_map)
        self.event_uuid = event_uuid
        self.priority = tuple(priority)
        self.off_state = off_state
        self.value_column = value_column
        self.time_column = time_column

    # ------------------------------------------------------------------ API

    def decode(self) -> pd.DataFrame:
        """Return a long-format, change-encoded signal of the decoded state.

        Returns:
            DataFrame with columns ``systime``, ``uuid`` (``event_uuid``),
            ``value_string`` (the state label), ``is_delta`` (True). One row
            per *change* of the decoded state, in time order. An empty input
            or a frame holding no mapped lamp returns an empty frame with
            the same schema.
        """
        if "uuid" not in self.dataframe.columns or self.dataframe.empty:
            return self._empty_result()

        wanted = set(self.lamp_map)
        lamp_frame = self.dataframe[self.dataframe["uuid"].isin(wanted)].copy()
        if lamp_frame.empty or self.value_column not in lamp_frame.columns:
            return self._empty_result()

        lamp_frame = lamp_frame.sort_values(self.time_column)
        lamp_frame[self.time_column] = pd.to_datetime(lamp_frame[self.time_column])
        lamp_frame["_on"] = bool_or_false(lamp_frame[self.value_column])

        # Pivot each lamp's state onto a shared time axis via forward-fill.
        # A lamp whose first reading is in the middle of the window counts as
        # off before that reading.
        wide = (
            lamp_frame.pivot_table(
                index=self.time_column,
                columns="uuid",
                values="_on",
                aggfunc="last",
                dropna=False,
            )
            .sort_index()
            .ffill()
            .fillna(False)
            .astype(bool)
        )

        # Resolve each row to a state by priority order; unmapped states keep
        # their declared order below the known priorities.
        rank: dict[str, int] = {}
        for i, state in enumerate(self.priority):
            rank.setdefault(state, i)
        next_rank = len(rank)
        for state in self.lamp_map.values():
            if state not in rank:
                rank[state] = next_rank
                next_rank += 1
        # Group lamps per state and OR them, so two "warning" lamps collapse.
        per_state: dict[str, pd.Series] = {}
        for lamp, state in self.lamp_map.items():
            if lamp not in wide.columns:
                continue
            lamp_on = wide[lamp]
            if state in per_state:
                per_state[state] = per_state[state] | lamp_on
            else:
                per_state[state] = lamp_on.copy()
        if not per_state:
            return self._empty_result()

        # Highest-priority state that is on; falls back to ``off_state``.
        ordered_states = sorted(per_state, key=lambda s: rank[s])
        state_frame = pd.DataFrame(
            {s: per_state[s].to_numpy() for s in ordered_states},
            index=wide.index,
        )
        resolved = pd.Series(index=wide.index, dtype=object)
        for state in ordered_states:
            mask = state_frame[state] & resolved.isna()
            resolved = resolved.mask(mask, state)
        if self.off_state is not None:
            resolved = resolved.fillna(self.off_state)
        else:
            resolved = resolved.ffill()

        # Change-encode: keep the first row and any row whose state differs
        # from the previous row's.
        changed = resolved.ne(resolved.shift())
        changed.iloc[0] = True
        if resolved.iloc[0] is None or (pd.isna(resolved.iloc[0]) if self.off_state is None else False):
            # ``off_state=None`` and first row has no on lamp: no state yet.
            changed.iloc[0] = False
        times = resolved.index[changed]
        # Preserve tz: pivot via a DatetimeIndex keeps the tz on ``wide.index``;
        # convert back using pd.Index to retain either naive or tz-aware kind.
        states = resolved[changed]
        out = pd.DataFrame(
            {
                "systime": pd.DatetimeIndex(times),
                "uuid": self.event_uuid,
                "value_string": states.astype(object).to_numpy(),
                "is_delta": True,
            }
        )
        out = out.dropna(subset=["value_string"]).reset_index(drop=True)
        return out[["systime", "uuid", "value_string", "is_delta"]]

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        """A canonical empty result frame with proper dtypes."""
        return pd.DataFrame(
            {
                "systime": pd.Series(dtype="datetime64[ns]"),
                "uuid": pd.Series(dtype=object),
                "value_string": pd.Series(dtype=object),
                "is_delta": pd.Series(dtype=bool),
            }
        )

    def to_long_format(self) -> pd.DataFrame:
        """Alias for :meth:`decode` — the ts-shape idiom for transforms."""
        return self.decode()
