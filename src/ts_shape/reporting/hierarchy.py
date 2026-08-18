"""ISA-95 style asset hierarchy and roll-up along it.

A global report is a walk up one tree: work unit -> line -> area -> site ->
region -> enterprise. :class:`AssetHierarchy` holds that tree as a mapping
frame (one row per leaf, one column per level, ordered coarse to fine),
validates it, attaches ancestry to a KPI frame, and aggregates to any level
using :mod:`ts_shape.reporting.aggregation` semantics.

The validation matters as much as the rollup: a line assigned to two sites
double-counts the enterprise total, and a NaN site quietly drops a line from
every regional view.

Example::

    from ts_shape.reporting import AssetHierarchy, MetricSpec

    hierarchy = AssetHierarchy(
        mapping_df, levels=["region", "site", "area", "line"]
    )
    hierarchy.validate()                 # empty frame == clean master data

    by_site = hierarchy.roll_up(
        kpi_df,
        to_level="site",
        specs=[MetricSpec.sum("parts"),
               MetricSpec.weighted("oee_pct", weight="planned_minutes")],
        group_extra=["period"],
    )
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from ts_shape.reporting.aggregation import MetricSpec, aggregate_metrics

__all__ = ["AssetHierarchy"]

logger = logging.getLogger(__name__)


class AssetHierarchy:
    """A validated equipment hierarchy plus roll-up along its levels.

    Args:
        mapping: One row per leaf node, with one column per level.
        levels: Level column names ordered **coarse to fine**, e.g.
            ``["region", "site", "area", "line"]``. The last entry is the leaf
            level and the default join key.
        drop_duplicates: Collapse identical rows in ``mapping``. Left on by
            default so a mapping exported per shift still describes one tree.

    Raises:
        ValueError: If fewer than one level is given or a level column is
            absent from ``mapping``.
    """

    def __init__(
        self,
        mapping: pd.DataFrame,
        *,
        levels: Sequence[str],
        drop_duplicates: bool = True,
    ) -> None:
        if not isinstance(mapping, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(mapping).__name__}")
        if not levels:
            raise ValueError("AssetHierarchy requires at least one level")
        missing = [level for level in levels if level not in mapping.columns]
        if missing:
            raise ValueError(
                f"Level column(s) {missing} not found in mapping. "
                f"Available: {list(mapping.columns)}"
            )
        self._levels: list[str] = list(levels)
        frame = mapping.loc[:, list(levels)].copy()
        if drop_duplicates:
            frame = frame.drop_duplicates()
        self._mapping = frame.reset_index(drop=True)

    def __repr__(self) -> str:
        return f"AssetHierarchy(levels={self._levels}, " f"nodes={len(self._mapping)})"

    # -- structure -------------------------------------------------------

    @property
    def levels(self) -> list[str]:
        """Level names, coarse to fine."""
        return list(self._levels)

    @property
    def leaf_level(self) -> str:
        """The finest level -- the default key a KPI frame is joined on."""
        return self._levels[-1]

    def to_frame(self) -> pd.DataFrame:
        """The mapping as a DataFrame, one row per leaf."""
        return self._mapping.copy()

    def _level_index(self, level: str) -> int:
        try:
            return self._levels.index(level)
        except ValueError:
            raise ValueError(
                f"Unknown level {level!r}. Levels: {self._levels}"
            ) from None

    def levels_up_to(self, level: str) -> list[str]:
        """Levels from the coarsest down to and including ``level``."""
        return self._levels[: self._level_index(level) + 1]

    def nodes(self, level: str) -> list[str]:
        """Sorted distinct node names at ``level``."""
        series = self._mapping[level].dropna()
        return sorted({str(v) for v in series})

    def children(self, node: str, *, level: str, child_level: str) -> list[str]:
        """Nodes at ``child_level`` that sit under ``node``.

        Raises:
            ValueError: If ``child_level`` is not finer than ``level``.
        """
        if self._level_index(child_level) <= self._level_index(level):
            raise ValueError(
                f"child_level {child_level!r} must be finer than level {level!r}"
            )
        subset = self._mapping[self._mapping[level].astype(str) == str(node)]
        return sorted({str(v) for v in subset[child_level].dropna()})

    def parent(self, node: str, *, level: str, parent_level: str) -> str | None:
        """The ``parent_level`` ancestor of ``node``.

        Returns:
            The ancestor name, or ``None`` when the node is unknown. If the
            node has conflicting ancestry the first is returned and a warning
            is logged -- :meth:`validate` reports it as ``multiple_parents``.
        """
        if self._level_index(parent_level) >= self._level_index(level):
            raise ValueError(
                f"parent_level {parent_level!r} must be coarser than level {level!r}"
            )
        subset = self._mapping[self._mapping[level].astype(str) == str(node)]
        parents = {str(v) for v in subset[parent_level].dropna()}
        if not parents:
            return None
        if len(parents) > 1:
            logger.warning(
                "Node %r at level %r has %d parents at level %r: %s",
                node,
                level,
                len(parents),
                parent_level,
                sorted(parents),
            )
        return sorted(parents)[0]

    def path(self, node: str, *, level: str | None = None) -> dict[str, str | None]:
        """Full ancestry of ``node`` as ``{level: ancestor}``."""
        level = level or self.leaf_level
        subset = self._mapping[self._mapping[level].astype(str) == str(node)]
        if subset.empty:
            return {lvl: None for lvl in self._levels}
        row = subset.iloc[0]
        return {
            lvl: (None if pd.isna(row[lvl]) else str(row[lvl])) for lvl in self._levels
        }

    # -- validation ------------------------------------------------------

    def validate(self) -> pd.DataFrame:
        """Return structural problems in the hierarchy, one row per issue.

        Checks:

        * ``missing_parent`` -- a node whose ancestor at a coarser level is
          NaN or empty. It would be grouped under NaN in every rollup.
        * ``multiple_parents`` -- a node mapped to more than one ancestor at a
          coarser level. Its output is counted once per parent, inflating the
          enterprise total.
        * ``duplicate_leaf`` -- the same leaf appearing with differing ancestry
          (the row-level view of ``multiple_parents``).

        Returns:
            DataFrame with columns ``level``, ``node``, ``issue``, ``detail``.
            Empty (with those columns) when the hierarchy is sound.
        """
        columns = ["level", "node", "issue", "detail"]
        rows: list[dict[str, str]] = []

        for depth, level in enumerate(self._levels):
            blank = self._mapping[level].isna() | (
                self._mapping[level].astype(str).str.strip() == ""
            )
            for _, row in self._mapping[blank].iterrows():
                leaf = row[self.leaf_level]
                rows.append(
                    {
                        "level": level,
                        "node": "" if pd.isna(leaf) else str(leaf),
                        "issue": "missing_parent",
                        "detail": f"no {level} assigned",
                    }
                )
            if depth == 0:
                continue
            for parent_level in self._levels[:depth]:
                counts = (
                    self._mapping.dropna(subset=[level, parent_level])
                    .groupby(self._mapping[level].astype(str))[parent_level]
                    .nunique()
                )
                for node, n_parents in counts[counts > 1].items():
                    parents = sorted(
                        {
                            str(v)
                            for v in self._mapping.loc[
                                self._mapping[level].astype(str) == str(node),
                                parent_level,
                            ].dropna()
                        }
                    )
                    rows.append(
                        {
                            "level": level,
                            "node": str(node),
                            "issue": "multiple_parents",
                            "detail": (
                                f"{n_parents} distinct {parent_level} values: "
                                f"{parents}"
                            ),
                        }
                    )

        leaf = self.leaf_level
        duplicated = self._mapping[self._mapping.duplicated(subset=[leaf], keep=False)]
        for node, group in duplicated.groupby(duplicated[leaf].astype(str)):
            if len(group.drop_duplicates()) > 1:
                rows.append(
                    {
                        "level": leaf,
                        "node": str(node),
                        "issue": "duplicate_leaf",
                        "detail": f"{len(group)} mapping rows with differing ancestry",
                    }
                )

        if not rows:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(rows, columns=columns)
            .drop_duplicates()
            .sort_values(["level", "node", "issue"])
            .reset_index(drop=True)
        )

    # -- use -------------------------------------------------------------

    def enrich(
        self,
        dataframe: pd.DataFrame,
        *,
        on: str | None = None,
        levels: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Attach ancestor level columns to a KPI frame.

        Args:
            dataframe: KPI frame carrying the join key.
            on: Column in ``dataframe`` holding the node name and, unless
                ``levels`` says otherwise, the hierarchy level to join at.
                Defaults to :attr:`leaf_level`.
            levels: Ancestor levels to attach. Defaults to every level coarser
                than the join level.

        Returns:
            A copy of ``dataframe`` with the ancestor columns added. Nodes
            absent from the hierarchy keep NaN ancestors rather than being
            dropped, so unmapped assets stay visible in the report.

        Raises:
            ValueError: If the join column is missing from ``dataframe``.
        """
        key = on or self.leaf_level
        if key not in dataframe.columns:
            raise ValueError(
                f"Join column {key!r} not found. Available: {list(dataframe.columns)}"
            )
        join_level = key if key in self._levels else self.leaf_level
        wanted = (
            list(levels)
            if levels is not None
            else self._levels[: self._level_index(join_level)]
        )
        unknown = [lvl for lvl in wanted if lvl not in self._levels]
        if unknown:
            raise ValueError(f"Unknown level(s) {unknown}. Levels: {self._levels}")
        if key == join_level:
            # The join column already *is* that level; re-attaching it would
            # duplicate the column.
            wanted = [lvl for lvl in wanted if lvl != join_level]
        if not wanted:
            return dataframe.copy()

        lookup = (
            self._mapping[[join_level, *wanted]]
            .dropna(subset=[join_level])
            .drop_duplicates(subset=[join_level])
        )
        lookup = lookup.assign(**{join_level: lookup[join_level].astype(str)})

        result = dataframe.copy()
        merged = result.merge(
            lookup,
            how="left",
            left_on=result[key].astype(str),
            right_on=lookup[join_level],
            suffixes=("", "_hierarchy"),
        )
        drop = [c for c in ("key_0", f"{join_level}_hierarchy") if c in merged.columns]
        if join_level != key and join_level in merged.columns:
            drop.append(join_level)
        return merged.drop(columns=drop, errors="ignore")

    def roll_up(
        self,
        dataframe: pd.DataFrame,
        *,
        to_level: str,
        specs: Sequence[MetricSpec],
        on: str | None = None,
        group_extra: Sequence[str] = (),
        contributor_column: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate a KPI frame up to ``to_level``.

        Args:
            dataframe: KPI frame at (or below) the join level.
            to_level: Hierarchy level to aggregate to. The output is grouped by
                every level from the coarsest down to this one, so a site
                rollup carries its region.
            specs: Aggregation semantics per metric. See
                :class:`~ts_shape.reporting.aggregation.MetricSpec`.
            on: Join column, defaulting to :attr:`leaf_level`.
            group_extra: Additional grouping columns kept alongside the levels
                -- typically ``["period"]``.
            contributor_column: Column whose distinct count per group is
                reported as ``contributors`` (defaults to the join column, so
                a site rollup shows how many lines fed it).

        Returns:
            One row per (levels up to ``to_level``, ``group_extra``) group.

        Raises:
            ValueError: If ``to_level`` is unknown or a column is missing.
        """
        key = on or self.leaf_level
        group_levels = self.levels_up_to(to_level)
        enriched = self.enrich(dataframe, on=key, levels=group_levels)

        # When the rollup target *is* the join level, the level column is the
        # join column itself and needs no ancestor copy.
        group_cols: list[str] = []
        for level in group_levels:
            if level == key or level in enriched.columns:
                group_cols.append(level)
            else:
                raise ValueError(
                    f"Level {level!r} is not available after enrichment; "
                    f"columns: {list(enriched.columns)}"
                )
        group_cols.extend(group_extra)

        return aggregate_metrics(
            enriched,
            by=group_cols,
            specs=specs,
            contributor_column=contributor_column or key,
        )
