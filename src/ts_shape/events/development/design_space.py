"""Multivariate design-space qualification and monitoring.

In process development the *design space* is the multidimensional region
of input parameters that has been demonstrated -- through DOE or process
characterisation runs -- to deliver acceptable output quality. ICH Q8
formalises the concept for pharma; it is equally useful in any process
industry where multiple critical process parameters (CPPs) interact.

This detector fits a design space to qualification data and emits events
when commercial-operation data exits the qualified region. Two fitting
modes are supported: per-axis quantile boxes (cheap, conservative) and a
scipy convex hull (tight, captures correlation between factors).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class DesignSpaceEvents(Base):
    """Fit and monitor a multivariate qualified operating window.

    Workflow::

        ds = DesignSpaceEvents(qualification_df, cpp_uuids=[...])
        ds.fit_box()                    # or ds.fit_hull()
        excursions = ds.detect_excursions(operation_df)
        near = ds.boundary_proximity(operation_df, warn_margin=0.1)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        cpp_uuids: list[str],
        *,
        event_uuid: str = "dev:design_space",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        if len(cpp_uuids) < 2:
            raise ValueError(
                "cpp_uuids must contain at least two signals; "
                "use ProcessWindowEvents for the univariate case"
            )
        self.cpp_uuids = list(cpp_uuids)
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        for uid in self.cpp_uuids:
            self._validate_uuid(self.dataframe, uid)

        self._qualification = self._wide_frame(self.dataframe)
        self._box: dict[str, tuple[float, float]] | None = None
        self._hull = None  # scipy.spatial.ConvexHull when fitted
        self._hull_points: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wide_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=self.cpp_uuids)
        sel = df[df["uuid"].isin(self.cpp_uuids)]
        if sel.empty:
            return pd.DataFrame(columns=self.cpp_uuids)
        wide = (
            sel.pivot_table(
                index=self.time_column,
                columns="uuid",
                values=self.value_column,
                aggfunc="last",
            )
            .sort_index()
            .ffill()
            .dropna(how="any")
        )
        present = [u for u in self.cpp_uuids if u in wide.columns]
        return wide[present] if present else pd.DataFrame(columns=self.cpp_uuids)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_box(
        self, quantiles: tuple[float, float] = (0.05, 0.95)
    ) -> DesignSpaceEvents:
        """Fit per-axis bounds from the qualification data.

        Args:
            quantiles: ``(low, high)`` quantile pair used to set each
                axis's bounds. ``(0.0, 1.0)`` reduces to absolute min/max.

        Returns:
            ``self`` to allow ``DesignSpaceEvents(...).fit_box()`` chaining.
        """
        if self._qualification.empty:
            raise ValueError("No qualification data available to fit_box on.")
        low_q, high_q = quantiles
        self._box = {
            col: (
                float(self._qualification[col].quantile(low_q)),
                float(self._qualification[col].quantile(high_q)),
            )
            for col in self._qualification.columns
        }
        self._hull = None
        self._hull_points = None
        return self

    def fit_hull(self) -> DesignSpaceEvents:
        """Fit a convex hull around the qualification data.

        Requires :mod:`scipy.spatial.ConvexHull`. The hull captures
        correlation between CPPs that a per-axis box cannot, at the cost
        of needing ``len(cpps) + 1`` non-degenerate qualification points.

        Returns:
            ``self`` for chaining.
        """
        from scipy.spatial import ConvexHull  # local import: optional scipy path

        if len(self._qualification) < len(self.cpp_uuids) + 1:
            raise ValueError(
                f"fit_hull needs at least {len(self.cpp_uuids) + 1} qualification "
                f"points in {len(self.cpp_uuids)} dimensions; "
                f"got {len(self._qualification)}."
            )
        pts = self._qualification.to_numpy(dtype=float)
        self._hull = ConvexHull(pts)
        self._hull_points = pts
        self._box = None
        return self

    # ------------------------------------------------------------------
    # Membership tests
    # ------------------------------------------------------------------

    def _inside_box(self, wide: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask of samples inside the per-axis box."""
        assert self._box is not None
        inside = np.ones(len(wide), dtype=bool)
        for col, (lo, hi) in self._box.items():
            if col not in wide.columns:
                continue
            v = wide[col].to_numpy(dtype=float)
            inside &= (v >= lo) & (v <= hi)
        return inside

    def _inside_hull(self, wide: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask of points inside the convex hull."""
        assert self._hull is not None and self._hull_points is not None
        pts = wide.to_numpy(dtype=float)
        eqs = self._hull.equations  # shape (n_facets, n_dim + 1)
        # Each facet inequality is A·x + b ≤ 0. Hull interior: all true.
        # Small positive tolerance keeps boundary points inside.
        tol = 1e-9
        return np.all(eqs[:, :-1] @ pts.T + eqs[:, -1:] <= tol, axis=0)

    # ------------------------------------------------------------------
    # Public detectors
    # ------------------------------------------------------------------

    def detect_excursions(self, operation_df: pd.DataFrame) -> pd.DataFrame:
        """Find contiguous intervals where operation leaves the design space.

        Args:
            operation_df: Long-form signal DataFrame containing all CPP
                signals; only the configured ``cpp_uuids`` are used.

        Returns:
            Interval-shape DataFrame with columns: ``start``, ``end``,
            ``duration_seconds``, ``uuid``, ``excursion_mode``
            (``"box"`` / ``"hull"``).
        """
        cols = ["start", "end", "duration_seconds", "uuid", "excursion_mode"]
        if self._box is None and self._hull is None:
            raise RuntimeError("Call fit_box() or fit_hull() before detect_excursions.")
        wide = self._wide_frame(operation_df)
        if wide.empty:
            return pd.DataFrame(columns=cols)

        if self._box is not None:
            inside = self._inside_box(wide)
            mode = "box"
        else:
            inside = self._inside_hull(wide)
            mode = "hull"

        outside = ~inside
        if not outside.any():
            return pd.DataFrame(columns=cols)

        # Group contiguous outside runs.
        group_id = (outside != np.roll(outside, 1)).cumsum()
        events: list[dict[str, Any]] = []
        for gid, grp in pd.DataFrame({"out": outside, "gid": group_id}).groupby("gid"):
            if not grp["out"].iloc[0]:
                continue
            idxs = grp.index
            start_t = wide.index[idxs[0]]
            end_t = wide.index[idxs[-1]]
            events.append(
                {
                    "start": start_t,
                    "end": end_t,
                    "duration_seconds": (end_t - start_t).total_seconds(),
                    "uuid": self.event_uuid,
                    "excursion_mode": mode,
                }
            )
        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def boundary_proximity(
        self,
        operation_df: pd.DataFrame,
        warn_margin: float = 0.1,
    ) -> pd.DataFrame:
        """Emit point events for samples within ``warn_margin`` of the boundary.

        Only supported for the ``fit_box`` mode -- a convex hull's
        boundary distance is more expensive to compute per facet and is
        not in scope for this detector. Call :meth:`detect_excursions`
        for hull-fitted spaces.

        Args:
            operation_df: Long-form signal DataFrame.
            warn_margin: Normalised distance threshold (fraction of the
                axis span). A sample is reported when its closest axis
                margin is below this value while still inside the box.

        Returns:
            Point-shape DataFrame: ``systime``, ``uuid``,
            ``signed_margin``, ``closest_axis``.
        """
        cols = ["systime", "uuid", "signed_margin", "closest_axis"]
        if self._box is None:
            raise RuntimeError(
                "boundary_proximity requires fit_box(); hull mode is not supported."
            )
        wide = self._wide_frame(operation_df)
        if wide.empty:
            return pd.DataFrame(columns=cols)

        # Per-axis margins, then identify the closest axis per sample.
        n = len(wide)
        per_axis = np.empty((n, len(self._box)), dtype=float)
        axis_names = list(self._box.keys())
        for i, col in enumerate(axis_names):
            lo, hi = self._box[col]
            v = wide[col].to_numpy(dtype=float)
            span = max(hi - lo, np.finfo(float).eps)
            per_axis[:, i] = np.minimum((v - lo) / span, (hi - v) / span)
        closest_idx = np.argmin(per_axis, axis=1)
        closest_margin = per_axis[np.arange(n), closest_idx]
        # Inside the box (margin >= 0) AND within warn_margin of boundary.
        flag = (closest_margin >= 0) & (closest_margin <= warn_margin)
        if not flag.any():
            return pd.DataFrame(columns=cols)
        rows = {
            "systime": wide.index[flag],
            "uuid": self.event_uuid,
            "signed_margin": closest_margin[flag],
            "closest_axis": [axis_names[i] for i in closest_idx[flag]],
        }
        return pd.DataFrame(rows, columns=cols)
