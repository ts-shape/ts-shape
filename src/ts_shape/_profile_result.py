"""Result types of the mapping-free profilers.

:class:`ProfileResult` holds every table of one profiler run plus a summary
of plain numbers (JSON-able); :class:`UnitProfilesResult` holds the results
of several profilers run on the same signals (see
:func:`ts_shape.unit_profiles.profile_units`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd

__all__ = ["ProfileResult", "UnitProfilesResult"]


@dataclass
class ProfileResult:
    """Every table of one profiler run and a summary of plain numbers.

    Attributes:
        tables: Name -> table (one CSV each when saved).
        summary: Plain numbers (counts, seconds, ratios), JSON-able; every
            profiler includes ``evaluable`` and ``reason``.
        kind: The profiler (``"cycles"``, ``"stops"``, ``"changeovers"`` ...).
    """

    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any] = field(default_factory=dict)
    kind: str = "profile"

    def save(self, directory: str | Path) -> list[Path]:
        """Write ``summary.json`` and one CSV per table into ``directory``.

        Args:
            directory: Target folder; created when missing.

        Returns:
            The written files.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written = [out / "summary.json"]
        written[0].write_text(
            json.dumps(self.summary, indent=1, allow_nan=False, default=str),
            encoding="utf-8",
        )
        for name, table in self.tables.items():
            path = out / f"{name}.csv"
            table.to_csv(path, index=False)
            written.append(path)
        return written

    @classmethod
    def load(cls, directory: str | Path) -> Self:
        """Read a result written by :meth:`save` (for dashboards).

        Args:
            directory: Folder written by :meth:`save`.

        Returns:
            The result with its summary and tables.
        """
        path = Path(directory)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        tables = {p.stem: pd.read_csv(p) for p in sorted(path.glob("*.csv"))}
        return cls(tables=tables, summary=summary)


@dataclass
class UnitProfilesResult:
    """The results of several profilers on one machine.

    Attributes:
        results: Profiler name (``"cycles"``, ``"stops"`` ...) -> result.
        summary: Per profiler: ``evaluable``, ``reason`` and its headline
            numbers (JSON-able).
    """

    results: dict[str, ProfileResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> ProfileResult:
        return self.results[name]

    def save(self, directory: str | Path) -> list[Path]:
        """Write ``summary.json`` and one subfolder per profiler.

        Args:
            directory: Target folder; created when missing.

        Returns:
            The written files.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written = [out / "summary.json"]
        written[0].write_text(
            json.dumps(self.summary, indent=1, allow_nan=False, default=str),
            encoding="utf-8",
        )
        for name, result in self.results.items():
            written.extend(result.save(out / name))
        return written

    @classmethod
    def load(cls, directory: str | Path) -> UnitProfilesResult:
        """Read a result written by :meth:`save`.

        Args:
            directory: Folder written by :meth:`save`.

        Returns:
            The results of every saved profiler.
        """
        path = Path(directory)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        results = {
            sub.name: ProfileResult.load(sub)
            for sub in sorted(path.iterdir())
            if sub.is_dir() and (sub / "summary.json").exists()
        }
        for name, result in results.items():
            result.kind = name
        return cls(results=results, summary=summary)
