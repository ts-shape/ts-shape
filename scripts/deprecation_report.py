"""List every deprecated name in ts-shape, read from the code itself.

Run from the repository root::

    python scripts/deprecation_report.py            # counts per kind
    python scripts/deprecation_report.py --write    # also writes docs/deprecations.md

``scripts/gen_ref_pages.py`` writes the page on every docs build. Nothing is
maintained by hand: each deprecation helper in ``ts_shape.utils.deprecation``
(and ``twin_columns`` in ``ts_shape.events._output``) leaves a marker that is
read here:

========================  =====================================================
kind                      marker
========================  =====================================================
class alias               ``__deprecated_alias_of__`` on a class
function alias            ``__deprecated_alias_of__`` on a function
method alias              a ``deprecated_method`` descriptor
renamed keyword           ``__deprecated_kwargs__`` (``renamed_kwargs``)
keyword-only parameters   ``__keyword_only__`` (``keyword_only``)
renamed attribute         a ``deprecated_attribute`` descriptor
legacy twin column        ``__twin_columns__`` (``twin_columns``)
moved module              a module that is only a ``moved_module`` shim
moved name                a package's ``_MOVED`` map (``moved_attribute``)
========================  =====================================================
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import re
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ts_shape"

# Report on the sources next to this script, not on whatever ts_shape is
# installed: the release job builds older docs versions from their own tags.
sys.path.insert(0, str(SRC.parent))

import ts_shape  # noqa: E402
from ts_shape.utils.deprecation import deprecated_attribute  # noqa: E402

OUT = ROOT / "docs" / "deprecations.md"
SHIM_RE = re.compile(r'moved_module\(\s*"([\w.]+)",\s*"([\w.]+)",?\s*\)')

KINDS = {
    "classes": ("Class aliases", ("Old class", "Use instead")),
    "functions": ("Function aliases", ("Old function", "Use instead")),
    "methods": ("Method aliases", ("Class", "Old method", "Use instead")),
    "keywords": ("Renamed keyword arguments", ("Method", "Old keyword", "Use instead")),
    "keyword_only": (
        "Arguments to pass by keyword",
        ("Method", "Pass by keyword (positional use is deprecated)"),
    ),
    "attributes": ("Renamed attributes", ("Class", "Old attribute", "Use instead")),
    "columns": (
        "Legacy twin columns (no warning possible)",
        ("Method", "Legacy column", "Canonical column"),
    ),
    "modules": ("Moved modules", ("Old module", "New module")),
    "moved_names": ("Moved names", ("Old name", "New place")),
}


@dataclass
class Report:
    rows: dict[str, list[tuple[str, ...]]] = field(
        default_factory=lambda: {k: [] for k in KINDS}
    )

    def add(self, kind: str, *cells: str) -> None:
        self.rows[kind].append(cells)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.rows.items()}

    def total(self) -> int:
        return sum(self.counts().values())


def _code(text: str) -> str:
    return f"`{text}`"


def _public_objects() -> Iterator[tuple[str, object]]:
    """``(qualified name, object)`` for every name ``ts_shape`` exports."""
    names = sorted(set(ts_shape._LAZY))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # resolving deprecated aliases warns
        for name in names:
            try:
                yield name, getattr(ts_shape, name)
            except ImportError:  # an optional extra is not installed
                continue


def _callable_markers(report: Report, label: str, func: object) -> None:
    func = getattr(func, "__func__", func)  # classmethod / staticmethod
    for old, new in sorted(getattr(func, "__deprecated_kwargs__", {}).items()):
        report.add("keywords", _code(label), _code(old), _code(new))
    movable = getattr(func, "__keyword_only__", ())
    if movable:
        report.add("keyword_only", _code(label), ", ".join(_code(m) for m in movable))
    for old, new in sorted(getattr(func, "__twin_columns__", {}).items()):
        report.add("columns", _code(label), _code(old), _code(new))


def _class_markers(report: Report, cls: type) -> None:
    for attr, raw in sorted(vars(cls).items()):
        if type(raw).__name__ == "_Forward":
            new = getattr(cls, attr).__deprecated_alias_of__
            report.add("methods", _code(cls.__name__), _code(attr), _code(new))
        elif isinstance(raw, deprecated_attribute):
            report.add(
                "attributes", _code(cls.__name__), _code(attr), _code(raw.new_name)
            )
        elif callable(raw) or isinstance(raw, (classmethod, staticmethod)):
            _callable_markers(report, f"{cls.__name__}.{attr}", raw)


def collect() -> Report:
    report = Report()
    seen: set[int] = set()
    for name, obj in _public_objects():
        alias_of = (
            vars(obj).get("__deprecated_alias_of__")
            if inspect.isclass(obj)
            else (getattr(obj, "__deprecated_alias_of__", None))
        )
        if alias_of is not None:
            kind = "classes" if inspect.isclass(obj) else "functions"
            report.add(
                kind, _code(name), _code(getattr(alias_of, "__name__", alias_of))
            )
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if inspect.isclass(obj):
            _class_markers(report, obj)
        elif callable(obj):
            _callable_markers(report, name, obj)

    for path in sorted(SRC.rglob("*.py")):
        match = SHIM_RE.search(path.read_text(encoding="utf-8"))
        if match and "def moved_module" not in path.read_text(encoding="utf-8"):
            report.add("modules", _code(match.group(1)), _code(match.group(2)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(ts_shape.__path__, "ts_shape."):
            if not info.ispkg:
                continue
            try:
                module = importlib.import_module(info.name)
            except ImportError:
                continue
            for old, new in sorted(vars(module).get("_MOVED", {}).items()):
                report.add(
                    "moved_names", _code(f"{info.name}.{old}"), _code(f"{new}.{old}")
                )

    for rows in report.rows.values():
        rows.sort()
    return report


def render(report: Report) -> str:
    lines = [
        "---",
        "title: Deprecated names",
        "description: Every deprecated name in ts-shape and what to use instead, "
        "generated from the code.",
        "---",
        "",
        "# Deprecated names",
        "",
        "Every name, argument and column below still works and (except the twin",
        "columns) emits a `DeprecationWarning` naming the replacement. They date",
        "from 0.0.63 (October 2026) or later and are removed in 1.0, following the",
        "[deprecation policy](naming.md#deprecation-policy). The list is generated",
        "from the code on every docs build (`scripts/deprecation_report.py`).",
        "",
        "To find what your own code still uses, run it or its tests with",
        "`python -W error::DeprecationWarning`.",
        "",
        "| Kind | Count |",
        "|---|---|",
    ]
    for kind, (title, _) in KINDS.items():
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        lines.append(f"| [{title}](#{anchor}) | {len(report.rows[kind])} |")
    lines.append(f"| **Total** | **{report.total()}** |")
    for kind, (title, header) in KINDS.items():
        lines += ["", f"## {title}", ""]
        rows = report.rows[kind]
        if not rows:
            lines.append("None.")
            continue
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write(path: Path = OUT) -> Report:
    report = collect()
    path.write_text(render(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help=f"write {OUT}")
    args = parser.parse_args()
    report = write() if args.write else collect()
    width = max(len(title) for title, _ in KINDS.values())
    for kind, (title, _) in KINDS.items():
        print(f"{title:<{width}}  {len(report.rows[kind]):>4}")
    print(f"{'Total':<{width}}  {report.total():>4}")


if __name__ == "__main__":
    main()
