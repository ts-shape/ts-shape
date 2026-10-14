"""Class and pack docstrings must list what really exists.

* A class docstring with a ``Methods:`` section lists exactly the class's
  public methods (no stale entries, nothing missing).
* A pack ``__init__`` docstring that lists ``- ClassName: ...`` entries
  lists classes importable from that pack, and every ``  - method: ...``
  sub-entry is a public method of that class.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import pytest

import ts_shape

_ENTRY = re.compile(r"^\s*-\s+([A-Za-z_]\w*)\s*:", re.M)
_CLASS_ENTRY = re.compile(r"^- ([A-Z]\w+)\s*:", re.M)
_METHOD_ENTRY = re.compile(r"^\s{2,}- ([a-z_]\w*)\s*:", re.M)


def _modules():
    for info in pkgutil.walk_packages(ts_shape.__path__, "ts_shape."):
        try:
            yield importlib.import_module(info.name)
        except ImportError:  # optional extras
            continue


MODULES = list(_modules())


def _public_methods(cls: type) -> set[str]:
    return {
        name
        for name, member in vars(cls).items()
        if not name.startswith("_")
        and (
            inspect.isfunction(member)
            or isinstance(member, (classmethod, staticmethod, property))
        )
    }


def _documented_classes():
    seen = set()
    for mod in MODULES:
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__ or cls in seen:
                continue
            seen.add(cls)
            doc = cls.__doc__ or ""
            if re.search(r"^\s*Methods:\s*$", doc, re.M):
                yield cls


CLASSES = list(_documented_classes())


@pytest.mark.parametrize("cls", CLASSES, ids=[c.__name__ for c in CLASSES])
def test_methods_section_matches_class(cls):
    section = re.split(r"^\s*Methods:\s*$", cls.__doc__ or "", flags=re.M)[1]
    listed = set(_ENTRY.findall(section))
    actual = _public_methods(cls)
    inherited = {n for n in listed if hasattr(cls, n)}
    assert not (listed - actual - inherited), (
        f"{cls.__name__} docstring lists missing methods: "
        f"{sorted(listed - actual - inherited)}"
    )
    assert not (
        actual - listed
    ), f"{cls.__name__} docstring does not list: {sorted(actual - listed)}"


PACKS = [
    m
    for m in MODULES
    if m.__name__.endswith(
        tuple(
            f".{p}"
            for p in (
                "production",
                "quality",
                "engineering",
                "maintenance",
                "energy",
                "supplychain",
                "correlation",
                "development",
            )
        )
    )
    and hasattr(m, "__path__")
    and _CLASS_ENTRY.search(m.__doc__ or "")
]


@pytest.mark.parametrize("pack", PACKS, ids=[p.__name__ for p in PACKS])
def test_pack_docstring_lists_importable_classes(pack):
    doc = pack.__doc__ or ""
    problems = []
    blocks = re.split(r"^(?=- [A-Z]\w+\s*:)", doc, flags=re.M)
    for block in blocks:
        m = _CLASS_ENTRY.match(block)
        if not m:
            continue
        name = m.group(1)
        cls = getattr(pack, name, None)
        if cls is None:
            problems.append(f"{name} is listed but not importable from the pack")
            continue
        for meth in _METHOD_ENTRY.findall(block):
            if not hasattr(cls, meth):
                problems.append(f"{name}.{meth} is listed but does not exist")
        missing = _public_methods(cls) - set(_METHOD_ENTRY.findall(block))
        if _METHOD_ENTRY.search(block) and missing:
            problems.append(f"{name} entry does not list {sorted(missing)}")
    listed = set(_CLASS_ENTRY.findall(doc))
    listed_objs = {id(getattr(pack, n, None)) for n in listed}
    exported = {  # aliases of a listed class need no entry of their own
        n
        for n in getattr(pack, "__all__", [])
        if n[:1].isupper()
        and id(getattr(pack, n, None)) not in listed_objs
        and not hasattr(getattr(pack, n, None), "__deprecated_alias_of__")
    }
    if exported - listed:
        problems.append(f"exported but not listed: {sorted(exported - listed)}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "pack",
    [
        "ts_shape.events.quality",
        "ts_shape.features",
        "ts_shape.transform",
        "ts_shape.loader",
    ],
)
def test_pack_exports_resolve(pack):
    """Every name a pack exports is importable from it (lazy exports)."""
    mod = importlib.import_module(pack)
    for name in mod.__all__:
        try:
            obj = getattr(mod, name)
        except ImportError as exc:  # optional extra not installed
            if "ts_shape" in str(exc):
                raise
            continue
        assert obj.__name__ == name
        if hasattr(obj, "__deprecated_alias_of__"):
            continue  # old name kept for compatibility, not documented
        assert name in (
            mod.__doc__ or ""
        ), f"{pack} exports {name} but its docstring does not mention it"
