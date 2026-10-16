"""Keep each lazy ``__init__``'s ``if TYPE_CHECKING:`` imports in sync with its map.

A package that re-exports names lazily (PEP 562 ``__getattr__``) also imports
them under ``if TYPE_CHECKING:``, so that type checkers and IDEs see the real
classes instead of ``object``. A name added to the lazy map without the
matching import (or the other way round) fails here.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import ts_shape

SRC = Path(ts_shape.__file__).resolve().parent


def _lazy_packages() -> list[str]:
    """Every ts_shape package whose ``__init__`` defines ``__getattr__``."""
    names = []
    for init in sorted(SRC.rglob("__init__.py")):
        tree = ast.parse(init.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
            for node in tree.body
        ):
            rel = init.parent.relative_to(SRC.parent)
            names.append(".".join(rel.parts))
    return names


LAZY_PACKAGES = _lazy_packages()


def _type_checking_imports(module_name: str) -> dict[str, str]:
    """``{name: source module}`` imported in the top-level TYPE_CHECKING block."""
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports: dict[str, str] = {}
    for node in tree.body:
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            continue
        for stmt in node.body:
            assert isinstance(stmt, ast.ImportFrom), ast.unparse(stmt)
            for alias in stmt.names:
                # ``import X as X`` marks an explicit re-export for mypy --strict.
                assert alias.asname == alias.name, ast.unparse(stmt)
                assert alias.name not in imports, f"{alias.name} imported twice"
                imports[alias.name] = stmt.module or ""
    return imports


def _lazy_map(module_name: str) -> dict[str, str]:
    """``{name: defining module}`` that the package's ``__getattr__`` resolves."""
    module = importlib.import_module(module_name)
    if hasattr(module, "_CLASSES"):
        return dict(module._CLASSES)
    return {**module._LAZY, **getattr(module, "_ERRORS", {})}


def test_lazy_packages_found():
    for expected in (
        "ts_shape",
        "ts_shape.context",
        "ts_shape.events",
        "ts_shape.events.quality",
        "ts_shape.features",
        "ts_shape.loader",
        "ts_shape.transform",
    ):
        assert expected in LAZY_PACKAGES


@pytest.mark.parametrize("module_name", LAZY_PACKAGES)
def test_type_checking_imports_mirror_lazy_map(module_name):
    module = importlib.import_module(module_name)
    lazy = _lazy_map(module_name)
    imported = _type_checking_imports(module_name)
    assert sorted(module.__all__) == sorted(lazy)
    missing = sorted(set(lazy) - set(imported))
    extra = sorted(set(imported) - set(lazy))
    assert not missing, f"add to the TYPE_CHECKING block of {module_name}: {missing}"
    assert not extra, f"not in the lazy map of {module_name}: {extra}"
    wrong = {n: (imported[n], lazy[n]) for n in lazy if imported[n] != lazy[n]}
    assert not wrong, f"TYPE_CHECKING import vs lazy map module: {wrong}"


@pytest.mark.parametrize("module_name", [m for m in LAZY_PACKAGES if m != "ts_shape"])
def test_subpackage_exports_are_top_level_exports(module_name):
    """``from ts_shape import X`` works for every class a pack exports."""
    top = _lazy_map("ts_shape")
    for name, target in _lazy_map(module_name).items():
        assert top.get(name) == target, (module_name, name)
