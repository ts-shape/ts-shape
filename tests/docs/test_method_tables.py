"""The method tables of the module pages match the real signatures.

Each page under ``docs/modules`` names its class in the ``# Heading`` and its
module on the line below. Every ``| `method(args)` |`` row must be a complete
call (a tuple default such as ``threshold=(1.5, 1.5)`` must not be cut off at
its comma), name a real method, use only real parameter names, and show the
real default for every literal default it lists. ``...`` stands for omitted
arguments.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted((ROOT / "docs" / "modules").rglob("*.md"))
_CELL = re.compile(r"^\| (`\w+\(.*?) \|", re.M)  # first cell of a method row
_SIG = re.compile(r"`(\w+)\(([^`]*)\)`")


def _rows() -> list[tuple[str, str, str, str, str]]:
    rows = []
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        head = re.search(r"^# (\w+)\s*$", text, re.M)
        module = re.search(r"^`(ts_shape[\w.]+)`", text, re.M)
        if not head or not module:
            continue
        rel = str(page.relative_to(ROOT))
        for cell in _CELL.findall(text):
            for name, args in _SIG.findall(cell):
                rows.append((rel, module.group(1), head.group(1), name, args))
    return rows


ROWS = _rows()


def test_tables_are_discovered():
    assert len(ROWS) > 300


def _same(shown, default) -> bool:
    if shown == default:
        return True
    # a list shown for a tuple default (or the reverse) is the same value
    return (
        isinstance(default, list | tuple)
        and isinstance(shown, list | tuple)
        and (list(shown) == list(default))
    )


@pytest.mark.parametrize(
    "page,module,cls_name,method,args",
    ROWS,
    ids=[f"{r[0]}::{r[3]}" for r in ROWS],
)
def test_method_row_matches_signature(page, module, cls_name, method, args):
    cls = getattr(importlib.import_module(module), cls_name)
    fn = getattr(cls, method, None)
    assert fn is not None, f"{page}: {cls_name} has no method {method}"
    try:
        call = ast.parse(f"f({args})", mode="eval").body
    except SyntaxError:
        # "..." after keyword arguments is not Python; drop it and retry
        call = ast.parse(f"f({args.replace(', ...', '')})", mode="eval").body
    params = inspect.signature(fn).parameters
    for kw in call.keywords:
        assert kw.arg in params, f"{page}: {method} has no parameter {kw.arg!r}"
        default = params[kw.arg].default
        try:
            shown = ast.literal_eval(kw.value)
        except ValueError:
            continue  # a placeholder such as ``value_range=(low, high)``
        if default is inspect.Parameter.empty:
            continue  # a required argument shown with an example value
        assert _same(
            shown, default
        ), f"{page}: {method}({kw.arg}={shown!r}) but the default is {default!r}"
