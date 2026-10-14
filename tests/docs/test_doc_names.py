"""Docs must only show ts_shape code that runs.

Scans the Python code blocks of every Markdown page (docs/ and README.md):

* every ``from ts_shape... import Name`` must resolve, and every
  ``Name.attribute`` used in the same block must exist;
* every call of an imported class, of its methods, or of a method on an
  instance assigned from it (``x = Cls(...); x.method(...)``) must bind
  against the real signature: no unknown keywords, no missing required
  arguments, no extra positional arguments.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted([ROOT / "README.md", *(ROOT / "docs").rglob("*.md")])
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
_IMPORT = re.compile(
    r"^\s*from\s+(ts_shape[\w.]*)\s+import\s+\(?([\w\s,]+?)\)?\s*$", re.M
)


def _references() -> list[tuple[str, str, str, list[str]]]:
    refs = []
    for page in PAGES:
        rel = str(page.relative_to(ROOT))
        for block in _BLOCK.findall(page.read_text(encoding="utf-8")):
            if block.lstrip().startswith("# Before"):
                continue  # migration notes show removed names on purpose
            for module, names in _IMPORT.findall(block):
                for raw in names.split(","):
                    name = raw.split(" as ")[-1].strip()
                    if not name:
                        continue
                    attrs = sorted(set(re.findall(rf"\b{name}\.(\w+)", block)))
                    refs.append((rel, module, raw.split(" as ")[0].strip(), attrs))
    return refs


REFS = _references()


@pytest.mark.parametrize(
    "page,module,name,attrs", REFS, ids=[f"{r[0]}:{r[2]}" for r in REFS]
)
def test_documented_name_exists(page, module, name, attrs):
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # optional extras are fine; wrong paths are not
        if "ts_shape" in str(exc):
            pytest.fail(f"{page}: cannot import {module}: {exc}")
        pytest.skip(f"optional dependency missing: {exc}")
    if not hasattr(mod, name):
        try:
            importlib.import_module(f"{module}.{name}")
            return
        except ImportError:
            pytest.fail(f"{page}: {module} has no {name!r}")
    obj = getattr(mod, name)
    missing = [a for a in attrs if not hasattr(obj, a)]
    assert not missing, f"{page}: {name} has no {missing}"


# ---------------------------------------------------------------------------
# Calls in the docs must bind against the real signatures
# ---------------------------------------------------------------------------


def _import_names(tree: ast.AST) -> dict[str, Any]:
    names: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "ts_shape"
        ):
            try:
                mod = importlib.import_module(node.module or "")
            except ImportError:
                continue
            for alias in node.names:
                obj = getattr(mod, alias.name, None)
                if obj is not None:
                    names[alias.asname or alias.name] = obj
    return names


def _params(fn: Any, drop_first: bool) -> list[inspect.Parameter] | None:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return None
    return params[1:] if drop_first and params else params


def _check_call(call: ast.Call, params: list[inspect.Parameter]) -> list[str]:
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        return []
    kinds = {p.kind for p in params}
    by_name = {p.name: p for p in params}
    problems = []
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if (
        len(call.args) > len(positional)
        and inspect.Parameter.VAR_POSITIONAL not in kinds
    ):
        problems.append(f"{len(call.args)} positional args, accepts {len(positional)}")
    given = {p.name for p in positional[: len(call.args)]}
    for kw in call.keywords:
        if kw.arg not in by_name and inspect.Parameter.VAR_KEYWORD not in kinds:
            problems.append(f"unknown keyword {kw.arg!r}")
        given.add(kw.arg or "")
    missing = [
        p.name
        for p in params
        if p.default is p.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and p.name not in given
    ]
    if missing:
        problems.append(f"missing required {missing}")
    return problems


def _call_problems(block: str) -> list[str]:
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return []
    names = _import_names(tree)
    instances: dict[str, Any] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and inspect.isclass(names.get(node.value.func.id))
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    instances[target.id] = names[node.value.func.id]
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        label, params = None, None
        if isinstance(func, ast.Name) and func.id in names:
            obj = names[func.id]
            label = func.id
            params = _params(obj, drop_first=False)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = names.get(func.value.id) or instances.get(func.value.id)
            if not inspect.isclass(owner) or not hasattr(owner, func.attr):
                continue
            raw = inspect.getattr_static(owner, func.attr)
            label = f"{owner.__name__}.{func.attr}"
            via_instance = func.value.id in instances and func.value.id not in names
            is_plain = inspect.isfunction(raw)
            params = _params(
                getattr(owner, func.attr), drop_first=is_plain and via_instance
            )
        if params is None or label is None:
            continue
        for problem in _check_call(node, params):
            problems.append(f"line {node.lineno}: {label}(): {problem}")
    return problems


CALL_PAGES = [
    (str(p.relative_to(ROOT)), p)
    for p in PAGES
    if "ts_shape" in p.read_text(encoding="utf-8")
]


@pytest.mark.parametrize("rel,page", CALL_PAGES, ids=[c[0] for c in CALL_PAGES])
def test_documented_calls_bind(rel, page):
    problems = []
    for i, block in enumerate(_BLOCK.findall(page.read_text(encoding="utf-8"))):
        if block.lstrip().startswith("# Before"):
            continue
        problems += [f"block {i}, {p}" for p in _call_problems(block)]
    assert not problems, f"{rel}:\n  " + "\n  ".join(problems)
