"""Docs must only show ts_shape names that exist.

Scans the Python code blocks of every Markdown page (docs/ and README.md)
for ``from ts_shape... import Name`` and checks that the import resolves and
that every ``Name.attribute`` used in the same block exists.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

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
