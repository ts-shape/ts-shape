"""Every ``python`` code block on every docs page runs.

The blocks of one page run in order in one namespace, so a later block can
use names an earlier one defined. A block fails on any exception, and on any
ts-shape ``DeprecationWarning`` (the suite's ``filterwarnings`` turns those
into errors), and on any ``DataQualityWarning``: an example whose data the
library flags as inconsistent teaches the wrong thing. A page that shows such
a warning on purpose catches it with ``warnings.catch_warnings``. Printed
frames must not be truncated.

Blocks that need a network service or credentials are skipped by putting an
HTML comment ``<!-- no-run -->`` on the line before the fence; the same
comment as the first line of a page skips the whole page.
"""

from __future__ import annotations

import contextlib
import io
import re
import textwrap
import warnings
from pathlib import Path

import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
NO_RUN = "<!-- no-run -->"
_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```python[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.M | re.S
)
# pandas marks elided rows or columns with "..." and a "[n rows x m columns]"
# footer; a page should print the columns it talks about instead.
# pandas' defaults, as a reader's console has them (a local config must not
# hide a truncated frame).
_DISPLAY = ("display.max_columns", 0, "display.width", 80, "display.max_rows", 60)
_TRUNCATED = re.compile(r"(^|\s)\.\.\.(\s|$)|\[\d+ rows x \d+ columns\]", re.M)


def _blocks(text: str) -> list[tuple[int, str]]:
    """``(line, code)`` for each runnable python block of a page."""
    if text.lstrip().startswith(NO_RUN):
        return []
    found = []
    for match in _FENCE.finditer(text):
        before = text[: match.start()].rstrip().splitlines()
        if before and before[-1].strip() == NO_RUN:
            continue
        line = text.count("\n", 0, match.start()) + 1
        found.append((line, textwrap.dedent(match.group("body"))))
    return found


PAGES = sorted(p for p in DOCS.rglob("*.md") if _blocks(p.read_text(encoding="utf-8")))


def test_pages_are_discovered():
    assert len(PAGES) > 20


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(DOCS)) for p in PAGES])
def test_page_examples_run(page: Path, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # blocks that write files write them here
    rel = page.relative_to(ROOT)
    namespace: dict = {"__name__": "__docs__"}
    for line, code in _blocks(page.read_text(encoding="utf-8")):
        out = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(out),
                pd.option_context(*_DISPLAY),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("error", DataQualityWarning)
                exec(compile(code, f"{rel}:{line}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report which block failed
            pytest.fail(f"{rel}, block at line {line}: {type(exc).__name__}: {exc}")
        printed = out.getvalue()
        assert not _TRUNCATED.search(printed), (
            f"{rel}, block at line {line} prints a truncated frame; select the "
            f"columns it discusses:\n{printed[-1500:]}"
        )
