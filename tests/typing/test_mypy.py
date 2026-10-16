"""Type-check user code that imports ts-shape, as a strict user project would.

Guards two things a plain ``mypy src`` cannot see:

* names imported through the lazy ``from ts_shape import X`` (and from the
  lazy subpackages) are typed as the real class or function, not ``object``;
* deprecated classes, methods and attributes are reported by mypy's
  ``deprecated`` check, so users see the rename before it becomes an error.
"""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
USER_SCRIPT = HERE / "user_script.py"
DEPRECATED_USAGE = HERE / "deprecated_usage.py"


@pytest.fixture(scope="module")
def mypy_output(tmp_path_factory: pytest.TempPathFactory) -> str:
    api = pytest.importorskip("mypy.api")
    cache = tmp_path_factory.mktemp("mypy_cache")
    stdout, stderr, _status = api.run(
        [
            "--config-file",
            str(HERE / "mypy.ini"),
            "--cache-dir",
            str(cache),
            "--no-error-summary",
            "--show-error-codes",
            str(USER_SCRIPT),
            str(DEPRECATED_USAGE),
        ]
    )
    assert not stderr, stderr
    return stdout


def _errors_in(output: str, path: Path) -> list[str]:
    name = path.name
    return [
        line for line in output.splitlines() if Path(line.split(":")[0]).name == name
    ]


def test_user_script_passes_mypy_strict(mypy_output: str) -> None:
    errors = _errors_in(mypy_output, USER_SCRIPT)
    assert not errors, "\n".join(errors)


def test_deprecated_names_are_flagged(mypy_output: str) -> None:
    errors = _errors_in(mypy_output, DEPRECATED_USAGE)
    assert errors and all(line.endswith("[deprecated]") for line in errors), errors
    flagged = "\n".join(errors)
    assert re.search(r"class .*OEECalculator is deprecated", flagged)
    assert re.search(r"class .*ScrapTracking is deprecated", flagged)
    assert re.search(r"OEEEvents\.calculate_oee is deprecated", flagged)
    assert re.search(r"deprecated_attribute\.__get__ is deprecated", flagged)


def test_user_script_runs() -> None:
    runpy.run_path(str(USER_SCRIPT), run_name="__main__")
