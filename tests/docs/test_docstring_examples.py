"""Every ``>>>`` example in a ts_shape docstring runs and prints what it shows.

Modules are discovered by walking the package; any module whose docstrings
(module, class, function or method) contain a ``>>>`` prompt is collected and
run through :mod:`doctest` with ``ELLIPSIS | NORMALIZE_WHITESPACE``. Examples
must use current API names, so ``DeprecationWarning`` is raised as an error.
"""

from __future__ import annotations

import doctest
import importlib
import io
import pkgutil
import warnings

import pytest

import ts_shape

OPTIONFLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


def _modules_with_examples() -> list[str]:
    finder = doctest.DocTestFinder()
    names = []
    for info in pkgutil.walk_packages(ts_shape.__path__, "ts_shape."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # optional extras not installed
            continue
        if any(test.examples for test in finder.find(module)):
            names.append(info.name)
    return sorted(names)


MODULES = _modules_with_examples()


def test_examples_are_discovered():
    assert MODULES, "no docstring examples found under ts_shape"


@pytest.mark.parametrize("module_name", MODULES)
def test_docstring_examples(module_name: str) -> None:
    module = importlib.import_module(module_name)
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=OPTIONFLAGS)
    report = io.StringIO()
    failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for test in finder.find(module):
            if test.examples:
                failed += runner.run(test, out=report.write).failed
    assert failed == 0, report.getvalue()
