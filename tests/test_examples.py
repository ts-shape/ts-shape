"""Every script in examples/ runs, without using deprecated names."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_runs(script):
    proc = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=script.parent,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
