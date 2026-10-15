"""Every script in examples/ runs, without using deprecated names.

Each script must print something, and no frame it prints may be truncated
by pandas (``...`` for elided rows or columns): a demo shows the columns it
talks about.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

# pandas marks elided rows or columns with "..." and a "[n rows x m columns]"
# footer.
TRUNCATED = re.compile(r"(^|\s)\.\.\.(\s|$)|\[\d+ rows x \d+ columns\]", re.M)

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
    assert proc.stdout.strip(), f"{script.name} prints nothing"
    match = TRUNCATED.search(proc.stdout)
    assert match is None, (
        f"{script.name} prints a truncated frame; select its columns:\n"
        + proc.stdout[max(0, match.start() - 600) : match.end() + 200]
    )
