"""The docs site's version selection (.github/scripts/docs_versions.py)."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "docs_versions.py"
_spec = importlib.util.spec_from_file_location("docs_versions", _PATH)
dv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dv)


def test_newest_patch_per_minor_line_oldest_line_first():
    tags = ["0.1.3", "0.0.90", "0.1.0", "0.0.91", "1.0.0", "0.1.10"]
    assert dv.select_versions(tags) == {
        "0.0": "0.0.91",
        "0.1": "0.1.10",
        "1.0": "1.0.0",
    }


def test_other_tags_are_ignored():
    assert dv.select_versions(["v0.1.1", "1.0.0rc1", "latest", "0.0.5"]) == {
        "0.0": "0.0.5"
    }


def test_tags_without_the_docs_site_are_skipped():
    # 0.0.90 predates the Docusaurus site, 0.0.91 has it; 0.1.0 lacks it, so
    # its line falls back to nothing.
    has_site = {"0.0.90": False, "0.0.91": True, "0.1.0": False}.get
    assert dv.select_versions(["0.0.90", "0.0.91", "0.1.0"], has_site) == {
        "0.0": "0.0.91"
    }


def test_newest_tag_without_site_falls_back_to_an_older_patch():
    has_site = {"0.0.92": False, "0.0.91": True}.get
    assert dv.select_versions(["0.0.91", "0.0.92"], has_site) == {"0.0": "0.0.91"}


def test_no_tags():
    assert dv.select_versions([]) == {}
