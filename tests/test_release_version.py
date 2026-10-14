"""The release workflow's version decision (.github/scripts/next_version.py)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "next_version.py"
_spec = importlib.util.spec_from_file_location("next_version", _PATH)
nv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nv)

TAGS = ["0.0.9", "0.0.62", "0.0.61", "v0.1.1", "1.0.0rc1"]


def test_latest_tag_is_numeric_and_ignores_other_tags():
    assert nv.latest_tag(TAGS) == "0.0.62"
    assert nv.latest_tag([]) == "0.0.0"


def test_default_is_a_patch_release():
    assert nv.next_version(TAGS, ["Fix a bug"]) == "0.0.63"


@pytest.mark.parametrize(
    "trailer,expected",
    [
        ("Release-Bump: minor", "0.1.0"),
        ("Release-Bump: major", "1.0.0"),
        ("release-bump: Patch", "0.0.63"),
        ("Release-As: 1.0.0", "1.0.0"),
        ("Release-Bump: none", None),
    ],
)
def test_trailers_choose_the_bump(trailer, expected):
    msg = f"Title\n\nBody text\n\n{trailer}\nCo-Authored-By: someone"
    assert nv.next_version(TAGS, [msg, "older commit"]) == expected


def test_newest_trailer_wins():
    msgs = ["new\n\nRelease-Bump: minor", "old\n\nRelease-As: 1.0.0"]
    assert nv.next_version(TAGS, msgs) == "0.1.0"


def test_manual_inputs_override_trailers():
    msgs = ["x\n\nRelease-Bump: major"]
    assert nv.next_version(TAGS, msgs, input_bump="minor") == "0.1.0"
    assert nv.next_version(TAGS, msgs, input_version="2.0.0") == "2.0.0"


def test_rejects_versions_that_do_not_move_forward_or_exist():
    with pytest.raises(ValueError, match="not higher"):
        nv.next_version(TAGS, ["x\n\nRelease-As: 0.0.62"])
    with pytest.raises(ValueError, match="X.Y.Z"):
        nv.next_version(TAGS, ["x\n\nRelease-As: 1.0"])
    with pytest.raises(ValueError, match="Release-Bump"):
        nv.next_version(TAGS, ["x\n\nRelease-Bump: huge"])


def test_trailer_must_start_a_line():
    assert nv.next_version(TAGS, ["mentions Release-Bump: major inline"]) == "0.0.63"
