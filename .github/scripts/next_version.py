"""Decide the next release version for the release workflow.

The version is the latest ``X.Y.Z`` tag plus a bump. Commits since that tag
choose the bump with git trailers (the most recent trailer wins)::

    Release-Bump: patch | minor | major | none
    Release-As: 1.0.0

Manual runs pass ``INPUT_BUMP`` (auto, patch, minor, major) and
``INPUT_VERSION`` instead; they take precedence over trailers. Prints
``version=...`` and ``release=true|false`` lines for ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
TRAILER_RE = re.compile(
    r"^(Release-Bump|Release-As):[ \t]*(\S+)[ \t]*$", re.IGNORECASE | re.MULTILINE
)
BUMPS = ("patch", "minor", "major", "none")


def parse(version: str) -> tuple[int, int, int]:
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"not an X.Y.Z version: {version!r}")
    major, minor, patch = (int(g) for g in match.groups())
    return major, minor, patch


def latest_tag(tags: list[str]) -> str:
    """Highest ``X.Y.Z`` tag; other tags (``v0.1.1``, ``1.0.0rc1``) are ignored."""
    versions = [t for t in tags if VERSION_RE.match(t)]
    return max(versions, key=parse) if versions else "0.0.0"


def bump(version: str, kind: str) -> str:
    major, minor, patch = parse(version)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump {kind!r}; expected one of {BUMPS}")


def choice_from_messages(messages: list[str]) -> tuple[str, str | None]:
    """``(bump, exact_version)`` from commit messages, newest first."""
    for message in messages:
        found = TRAILER_RE.findall(message)
        if not found:
            continue
        key, value = found[-1]
        if key.lower() == "release-as":
            parse(value)
            return "exact", value
        value = value.lower()
        if value not in BUMPS:
            raise ValueError(f"Release-Bump must be one of {BUMPS}, got {value!r}")
        return value, None
    return "patch", None


def next_version(
    tags: list[str],
    messages: list[str],
    input_bump: str = "auto",
    input_version: str = "",
) -> str | None:
    """The version to release, or ``None`` when the release is skipped."""
    current = latest_tag(tags)
    if input_version:
        kind, exact = "exact", input_version.strip()
    elif input_bump and input_bump != "auto":
        kind, exact = input_bump, None
    else:
        kind, exact = choice_from_messages(messages)
    if kind == "none":
        return None
    new = exact if exact is not None else bump(current, kind)
    if parse(new) <= parse(current):
        raise ValueError(f"version {new} is not higher than the last release {current}")
    if new in tags:
        raise ValueError(f"tag {new} already exists")
    return new


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def main() -> int:
    tags = _git("tag", "--list").split()
    current = latest_tag(tags)
    span = "HEAD" if current == "0.0.0" else f"{current}..HEAD"
    messages = [m for m in _git("log", "--format=%B%x00", span).split("\x00") if m]
    new = next_version(
        tags,
        messages,
        os.environ.get("INPUT_BUMP", "auto"),
        os.environ.get("INPUT_VERSION", ""),
    )
    print(f"Last release {current}; next {new or '(skipped)'}", file=sys.stderr)
    print(f"version={new or ''}")
    print(f"release={'true' if new else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
