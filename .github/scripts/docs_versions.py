"""Snapshot one docs version per release line for the docs site.

Run from the repository root before ``npm run build`` (the release workflow's
docs job does)::

    python .github/scripts/docs_versions.py

For every minor line ``X.Y`` the newest ``X.Y.Z`` release tag that contains
this script becomes a docs version named ``X.Y`` and labelled
``X.Y.Z``. For each one the tag is checked out in a temporary worktree, its
own ``scripts/gen_ref_pages.py`` renders its API reference, and
``docusaurus docs:version`` copies its docs and sidebars into
``website/versioned_docs`` / ``website/versioned_sidebars``. The labels go to
``website/versions-meta.json``, which ``docusaurus.config.ts`` reads. All of
these files are build products (git-ignored); without them the site builds
unversioned, as in pull-request CI.

Tags are read from the local repository, so the checkout needs them
(``fetch-depth: 0``). Requires the docs Python extra and ``npm ci`` in
``website/``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEBSITE = ROOT / "website"
# The same rule as next_version.py: only plain X.Y.Z tags are releases.
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
# Versions start with the first release that ships this script: older tags
# predate the landing page (their index.md claims "/") and docs/tags.yml.
SITE_MARKER = ".github/scripts/docs_versions.py"
GENERATED = (
    WEBSITE / "versioned_docs",
    WEBSITE / "versioned_sidebars",
    WEBSITE / "versions.json",
    WEBSITE / "versions-meta.json",
)


def select_versions(
    tags: Iterable[str], has_site: Callable[[str], bool] = lambda tag: True
) -> dict[str, str]:
    """Newest release tag with the docs site per minor line, oldest line first.

    >>> select_versions(["0.0.90", "0.0.91", "0.1.0", "0.1.3", "v0.1.1"])
    {'0.0': '0.0.91', '0.1': '0.1.3'}
    """
    best: dict[tuple[int, int], tuple[int, str]] = {}
    for tag in tags:
        match = VERSION_RE.match(tag)
        if match is None:
            continue
        major, minor, patch = (int(g) for g in match.groups())
        if (major, minor) in best and best[(major, minor)][0] >= patch:
            continue
        if not has_site(tag):
            continue
        best[(major, minor)] = (patch, tag)
    return {
        f"{major}.{minor}": best[(major, minor)][1] for major, minor in sorted(best)
    }


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def tag_has_site(tag: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{tag}:{SITE_MARKER}"],
            check=False,
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def snapshot(name: str, tag: str) -> None:
    """Add release ``tag`` as docs version ``name`` (``docusaurus docs:version``)."""
    with tempfile.TemporaryDirectory(prefix=f"docs-{tag}-") as tmp:
        tree = Path(tmp) / "tree"
        git("worktree", "add", "--detach", str(tree), tag)
        try:
            # The tag's own generator renders the tag's own sources.
            subprocess.run(
                [sys.executable, str(tree / "scripts" / "gen_ref_pages.py")],
                check=True,
            )
            env = {
                **os.environ,
                "DOCS_PATH": str(tree / "docs"),
                "SIDEBARS_PATH": str(tree / "website" / "sidebars.ts"),
            }
            subprocess.run(
                ["npx", "docusaurus", "docs:version", name],
                cwd=WEBSITE,
                env=env,
                check=True,
            )
        finally:
            git("worktree", "remove", "--force", str(tree))


def main() -> None:
    for path in GENERATED:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    versions = select_versions(git("tag", "--list").split(), tag_has_site)
    if not versions:
        print("No release tag has versioned docs yet; building unversioned.")
        return

    # docs:version prepends to versions.json, so the newest line ends up first.
    for name, tag in versions.items():
        print(f"Docs version {name} from tag {tag}")
        snapshot(name, tag)

    meta = {name: {"label": tag, "tag": tag} for name, tag in versions.items()}
    (WEBSITE / "versions-meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Docs versions: {', '.join(f'{n} ({t})' for n, t in versions.items())}")


if __name__ == "__main__":
    main()
