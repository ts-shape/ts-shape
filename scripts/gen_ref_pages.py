"""Generate the API reference pages (and the changelog/license pages) for the docs site.

Run before building the Docusaurus site::

    python scripts/gen_ref_pages.py

One Markdown page per public module is written to ``docs/reference/``, with
package ``__init__`` modules rendered as the package's ``index.md``. Private
modules (any path part starting with ``_`` other than ``__init__``) and
``__main__`` are skipped. Docstrings are parsed with Griffe (Google style) and
rendered with griffe2md. Every object heading gets an explicit
``{#ts_shape.path.to.object}`` anchor, and cross-references are rewritten to
point at the page that documents the target; references to objects outside
ts-shape (pandas, builtins) become plain text.

``docs/changelog.md`` and ``docs/license.md`` are written from ``CHANGES.txt``
and ``LICENSE.txt``. All generated files are git-ignored.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from griffe import GriffeLoader, Object, Parser
from griffe2md import default_config, prepare_context, prepare_env

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DOCS = ROOT / "docs"
REF = DOCS / "reference"
PACKAGE = "ts_shape"
REPO_BLOB = "https://github.com/ts-shape/ts-shape/blob/main"

# Mirrors the former mkdocstrings options.
CONFIG = {
    **default_config,
    "docstring_section_style": "table",
    "members_order": "source",
    "filters": ["!^_"],
    "merge_init_into_class": True,
    "separate_signature": True,
    "show_signature_annotations": True,
    "heading_level": 2,
    "show_root_heading": False,
    "show_root_full_path": False,
    "show_root_members_full_path": False,
    "show_object_full_path": False,
    "show_submodules": False,
    "inherited_members": False,
}

LINK_RE = re.compile(r"\[([^\]]*)\]\(#([A-Za-z_][\w.]*)\)")


def public_modules() -> list[tuple[tuple[str, ...], Path, bool]]:
    """(dotted parts, source path, is_package) for every public module."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        is_pkg = parts[-1] == "__init__"
        if is_pkg:
            parts = parts[:-1]
        elif parts[-1] == "__main__":
            continue
        if any(p.startswith("_") for p in parts):
            continue
        out.append((parts, path, is_pkg))
    return out


def page_path(parts: tuple[str, ...], is_pkg: bool) -> Path:
    """Doc file for a module, relative to ``docs/reference``."""
    rel = Path(*parts)
    return rel / "index.md" if is_pkg else rel.with_suffix(".md")


HEADING_RE = re.compile(r"^(#{2,6}) `([A-Za-z_]\w*)`\s*$")


def add_anchors(body: str, module_path: str, anchors: set[str]) -> str:
    """Give each object heading an explicit ``{#full.path}`` anchor.

    griffe2md nests members one heading level per object level, starting at
    ``CONFIG["heading_level"]`` for the module's direct members.
    """
    stack: list[str] = []
    lines = []
    for line in body.splitlines():
        m = HEADING_RE.match(line)
        if m:
            depth = len(m.group(1)) - CONFIG["heading_level"]
            stack = stack[:depth] + [m.group(2)]
            path = ".".join([module_path, *stack])
            anchors.add(path)
            line = f"{line.rstrip()} {{#{path}}}"
        lines.append(line)
    return "\n".join(lines)


def render(module: Object, anchors: set[str]) -> str:
    members = [
        name
        for name, member in module.members.items()
        if not name.startswith("_") and not member.is_alias and not member.is_module
    ]
    context = prepare_context(module, {**CONFIG, "members": members})
    body = prepare_env().get_template("module.md.jinja").render(**context)
    return add_anchors(body, module.path, anchors)


def resolve(target: str, page: Path, owner: dict[str, Path]) -> str | None:
    """Link for an object path, relative to ``page``; None when not documented."""
    if target in owner:
        dest = owner[target]
        if dest == page:
            return f"#{target}"
        rel = Path(*([".."] * (len(page.parts) - 1))) / dest
        return f"{rel.as_posix()}#{target}"
    return None


def main() -> None:
    loader = GriffeLoader(
        search_paths=[str(SRC)],
        docstring_parser=Parser("google"),
        docstring_options={"ignore_init_summary": True},
    )
    package = loader.load(PACKAGE)
    loader.resolve_aliases(external=False, implicit=False)

    # Fresh output, keeping the hand-written overview page.
    for child in REF.iterdir() if REF.exists() else []:
        if child.name == "index.md":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    rendered: dict[Path, tuple[str, tuple[str, ...], Path]] = {}
    owner: dict[str, Path] = {}
    for parts, src_path, is_pkg in public_modules():
        module = package if len(parts) == 1 else package[".".join(parts[1:])]
        anchors: set[str] = {module.path}
        body = render(module, anchors)
        page = page_path(parts, is_pkg)
        rendered[page] = (body, parts, src_path)
        for anchor in anchors:
            owner.setdefault(anchor, page)

    for page, (body, parts, src_path) in rendered.items():

        def relink(m: re.Match, page: Path = page) -> str:
            href = resolve(m.group(2), page, owner)
            return f"[{m.group(1)}]({href})" if href else m.group(1)

        body = LINK_RE.sub(relink, body)
        ident = ".".join(parts)
        front = {
            "title": ident,
            "sidebar_label": parts[-1],
            "custom_edit_url": f"{REPO_BLOB}/{src_path.relative_to(ROOT).as_posix()}",
        }
        header = (
            "---\n"
            + "".join(f"{k}: {json.dumps(v)}\n" for k, v in front.items())
            + "---\n\n"
        )
        out = REF / page
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"{header}# `{ident}` {{#{ident}}}\n\n{body.strip()}\n", encoding="utf-8"
        )

    # Category metadata so the autogenerated sidebar keeps package names.
    for directory in sorted(p for p in REF.rglob("*") if p.is_dir()):
        (directory / "_category_.json").write_text(
            json.dumps({"label": directory.name, "collapsed": True}) + "\n",
            encoding="utf-8",
        )

    for name, source, title in (
        ("changelog.md", "CHANGES.txt", "Changelog"),
        ("license.md", "LICENSE.txt", "License"),
    ):
        text = (ROOT / source).read_text(encoding="utf-8")
        (DOCS / name).write_text(
            f"---\ntitle: {title}\ncustom_edit_url: {REPO_BLOB}/{source}\n---\n\n{text}",
            encoding="utf-8",
        )

    print(f"Wrote {len(rendered)} API reference pages to {REF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
