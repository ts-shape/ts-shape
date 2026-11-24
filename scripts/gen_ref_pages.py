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
    # Like mkdocstrings: undocumented members are left out.
    "show_if_no_docstring": False,
    # Summary tables for classes (on module pages) and methods (on classes).
    "summary": {"classes": True, "functions": True},
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
SECTION_RE = re.compile(r"^\*\*([A-Z][A-Za-z ]*):\*\*$")
CODE_RE = re.compile(r"<code>(.*?)</code>")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")


def kind_of(obj: Object) -> str:
    """Badge text for an object: class, method, function, property, ..."""
    labels = set(getattr(obj, "labels", ()) or ())
    if obj.is_class:
        return "class"
    if obj.is_function:
        for label in ("property", "classmethod", "staticmethod"):
            if label in labels:
                return label
        return (
            "method" if obj.parent is not None and obj.parent.is_class else "function"
        )
    if "property" in labels:
        return "property"
    return "attribute"


def meta_line(obj: Object | None) -> str:
    """Kind badge and a link to the object's source lines."""
    if obj is None:
        return ""
    kind = kind_of(obj)
    parts = [f'<span class="api-kind api-kind--{kind}">{kind}</span>']
    if obj.lineno and obj.filepath and not isinstance(obj.filepath, list):
        rel = Path(obj.filepath).resolve().relative_to(ROOT).as_posix()
        lines = f"L{obj.lineno}"
        if obj.endlineno and obj.endlineno != obj.lineno:
            lines += f"-L{obj.endlineno}"
        parts.append(
            f'<a class="api-source" href="{REPO_BLOB}/{rel}#{lines}">source</a>'
        )
    return f'<div class="api-meta">{"".join(parts)}</div>'


def decorate(body: str, module: Object, anchors: set[str]) -> str:
    """Post-process griffe2md output for the docs site.

    - Each object heading gets an explicit ``{#full.path}`` anchor (griffe2md
      nests members one heading level per object level, starting at
      ``CONFIG["heading_level"]``) and a kind badge with a source link.
    - ``**Parameters:**`` style labels become ``api-section`` labels.
    - Doctest examples (``>>>``) are fenced as Python code; unfenced they
      would render as nested blockquotes.
    """
    stack: list[str] = []
    out: list[str] = []
    in_fence = False
    in_doctest = False
    for line in body.splitlines():
        if in_doctest:
            if line.strip():
                out.append(line)
                continue
            out.append("```")
            in_doctest = False
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if line.startswith(">>> "):
            out.extend(["```python", line])
            in_doctest = True
            continue
        m = HEADING_RE.match(line)
        if m:
            depth = len(m.group(1)) - CONFIG["heading_level"]
            stack = stack[:depth] + [m.group(2)]
            path = ".".join([module.path, *stack])
            anchors.add(path)
            try:
                obj = module[".".join(stack)]
            except (KeyError, ValueError):
                obj = None
            out.append(f"{line.rstrip()} {{#{path}}}")
            meta = meta_line(obj)
            if meta:
                out.extend(["", meta])
            continue
        m = SECTION_RE.match(line)
        if m:
            out.append(f'<div class="api-section">{m.group(1)}</div>')
            continue
        out.append(line)
    if in_doctest:
        out.append("```")
    return "\n".join(out)


def plain_code(body: str) -> str:
    r"""Turn griffe2md's ``<code>`` spans into Markdown code spans and links.

    Docusaurus renders a raw ``<code>`` element as a code block, so
    ``<code>[Base](url)</code>`` becomes ``[`Base`](url)`` and
    ``<code>str \| None</code>`` becomes ```str \| None``` (the escaped pipe
    keeps table cells intact).
    """

    def convert(m: re.Match) -> str:
        inner, pieces, pos = m.group(1), [], 0
        for link in MD_LINK_RE.finditer(inner):
            if inner[pos : link.start()].strip():
                pieces.append(f"`{inner[pos:link.start()].strip()}`")
            pieces.append(f"[`{link.group(1)}`]({link.group(2)})")
            pos = link.end()
        if inner[pos:].strip():
            pieces.append(f"`{inner[pos:].strip()}`")
        return " ".join(pieces)

    out, in_fence = [], False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        out.append(line if in_fence else CODE_RE.sub(convert, line))
    return "\n".join(out)


def render(module: Object, anchors: set[str]) -> str:
    members = [
        name
        for name, member in module.members.items()
        if not name.startswith("_") and not member.is_alias and not member.is_module
        # Like mkdocstrings: undocumented members are left out.
        and member.has_docstrings
    ]
    context = prepare_context(module, {**CONFIG, "members": members})
    body = prepare_env().get_template("module.md.jinja").render(**context)
    return decorate(body, module, anchors)


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

        body = plain_code(LINK_RE.sub(relink, body))
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
