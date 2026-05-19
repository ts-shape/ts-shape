"""Generate the interactive architecture graph data for docs/guides/architecture.md.

Emits a single JSON file ``assets/architecture-graph.json`` into the
mkdocs build output. The JSON is consumed client-side by
``docs/assets/architecture-graph.js`` to render a Cytoscape.js graph.

Data source is the Python library itself:

* ``ts_shape.eventlog.taxonomy.REGISTRY`` — every detector method.
* ``src/ts_shape/`` directory walk — package / module structure.

Every node's click target is the matching auto-generated reference page
under ``docs/reference/ts_shape/...`` (produced by ``mkdocstrings``).
The graph never links into the hand-written prose pages under
``docs/guides/``, ``docs/modules/``, ``docs/pipelines/``, or
``docs/examples/`` — the map represents only the Python library.
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections import defaultdict
from pathlib import Path

import mkdocs_gen_files

from ts_shape.eventlog.taxonomy import REGISTRY


ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"


LAYERS = (
    "loader",
    "transform",
    "features",
    "context",
    "events",
    "eventlog",
    "utils",
)


# Layer-to-layer data flow is shown statically in the architecture page's
# Mermaid diagram. We deliberately do NOT emit these as Cytoscape edges:
# the layer nodes are compound parents (they contain packs, classes, and
# methods), and edges to/from compound parents in Cytoscape render at the
# bounding-box border, producing visual noise rather than information.


def _module_for_class(class_name: str) -> str | None:
    """Resolve ``ClassName`` to its dotted ``ts_shape.*.module`` path.

    Walks every ts_shape submodule under ``src/`` and returns the dotted
    name of the first module that defines a class with this name. Returns
    ``None`` when the class is a Lambda-rule detector (not present in the
    physical source tree).
    """
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(SRC).with_suffix("")
        dotted = ".".join(rel.parts)
        try:
            mod = importlib.import_module(dotted)
        except Exception:
            continue
        if getattr(mod, class_name, None) is not None:
            obj = mod.__dict__[class_name]
            if inspect.isclass(obj) and obj.__module__ == dotted:
                return dotted
    return None


def _reference_url(dotted_module: str, class_name: str, method: str | None) -> str:
    """Build the mkdocstrings reference URL for a class or method.

    The page is ``reference/<path>/<module>/`` (the trailing slash is the
    mkdocs Material default for ``use_directory_urls: true``). The anchor
    is ``ts_shape.<dotted>.<ClassName>[.<method>]``.
    """
    path = dotted_module.replace(".", "/")
    anchor = f"{dotted_module}.{class_name}"
    if method is not None:
        anchor = f"{anchor}.{method}"
    return f"../reference/{path}/#{anchor}"


def build_graph() -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []

    # ------------------------------------------------------------------
    # Layer nodes (the seven top-level packages)
    # ------------------------------------------------------------------
    for layer in LAYERS:
        nodes.append(
            {
                "data": {
                    "id": f"layer:{layer}",
                    "label": layer,
                    "type": "layer",
                }
            }
        )

    # Edges between layers intentionally omitted — see comment above the
    # LAYERS constant. The hierarchy expresses containment; data flow is
    # explained in the prose page.

    # ------------------------------------------------------------------
    # Pack nodes — sub-packages of each layer that physically exist
    # under src/ts_shape/<layer>/<pack>/
    # ------------------------------------------------------------------
    pack_dirs: dict[str, set[str]] = defaultdict(set)
    for layer in LAYERS:
        layer_dir = SRC / "ts_shape" / layer
        if not layer_dir.is_dir():
            continue
        for child in sorted(layer_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                pack_dirs[layer].add(child.name)

    for layer, packs in pack_dirs.items():
        for pack in sorted(packs):
            nodes.append(
                {
                    "data": {
                        "id": f"pack:{layer}.{pack}",
                        "label": pack,
                        "type": "pack",
                        "parent": f"layer:{layer}",
                    }
                }
            )

    # ------------------------------------------------------------------
    # Class and method nodes — driven by taxonomy.REGISTRY
    # ------------------------------------------------------------------
    classes_seen: dict[str, str] = {}  # ClassName -> dotted module
    methods_by_class: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for (class_name, method_name), rule in REGISTRY.items():
        if class_name not in classes_seen:
            dotted = _module_for_class(class_name)
            if dotted is None:
                # Lambda-rule detector or other dynamic registration —
                # represent it but without a click-through URL.
                classes_seen[class_name] = ""
            else:
                classes_seen[class_name] = dotted

        methods_by_class[class_name].append(
            (method_name, rule.shape, rule.pack)
        )

    for class_name, dotted in classes_seen.items():
        if not dotted:
            # Dynamic registration (lambda rules etc) — pin to events
            # layer with no pack parent, no URL.
            nodes.append(
                {
                    "data": {
                        "id": f"class:{class_name}",
                        "label": class_name,
                        "type": "class",
                        "parent": "layer:events",
                        "url": "",
                        "file": "",
                    }
                }
            )
            continue

        parts = dotted.split(".")  # e.g. ts_shape.events.quality.outlier_detection
        layer = parts[1]
        # parent: pack if one exists between layer and module, else layer
        if len(parts) >= 4 and (layer, parts[2]) in {
            (l, p) for l, ps in pack_dirs.items() for p in ps
        }:
            parent = f"pack:{layer}.{parts[2]}"
        else:
            parent = f"layer:{layer}"

        # File path relative to repo root, for hover tooltip
        rel_file = "/".join(parts) + ".py"
        rel_file = f"src/{rel_file}"

        nodes.append(
            {
                "data": {
                    "id": f"class:{class_name}",
                    "label": class_name,
                    "type": "class",
                    "parent": parent,
                    "url": _reference_url(dotted, class_name, None),
                    "file": rel_file,
                }
            }
        )

        for method_name, shape, pack in sorted(methods_by_class[class_name]):
            nodes.append(
                {
                    "data": {
                        "id": f"method:{class_name}.{method_name}",
                        "label": method_name,
                        "type": "method",
                        "parent": f"class:{class_name}",
                        "url": _reference_url(dotted, class_name, method_name),
                        "shape": shape,
                        "pack": pack,
                    }
                }
            )

    return {"nodes": nodes, "edges": edges}


def main() -> None:
    graph = build_graph()
    payload = json.dumps(graph, separators=(",", ":"))
    with mkdocs_gen_files.open("assets/architecture-graph.json", "w") as fd:
        fd.write(payload)


main()
