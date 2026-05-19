"""Bundle the architecture graph into a single standalone HTML file.

Reads:
- ``docs/assets/architecture-graph.css``      (inlined into <style>)
- ``docs/assets/architecture-graph.js``       (inlined into <script>)
- ``site/assets/architecture-graph.json``     (inlined into a JSON script tag;
                                               run ``mkdocs build`` first to
                                               refresh this file)

Writes:
- ``site/assets/architecture-map.html`` — opens in any browser via file://
  or a static host. No mkdocs / Material theme dependency. Cytoscape.js is
  the only external load (pinned CDN URL).

The standalone page intercepts the page-level JS's fetch() call so it
reads the inlined JSON rather than hitting the network for a missing
``architecture-graph.json`` next to the HTML file.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS_ASSETS = ROOT / "docs" / "assets"
SITE_ASSETS = ROOT / "site" / "assets"

CSS = (DOCS_ASSETS / "architecture-graph.css").read_text(encoding="utf-8")
JS = (DOCS_ASSETS / "architecture-graph.js").read_text(encoding="utf-8")
GRAPH = json.loads((SITE_ASSETS / "architecture-graph.json").read_text(encoding="utf-8"))


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ts-shape — Architecture Map (standalone)</title>
<script src="https://unpkg.com/cytoscape@3.30.1/dist/cytoscape.min.js"></script>
<style>
:root {{
  --md-default-bg-color: #ffffff;
  --md-default-fg-color: #0f172a;
  --md-default-fg-color--light: #475569;
  --md-default-fg-color--lighter: #cbd5e1;
  --md-default-fg-color--lightest: #f1f5f9;
  --md-accent-fg-color: #f59e0b;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --md-default-bg-color: #0f172a;
    --md-default-fg-color: #e2e8f0;
    --md-default-fg-color--light: #94a3b8;
    --md-default-fg-color--lighter: #475569;
    --md-default-fg-color--lightest: #1e293b;
  }}
}}
html, body {{
  margin: 0;
  padding: 0;
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}}
.wrap {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 1.75rem;
}}
.subtitle {{
  margin: 0 0 16px;
  color: var(--md-default-fg-color--light);
  font-size: 0.9rem;
}}
.ts-map-hint a {{
  color: var(--md-accent-fg-color);
}}
{embedded_css}
.ts-graph-fullscreen #ts-shape-graph {{
  height: calc(100vh - 240px);
}}
/* Standalone-only: keep the graph visible at all viewport widths.
 * The docs page uses a mobile fallback because of the Material chrome,
 * but the standalone HTML has the full viewport to itself. */
@media (max-width: 800px) {{
  #ts-shape-graph,
  #ts-graph-controls {{
    display: block;
  }}
  .ts-graph-mobile-notice {{
    display: none;
  }}
  .ts-graph-fullscreen #ts-shape-graph {{
    height: calc(100vh - 320px);
  }}
}}
</style>
<script id="ts-graph-data" type="application/json">{embedded_json}</script>
</head>
<body>
<div class="wrap">
  <h1>ts-shape — Architecture Map</h1>
  <p class="subtitle">
    {n_layers} layers · {n_packs} packs · {n_classes} classes · {n_methods}
    detector methods. Generated from
    <code>ts_shape.eventlog.taxonomy.REGISTRY</code> at docs build time.
  </p>
  <p class="ts-map-hint">
    Pan with drag, zoom with scroll. <b>Search</b> finds class and method
    names. <b>Depth</b> controls how deep the hierarchy expands. Click any
    class or method to open its reference page on the docs site
    (<a href="https://jakobgabriel.github.io/ts-shape/" target="_blank"
       rel="noopener">jakobgabriel.github.io/ts-shape</a>).
  </p>

  <div class="ts-graph-card ts-graph-fullscreen">
    <div id="ts-graph-controls">
      <input id="ts-search" type="text" placeholder="Search class or method..." aria-label="Search">

      <fieldset>
        <legend>Depth</legend>
        <label><input type="radio" name="ts-depth" value="layers"> Layers</label>
        <label><input type="radio" name="ts-depth" value="packs" checked> + Packs</label>
        <label><input type="radio" name="ts-depth" value="classes"> + Classes</label>
        <label><input type="radio" name="ts-depth" value="everything"> Everything</label>
      </fieldset>

      <fieldset>
        <legend>Layers</legend>
        <label><input type="checkbox" name="ts-layer" value="loader" checked> loader</label>
        <label><input type="checkbox" name="ts-layer" value="transform" checked> transform</label>
        <label><input type="checkbox" name="ts-layer" value="features" checked> features</label>
        <label><input type="checkbox" name="ts-layer" value="context" checked> context</label>
        <label><input type="checkbox" name="ts-layer" value="events" checked> events</label>
        <label><input type="checkbox" name="ts-layer" value="eventlog" checked> eventlog</label>
        <label><input type="checkbox" name="ts-layer" value="utils" checked> utils</label>
      </fieldset>

      <button id="ts-reset" type="button">Reset</button>
    </div>

    <div id="ts-shape-graph" role="img" aria-label="ts-shape architecture graph"></div>
  </div>
</div>

<script>
// Synchronous bootstrap: read the embedded JSON, rewrite the relative
// reference URLs into absolute docs.github.io URLs (so clicking a node
// in a downloaded file still lands on a real page), and install a fetch
// shim so the page-level JS reads this data instead of trying to fetch
// architecture-graph.json from disk. All sync so there's no race with
// the page-level JS's DOMContentLoaded handler.
(function () {{
  var DOCS_BASE = "https://jakobgabriel.github.io/ts-shape/";
  var raw = document.getElementById("ts-graph-data").textContent;
  var data = JSON.parse(raw);
  data.nodes.forEach(function (n) {{
    var u = n.data && n.data.url;
    if (u && u.indexOf("../") === 0) {{
      n.data.url = DOCS_BASE + u.replace(/^\\.\\.\\//, "");
    }}
  }});
  var origFetch = typeof window.fetch === "function" ? window.fetch.bind(window) : null;
  window.fetch = function (input) {{
    var url = typeof input === "string" ? input : (input && input.url) || "";
    if (url.indexOf("architecture-graph.json") >= 0) {{
      return Promise.resolve({{
        ok: true,
        status: 200,
        json: function () {{ return Promise.resolve(data); }},
      }});
    }}
    if (origFetch) return origFetch.apply(null, arguments);
    return Promise.reject(new Error("fetch unavailable"));
  }};
}})();
</script>
<script>
{embedded_js}
</script>
</body>
</html>
"""


def main() -> None:
    n_layers = sum(1 for n in GRAPH["nodes"] if n["data"]["type"] == "layer")
    n_packs = sum(1 for n in GRAPH["nodes"] if n["data"]["type"] == "pack")
    n_classes = sum(1 for n in GRAPH["nodes"] if n["data"]["type"] == "class")
    n_methods = sum(1 for n in GRAPH["nodes"] if n["data"]["type"] == "method")

    html = HTML_TEMPLATE.format(
        embedded_css=CSS,
        embedded_js=JS,
        # JSON safely embedded: only </script> needs escaping. The graph
        # never contains that string, but we encode defensively.
        embedded_json=json.dumps(GRAPH).replace("</", "<\\/"),
        n_layers=n_layers,
        n_packs=n_packs,
        n_classes=n_classes,
        n_methods=n_methods,
    )

    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    out = SITE_ASSETS / "architecture-map.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
