"""Bundle the architecture sunburst into a single standalone HTML file.

Reads:
- ``docs/assets/architecture-sunburst.css``   (inlined into <style>)
- ``docs/assets/architecture-sunburst.js``    (inlined into <script>)
- ``site/assets/architecture-graph.json``     (inlined into a JSON script tag;
                                               run ``mkdocs build`` first to
                                               refresh this file)

Writes:
- ``site/assets/architecture-map.html`` — opens in any browser via file://
  or a static host. No mkdocs / Material theme dependency. D3.js v7 is the
  only external load (pinned CDN URL).

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

CSS = (DOCS_ASSETS / "architecture-sunburst.css").read_text(encoding="utf-8")
JS = (DOCS_ASSETS / "architecture-sunburst.js").read_text(encoding="utf-8")
GRAPH = json.loads((SITE_ASSETS / "architecture-graph.json").read_text(encoding="utf-8"))


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ts-shape — Architecture Map (standalone)</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
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
  max-width: 1100px;
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
.ts-map-hint {{
  margin: 0 0 16px;
  padding: 12px 16px;
  background: var(--md-default-fg-color--lightest, #f1f5f9);
  border-left: 3px solid var(--md-accent-fg-color, #f59e0b);
  border-radius: 4px;
  font-size: 0.85rem;
  line-height: 1.5;
}}
.ts-map-hint a {{
  color: var(--md-accent-fg-color);
  font-weight: 600;
}}
{embedded_css}
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
    Click any ring to zoom into that branch and load its details below.
    Click the centre to zoom out. The details panel under the chart shows
    the description, source file, child list, and a link to the live
    reference docs
    (<a href="https://ts-shape.github.io/ts-shape/" target="_blank"
       rel="noopener">ts-shape.github.io/ts-shape</a>).
    Type in the search box to fade non-matching arcs.
  </p>

  <div class="ts-sunburst-card">
    <input id="ts-search" type="text" placeholder="Search class or method..." aria-label="Search">
    <div id="ts-sunburst" role="img" aria-label="ts-shape architecture sunburst"></div>
  </div>

  <div id="ts-details" aria-live="polite"></div>
</div>

<script>
// Synchronous bootstrap: read the embedded JSON, rewrite the relative
// reference URLs into absolute docs.github.io URLs (so clicking a node
// in a downloaded file still lands on a real page), and install a fetch
// shim so the page-level JS reads this data instead of trying to fetch
// architecture-graph.json from disk. All sync so there's no race with
// the page-level JS's DOMContentLoaded handler.
(function () {{
  var DOCS_BASE = "https://ts-shape.github.io/ts-shape/";
  var raw = document.getElementById("ts-graph-data").textContent;
  var data = JSON.parse(raw);
  data.nodes.forEach(function (n) {{
    var u = n.data && n.data.url;
    if (u && u.indexOf("../") === 0) {{
      // Strip ALL leading `../` segments. The gen script emits two
      // (`../../reference/...`) — anything we leave behind here would
      // confuse the browser's URL resolver and produce a 404.
      n.data.url = DOCS_BASE + u.replace(/^(?:\\.\\.\\/)+/, "");
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
