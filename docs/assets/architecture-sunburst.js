/* ts-shape architecture map — zoomable D3 v7 sunburst.
 *
 * Mounts on #ts-sunburst in docs/guides/architecture-map.md and consumes
 * assets/architecture-graph.json (generated at docs build time by
 * scripts/gen_architecture_graph.py from ts_shape.eventlog.taxonomy.REGISTRY
 * and a walk of src/ts_shape/). Same data source as the previous Cytoscape
 * renderer; click-throughs target the auto-generated mkdocstrings reference
 * pages only.
 *
 * Interaction model (Bostock's zoomable-sunburst pattern):
 *   - click an arc        → zoom into that subtree
 *   - click the centre    → zoom out one level
 *   - hover an arc        → breadcrumb updates, tooltip shows details
 *   - click a class/method arc → open its docs reference page
 *   - search input        → fade non-matching arcs
 */
(function () {
  "use strict";

  // Layer colour palette — matches the Mermaid in docs/concept.md so the
  // sunburst and the static diagram speak the same visual language.
  var LAYER_HSL = {
    loader: { h: 200, s: 80 },     // blue
    transform: { h: 170, s: 60 },  // teal
    features: { h: 170, s: 60 },   // teal
    context: { h: 170, s: 60 },    // teal
    events: { h: 32, s: 92 },      // orange
    eventlog: { h: 45, s: 90 },    // amber
    utils: { h: 220, s: 8 },       // grey
  };

  function init() {
    var container = document.getElementById("ts-sunburst");
    if (!container || typeof window.d3 === "undefined") {
      if (container) {
        container.innerHTML =
          '<p class="ts-sunburst-fallback">' +
          "Could not load the sunburst visualisation (D3.js missing). " +
          "See the static Mermaid on the prose Architecture page.</p>";
      }
      return;
    }

    var jsonUrl = new URL(
      "../../assets/architecture-graph.json",
      document.baseURI
    ).toString();

    fetch(jsonUrl)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        render(container, data);
      })
      .catch(function (err) {
        container.innerHTML =
          '<p class="ts-sunburst-fallback">Could not load graph data (' +
          err.message +
          ").</p>";
      });
  }

  // ----------------------------------------------------------------
  // Convert the flat {nodes:[…]} list into a d3.hierarchy tree.
  // Adds a synthetic root "ts_shape" parent so the seven layers share
  // a single ring boundary.
  // ----------------------------------------------------------------
  function buildHierarchy(data) {
    var d3 = window.d3;
    var nodes = [{ id: "root", parent: null, label: "ts_shape", type: "root" }];
    data.nodes.forEach(function (n) {
      nodes.push({
        id: n.data.id,
        parent: n.data.parent || "root",
        label: n.data.label,
        type: n.data.type,
        url: n.data.url || null,
        shape: n.data.shape || null,
        pack: n.data.pack || null,
        file: n.data.file || null,
      });
    });
    var root = d3
      .stratify()
      .id(function (d) { return d.id; })
      .parentId(function (d) { return d.parent; })(nodes);
    // Leaf value = 1; partition() sums these for arc sizing so layers
    // scale with their detector-method count.
    root.sum(function (d) { return d.type === "method" ? 1 : 0; });
    root.sort(function (a, b) { return b.value - a.value; });
    return root;
  }

  // ----------------------------------------------------------------
  // Colour: pick a hue from the top-level layer ancestor, then walk
  // outward making each ring slightly lighter.
  // ----------------------------------------------------------------
  function colourFor(node) {
    var d3 = window.d3;
    if (node.depth === 0) return "transparent";
    var layer = node;
    while (layer.depth > 1) layer = layer.parent;
    var layerName = layer.data.label;
    var base = LAYER_HSL[layerName] || { h: 220, s: 30 };
    // Lightness increases with depth: layer 50%, pack 60%, class 70%, method 80%.
    var l = 38 + (node.depth - 1) * 12;
    return d3.hsl(base.h, base.s / 100, l / 100).toString();
  }

  function render(container, data) {
    var d3 = window.d3;
    var width = container.clientWidth;
    var size = Math.min(width, 720);
    var radius = size / 2;

    var root = buildHierarchy(data);
    var currentSelection = null;
    // Index id -> hierarchy node so the panel's child links can drill in.
    var byId = {};
    root.each(function (n) { byId[n.id] = n; });
    // Cache initial layout co-ordinates on each node so zoom transitions
    // can interpolate from "current" to "target".
    d3.partition().size([2 * Math.PI, radius])(root);
    root.each(function (d) {
      d.current = { x0: d.x0, x1: d.x1, y0: d.y0, y1: d.y1 };
    });

    var arc = d3
      .arc()
      .startAngle(function (d) { return d.x0; })
      .endAngle(function (d) { return d.x1; })
      .padAngle(function (d) { return Math.min((d.x1 - d.x0) / 2, 0.004); })
      .padRadius(radius / 2)
      .innerRadius(function (d) { return d.y0; })
      .outerRadius(function (d) { return d.y1 - 1; });

    container.innerHTML = "";
    var svg = d3
      .select(container)
      .append("svg")
      .attr("viewBox", [-radius, -radius, size, size])
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("class", "ts-sunburst-svg");

    // Centre invisible click-target for zoom-out.
    svg
      .append("circle")
      .attr("r", radius / 5)
      .attr("class", "ts-sunburst-centre")
      .on("click", function () { zoomTo(root); });

    var centreLabel = svg
      .append("text")
      .attr("class", "ts-sunburst-centre-label")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .text("ts_shape");

    var paths = svg
      .selectAll("path.arc")
      .data(root.descendants().filter(function (d) { return d.depth; }))
      .join("path")
      .attr("class", "arc")
      .attr("fill", colourFor)
      .attr("fill-opacity", function (d) {
        return arcVisible(d.current) ? (d.children ? 0.85 : 0.95) : 0;
      })
      .attr("pointer-events", function (d) {
        return arcVisible(d.current) ? "auto" : "none";
      })
      .attr("d", function (d) { return arc(d.current); })
      .style("cursor", function (d) {
        return d.children || d.data.url ? "pointer" : "default";
      });

    // Hover + click handlers
    var breadcrumb = document.getElementById("ts-breadcrumb");
    var tooltip = ensureTooltip(container);

    paths
      .on("mouseover", function (event, d) {
        d3.select(this).attr("fill-opacity", 1);
        updateBreadcrumb(breadcrumb, d);
        var t = d.data;
        var lines = [
          "<b>" + t.label + "</b> <span class='ts-type'>(" + t.type + ")</span>",
        ];
        if (t.shape) lines.push("shape: " + t.shape);
        if (t.pack) lines.push("pack: " + t.pack);
        tooltip.innerHTML = lines.join("<br>");
        tooltip.style.display = "block";
      })
      .on("mousemove", function (event) {
        var rect = container.getBoundingClientRect();
        tooltip.style.left = event.clientX - rect.left + 12 + "px";
        tooltip.style.top = event.clientY - rect.top + 12 + "px";
      })
      .on("mouseout", function (event, d) {
        d3.select(this).attr("fill-opacity", function (d2) {
          return arcVisible(d2.current) ? (d2.children ? 0.85 : 0.95) : 0;
        });
        // Don't clear breadcrumb on mouseout — it reflects the
        // currently-selected node, not the hovered one.
        tooltip.style.display = "none";
      })
      .on("click", function (event, d) {
        // Click always selects + updates the details panel. Containers
        // also zoom; leaves don't navigate from a click (the panel's
        // "Open reference" button does that). This separation makes
        // exploration safe on touch devices.
        selectNode(d);
        if (d.children) zoomTo(d);
      });

    // Initially focus the root so the details panel shows something
    // meaningful before any click.
    selectNode(root);

    function selectNode(d) {
      currentSelection = d;
      updateBreadcrumb(breadcrumb, d.depth ? d : null);
      renderDetails(d);
    }

    // ----------------------------------------------------------------
    // Details panel below the sunburst.
    //
    // Renders the currently-selected node: title, type badge, breadcrumb
    // path with each segment clickable, description (pulled from the
    // Python docstrings at build time), file path, child list as
    // drill-in chips, and an "Open reference docs" CTA for class /
    // method nodes. Replaces the previous hover-tooltip-only model,
    // which was unusable on touch.
    // ----------------------------------------------------------------
    function renderDetails(node) {
      var panel = document.getElementById("ts-details");
      if (!panel) return;

      var d = node.data;
      var path = [];
      var n = node;
      while (n && n.depth > 0) {
        path.unshift(n);
        n = n.parent;
      }

      var html = [];
      html.push('<div class="ts-details-head">');
      html.push('<h2 class="ts-details-title">' + esc(d.label) + "</h2>");
      html.push(
        '<span class="ts-details-type ts-details-type--' +
          esc(d.type) +
          '">' +
          esc(d.type) +
          (d.shape ? " · " + esc(d.shape) : "") +
          "</span>"
      );
      html.push("</div>");

      html.push('<div class="ts-details-path">');
      html.push(
        '<a href="#" data-target="' +
          esc(root.id) +
          '" class="ts-crumb-link ts-crumb-root">ts_shape</a>'
      );
      path.forEach(function (p, i) {
        html.push(' <span class="ts-crumb-sep">›</span> ');
        if (i === path.length - 1) {
          html.push(
            '<span class="ts-crumb-current">' + esc(p.data.label) + "</span>"
          );
        } else {
          html.push(
            '<a href="#" data-target="' +
              esc(p.id) +
              '" class="ts-crumb-link">' +
              esc(p.data.label) +
              "</a>"
          );
        }
      });
      html.push("</div>");

      if (d.description) {
        html.push(
          '<p class="ts-details-desc">' + renderDescription(d.description) + "</p>"
        );
      } else {
        html.push(
          '<p class="ts-details-desc ts-details-desc--empty">No description available.</p>'
        );
      }

      if (d.file) {
        html.push(
          '<div class="ts-details-file"><code>' + esc(d.file) + "</code></div>"
        );
      }

      if (node.children && node.children.length) {
        var counts = countDescendants(node);
        var summaryBits = [];
        if (counts.pack)
          summaryBits.push(counts.pack + " pack" + (counts.pack > 1 ? "s" : ""));
        if (counts.class)
          summaryBits.push(
            counts.class + " class" + (counts.class > 1 ? "es" : "")
          );
        if (counts.method)
          summaryBits.push(
            counts.method + " method" + (counts.method > 1 ? "s" : "")
          );
        if (summaryBits.length) {
          html.push(
            '<div class="ts-details-counts">Contains ' +
              summaryBits.join(" · ") +
              "</div>"
          );
        }

        var children = node.children.slice().sort(function (a, b) {
          return (a.data.label || "").localeCompare(b.data.label || "");
        });
        var childTypeLabel = children[0].data.type;
        html.push(
          '<div class="ts-details-children-label">' +
            esc(
              childTypeLabel + (children.length > 1 ? "s" : "")
            ) +
            " (" +
            children.length +
            "):</div>"
        );
        html.push('<div class="ts-details-children">');
        children.forEach(function (c) {
          var shapeBadge = c.data.shape
            ? ' <span class="ts-shape-badge ts-shape-badge--' +
              esc(c.data.shape) +
              '">' +
              esc(c.data.shape) +
              "</span>"
            : "";
          html.push(
            '<a href="#" data-target="' +
              esc(c.id) +
              '" class="ts-child-chip ts-child-chip--' +
              esc(c.data.type) +
              '">' +
              esc(c.data.label) +
              shapeBadge +
              "</a>"
          );
        });
        html.push("</div>");
      }

      if (d.url) {
        html.push(
          '<a class="ts-details-cta" href="' +
            esc(d.url) +
            '" target="_blank" rel="noopener">Open reference docs ↗</a>'
        );
      }

      panel.innerHTML = html.join("");

      // Wire breadcrumb + child links to re-select + zoom in the chart.
      panel.querySelectorAll("[data-target]").forEach(function (a) {
        a.addEventListener("click", function (ev) {
          ev.preventDefault();
          var t = byId[a.getAttribute("data-target")];
          if (!t) return;
          selectNode(t);
          if (t.children) zoomTo(t);
        });
      });
    }

    // ----------------------------------------------------------------
    // Zoom — Bostock's standard zoomable-sunburst transform
    // ----------------------------------------------------------------
    function zoomTo(p) {
      centreLabel.text(p.depth === 0 ? "ts_shape" : p.data.label);
      root.each(function (d) {
        d.target = {
          x0:
            Math.max(0, Math.min(1, (d.x0 - p.x0) / (p.x1 - p.x0))) *
            2 *
            Math.PI,
          x1:
            Math.max(0, Math.min(1, (d.x1 - p.x0) / (p.x1 - p.x0))) *
            2 *
            Math.PI,
          y0: Math.max(0, d.y0 - p.y0),
          y1: Math.max(0, d.y1 - p.y0),
        };
      });

      var t = svg.transition().duration(650);
      paths
        .transition(t)
        .tween("data", function (d) {
          var i = d3.interpolate(d.current, d.target);
          return function (it) { d.current = i(it); };
        })
        .filter(function (d) {
          return +this.getAttribute("fill-opacity") || arcVisible(d.target);
        })
        .attr("fill-opacity", function (d) {
          return arcVisible(d.target) ? (d.children ? 0.85 : 0.95) : 0;
        })
        .attr("pointer-events", function (d) {
          return arcVisible(d.target) ? "auto" : "none";
        })
        .attrTween("d", function (d) {
          return function () { return arc(d.current); };
        });
    }

    // ----------------------------------------------------------------
    // Search — fade non-matching arcs
    // ----------------------------------------------------------------
    var searchInput = document.getElementById("ts-search");
    if (searchInput) {
      searchInput.addEventListener("input", function () {
        var q = this.value.trim().toLowerCase();
        if (!q) {
          paths.attr("fill-opacity", function (d) {
            return arcVisible(d.current) ? (d.children ? 0.85 : 0.95) : 0;
          });
          return;
        }
        paths.attr("fill-opacity", function (d) {
          if (!arcVisible(d.current)) return 0;
          var match =
            (d.data.label || "").toLowerCase().indexOf(q) >= 0 ||
            (d.data.pack || "").toLowerCase().indexOf(q) >= 0;
          if (match) return d.children ? 0.9 : 1;
          // If any descendant matches, keep parent visible at reduced opacity.
          var hasMatch = false;
          d.each(function (dd) {
            if (
              (dd.data.label || "").toLowerCase().indexOf(q) >= 0 ||
              (dd.data.pack || "").toLowerCase().indexOf(q) >= 0
            ) {
              hasMatch = true;
            }
          });
          return hasMatch ? 0.4 : 0.05;
        });
      });
    }
  }

  function arcVisible(d) {
    return d.y1 <= 1e9 && d.y0 >= 0 && d.x1 > d.x0;
  }

  function countDescendants(node) {
    var c = { pack: 0, class: 0, method: 0 };
    node.each(function (d) {
      if (d === node) return;
      if (c[d.data.type] != null) c[d.data.type]++;
    });
    return c;
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (ch) {
      return (
        {
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        }[ch] || ch
      );
    });
  }

  // Lightweight inline-markdown for description text. Recognises the
  // backtick conventions used in ts_shape docstrings — both single
  // (`name`) and Sphinx-style double (``name``) become <code>name</code>.
  // Everything outside backticks is HTML-escaped. Match the doubles
  // first so a chunk like ``foo`` isn't mistaken for two empty `` pairs.
  function renderDescription(text) {
    if (!text) return "";
    var out = "";
    var last = 0;
    var re = /``([^`\n]+?)``|`([^`\n]+?)`/g;
    var m;
    while ((m = re.exec(text)) !== null) {
      out += esc(text.slice(last, m.index));
      out += "<code>" + esc(m[1] || m[2]) + "</code>";
      last = re.lastIndex;
    }
    out += esc(text.slice(last));
    return out;
  }

  function updateBreadcrumb(el, d) {
    if (!el) return;
    if (!d) {
      el.textContent = "ts_shape";
      return;
    }
    var parts = [];
    var n = d;
    while (n && n.depth > 0) {
      parts.unshift(n.data.label);
      n = n.parent;
    }
    el.innerHTML =
      '<span class="ts-crumb-root">ts_shape</span>' +
      parts
        .map(function (p) { return ' <span class="ts-crumb-sep">›</span> ' + p; })
        .join("");
  }

  function ensureTooltip(container) {
    var tip = container.querySelector(".ts-sunburst-tooltip");
    if (tip) return tip;
    tip = document.createElement("div");
    tip.className = "ts-sunburst-tooltip";
    tip.style.display = "none";
    container.appendChild(tip);
    return tip;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
