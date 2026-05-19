/* ts-shape architecture graph — Cytoscape.js renderer.
 *
 * Mounts on #ts-shape-graph in docs/guides/architecture.md and consumes
 * assets/architecture-graph.json (generated at build time by
 * scripts/gen_architecture_graph.py from ts_shape.eventlog.taxonomy.REGISTRY
 * and a walk of src/ts_shape/). Click-throughs target the auto-generated
 * mkdocstrings reference pages only.
 */
(function () {
  "use strict";

  function init() {
    var container = document.getElementById("ts-shape-graph");
    if (!container || typeof window.cytoscape !== "function") {
      // Not on the architecture page, or Cytoscape failed to load.
      if (container) {
        container.innerHTML =
          '<p class="ts-graph-fallback">Interactive graph failed to load. ' +
          "See the Mermaid diagram above for the static architecture overview.</p>";
      }
      return;
    }

    // Resolve the JSON URL relative to the architecture page itself.
    // Material URL is /guides/architecture/ so we walk up two levels.
    var jsonUrl = new URL("../../assets/architecture-graph.json", document.baseURI).toString();

    fetch(jsonUrl)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        renderGraph(container, data);
      })
      .catch(function (err) {
        container.innerHTML =
          '<p class="ts-graph-fallback">Could not load graph data (' +
          err.message +
          "). See the Mermaid diagram above.</p>";
      });
  }

  // Pleasant fixed colours per layer; matches the Mermaid scheme on the page.
  var LAYER_COLOURS = {
    loader: "#38bdf8",
    transform: "#2dd4bf",
    features: "#2dd4bf",
    context: "#2dd4bf",
    events: "#f59e0b",
    eventlog: "#fbbf24",
    utils: "#a1a1aa",
  };

  function depthVisibleTypes(depth) {
    switch (depth) {
      case "layers":
        return ["layer"];
      case "packs":
        return ["layer", "pack"];
      case "classes":
        return ["layer", "pack", "class"];
      default:
        return ["layer", "pack", "class", "method"];
    }
  }

  function renderGraph(container, data) {
    var cy = window.cytoscape({
      container: container,
      elements: data,
      wheelSensitivity: 0.2,
      layout: { name: "preset" }, // overridden after filter applies
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-size": 11,
            "text-valign": "center",
            "text-halign": "center",
            color: "#0f172a",
            "background-color": "#e2e8f0",
            "border-width": 1,
            "border-color": "#94a3b8",
            "text-wrap": "wrap",
            "text-max-width": 90,
            padding: 6,
          },
        },
        {
          selector: 'node[type = "layer"]',
          style: {
            "background-opacity": 0.18,
            "background-color": function (ele) {
              return LAYER_COLOURS[ele.data("label")] || "#cbd5e1";
            },
            "border-color": function (ele) {
              return LAYER_COLOURS[ele.data("label")] || "#94a3b8";
            },
            "border-width": 2,
            "font-size": 14,
            "font-weight": "bold",
            "text-valign": "top",
            "text-margin-y": -4,
            shape: "round-rectangle",
            padding: 16,
          },
        },
        {
          selector: 'node[type = "pack"]',
          style: {
            "background-opacity": 0.4,
            "background-color": "#f1f5f9",
            "border-color": "#64748b",
            shape: "round-rectangle",
            "font-weight": "bold",
            "text-valign": "top",
            "text-margin-y": -2,
            padding: 10,
          },
        },
        {
          selector: 'node[type = "class"]',
          style: {
            "background-color": "#fef3c7",
            "border-color": "#f59e0b",
            shape: "round-rectangle",
            "font-weight": "bold",
          },
        },
        {
          selector: 'node[type = "method"]',
          style: {
            "background-color": "#ffffff",
            "border-color": "#cbd5e1",
            shape: "round-rectangle",
            "font-size": 10,
          },
        },
        {
          selector: 'node[type = "method"][shape = "point"]',
          style: { "border-color": "#3b82f6" },
        },
        {
          selector: 'node[type = "method"][shape = "interval"]',
          style: { "border-color": "#10b981" },
        },
        {
          selector: 'node[type = "method"][shape = "summary"]',
          style: { "border-color": "#f59e0b" },
        },
        {
          selector: 'node[type = "method"][shape = "static"]',
          style: { "border-color": "#a78bfa" },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#94a3b8",
            "target-arrow-color": "#94a3b8",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
        {
          selector: ".hit",
          style: {
            "background-color": "#fde68a",
            "border-color": "#f59e0b",
            "border-width": 3,
          },
        },
        {
          selector: ".dim",
          style: { opacity: 0.2 },
        },
      ],
    });

    // ----------------------------------------------------------------
    // Controls
    // ----------------------------------------------------------------
    var controls = document.getElementById("ts-graph-controls");

    function applyDepthFilter() {
      var depth = controls.querySelector('input[name="ts-depth"]:checked').value;
      var visible = depthVisibleTypes(depth);
      cy.batch(function () {
        cy.nodes().forEach(function (n) {
          n.style("display", visible.indexOf(n.data("type")) >= 0 ? "element" : "none");
        });
      });
      // `cose` is Cytoscape's built-in physics layout. It handles compound
      // (parent/child) graphs well and doesn't require an extension script.
      cy.layout({
        name: "cose",
        animate: false,
        nodeDimensionsIncludeLabels: true,
        padding: 24,
        randomize: false,
        componentSpacing: 60,
        nodeRepulsion: function () {
          return 8000;
        },
        idealEdgeLength: function () {
          return 80;
        },
        nestingFactor: 0.6,
        gravity: 0.4,
        numIter: 1200,
      }).run();
      cy.fit(undefined, 30);
    }

    function applyLayerFilter() {
      var hidden = new Set();
      controls.querySelectorAll('input[name="ts-layer"]').forEach(function (cb) {
        if (!cb.checked) hidden.add(cb.value);
      });
      cy.batch(function () {
        cy.nodes().forEach(function (n) {
          var parent = n.data("parent") || n.data("id");
          var layerId = parent.split(":")[0] === "layer" ? parent.split(":")[1] : null;
          if (!layerId) {
            // Resolve up to the topmost ancestor.
            var anc = n.ancestors().filter('node[type = "layer"]').first();
            layerId = anc.length ? anc.data("label") : null;
          }
          if (n.data("type") === "layer") layerId = n.data("label");
          if (layerId && hidden.has(layerId)) {
            n.style("display", "none");
          }
        });
      });
    }

    function applySearch() {
      var q = controls.querySelector("#ts-search").value.trim().toLowerCase();
      cy.elements().removeClass("hit dim");
      if (!q) return;
      var hits = cy.nodes().filter(function (n) {
        var d = n.data();
        return (
          (d.label || "").toLowerCase().indexOf(q) >= 0 ||
          (d.pack || "").toLowerCase().indexOf(q) >= 0
        );
      });
      if (hits.length === 0) return;
      cy.elements().addClass("dim");
      hits.removeClass("dim").addClass("hit");
      hits.ancestors().removeClass("dim");
      cy.animate({ fit: { eles: hits, padding: 60 }, duration: 400 });
    }

    controls.querySelectorAll('input[name="ts-depth"]').forEach(function (el) {
      el.addEventListener("change", function () {
        applyDepthFilter();
        applyLayerFilter();
        applySearch();
        try {
          localStorage.setItem("ts-graph-depth", el.value);
        } catch (e) {
          /* localStorage unavailable */
        }
      });
    });
    controls.querySelectorAll('input[name="ts-layer"]').forEach(function (el) {
      el.addEventListener("change", function () {
        applyLayerFilter();
        applySearch();
      });
    });
    controls.querySelector("#ts-search").addEventListener("input", applySearch);
    controls.querySelector("#ts-reset").addEventListener("click", function () {
      controls.querySelector("#ts-search").value = "";
      applySearch();
      cy.animate({ fit: { padding: 30 }, duration: 400 });
    });

    // Click-through to mkdocstrings reference page.
    cy.on("tap", "node", function (evt) {
      var t = evt.target;
      var type = t.data("type");
      if (type === "layer" || type === "pack") {
        // Toggle children visibility.
        var children = t.descendants();
        var anyVisible = children.some(function (c) {
          return c.style("display") !== "none";
        });
        children.style("display", anyVisible ? "none" : "element");
        return;
      }
      var url = t.data("url");
      if (url) window.location.href = url;
    });

    // Hover tooltip via Cytoscape qtip alternative — use the native title attr
    // on the canvas by binding mouseover / mouseout to a floating div.
    var tip = document.createElement("div");
    tip.className = "ts-graph-tooltip";
    tip.style.display = "none";
    container.appendChild(tip);

    cy.on("mouseover", "node", function (evt) {
      var d = evt.target.data();
      var lines = ["<b>" + (d.label || "") + "</b> (" + (d.type || "") + ")"];
      if (d.pack) lines.push("pack: " + d.pack);
      if (d.shape) lines.push("shape: " + d.shape);
      if (d.file) lines.push("<code>" + d.file + "</code>");
      tip.innerHTML = lines.join("<br>");
      tip.style.display = "block";
    });
    cy.on("mousemove", "node", function (evt) {
      var rect = container.getBoundingClientRect();
      tip.style.left = evt.originalEvent.clientX - rect.left + 12 + "px";
      tip.style.top = evt.originalEvent.clientY - rect.top + 12 + "px";
    });
    cy.on("mouseout", "node", function () {
      tip.style.display = "none";
    });

    // Restore preferred depth from localStorage.
    var pref = null;
    try {
      pref = localStorage.getItem("ts-graph-depth");
    } catch (e) {
      /* localStorage unavailable */
    }
    if (pref) {
      var radio = controls.querySelector(
        'input[name="ts-depth"][value="' + pref + '"]'
      );
      if (radio) radio.checked = true;
    }

    applyDepthFilter();
    applyLayerFilter();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
