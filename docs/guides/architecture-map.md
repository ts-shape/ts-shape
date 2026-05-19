---
hide:
  - navigation
  - toc
---

# Architecture Map

<p class="ts-map-hint">
  Pan with drag, zoom with scroll. <b>Search</b> finds class and method names.
  <b>Depth</b> controls how deep the hierarchy expands.
  Click any class or method to open its reference page.
  <a href="architecture/">Back to the prose overview →</a>
</p>

<link rel="stylesheet" href="../assets/architecture-graph.css">
<script src="https://unpkg.com/cytoscape@3.30.1/dist/cytoscape.min.js" defer></script>

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
  <div class="ts-graph-mobile-notice">
    Open this page on a wider screen for the interactive map.
  </div>
</div>
