/* In-browser ts-shape REPL, powered by Pyodide.
   Inert on every page that does not contain a #ts-shape-repl container. */
(function () {
  "use strict";

  var STORAGE_KEY = "ts-shape-repl-snippet";

  var DEFAULT_SNIPPET = [
    "import pandas as pd",
    "from ts_shape.features.cycles.cycles_extractor import CycleExtractor",
    "",
    "df = pd.DataFrame({",
    '    "systime": pd.to_datetime([',
    '        "2024-01-01 08:00:00", "2024-01-01 08:00:30",',
    '        "2024-01-01 08:01:00", "2024-01-01 08:01:30",',
    '        "2024-01-01 08:02:00",',
    "    ]),",
    '    "value_bool":    [True, True, False, True, False],',
    '    "value_integer": [0, 0, 1, 0, 1],',
    '    "value_double":  [0.0, 0.0, 0.0, 0.0, 0.0],',
    '    "value_string":  ["", "", "", "", ""],',
    "})",
    "",
    'cycles = CycleExtractor(df, start_uuid="clamp").process_trigger_cycle()',
    "cycles",
    "",
  ].join("\n");

  // Cached across runs so the second run is instant.
  var runtimePromise = null;
  var installPromise = null;

  function loadRuntime() {
    if (!runtimePromise) {
      runtimePromise = loadPyodide();
    }
    return runtimePromise;
  }

  // Resolves to a Pyodide instance with ts-shape installed.
  function ensureReady() {
    return loadRuntime().then(function (py) {
      if (!installPromise) {
        installPromise = py
          .loadPackage(["micropip", "pandas", "numpy", "scipy"])
          .then(function () {
            // deps=False skips ts-shape's cloud/SQL loader dependencies,
            // which are not needed for in-browser feature/event code.
            // TODO: verify ts-shape installs cleanly under Pyodide.
            return py.runPythonAsync(
              "import micropip\n" +
                'await micropip.install("ts-shape", deps=False)\n'
            );
          })
          .catch(function (err) {
            installPromise = null; // allow a retry on the next run
            throw err;
          });
      }
      return installPromise.then(function () {
        return py;
      });
    });
  }

  function init() {
    var root = document.getElementById("ts-shape-repl");
    if (!root) {
      return;
    }

    var editor = root.querySelector(".ts-repl__editor");
    var runBtn = root.querySelector(".ts-repl__run");
    var resetBtn = root.querySelector(".ts-repl__reset");
    var statusEl = root.querySelector(".ts-repl__status");
    var output = root.querySelector(".ts-repl__output");
    if (!editor || !runBtn || !resetBtn || !output) {
      return;
    }

    var firstRun = true;

    function setStatus(text) {
      if (statusEl) {
        statusEl.textContent = text || "";
      }
    }

    function persist() {
      try {
        localStorage.setItem(STORAGE_KEY, editor.value);
      } catch (e) {
        /* localStorage unavailable (private mode) — non-fatal. */
      }
    }

    // Restore the last snippet, or fall back to the default.
    var saved = null;
    try {
      saved = localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      saved = null;
    }
    editor.value = saved != null ? saved : DEFAULT_SNIPPET;

    editor.addEventListener("input", persist);

    // Tab inserts spaces instead of moving focus out of the editor.
    editor.addEventListener("keydown", function (e) {
      if (e.key !== "Tab") {
        return;
      }
      e.preventDefault();
      var start = editor.selectionStart;
      var end = editor.selectionEnd;
      editor.value =
        editor.value.slice(0, start) + "    " + editor.value.slice(end);
      editor.selectionStart = editor.selectionEnd = start + 4;
      persist();
    });

    resetBtn.addEventListener("click", function () {
      editor.value = DEFAULT_SNIPPET;
      persist();
      output.textContent = "";
      output.classList.remove("ts-repl__output--error");
      setStatus("");
    });

    runBtn.addEventListener("click", function () {
      runBtn.disabled = true;
      output.classList.remove("ts-repl__output--error");
      setStatus(firstRun ? "Loading Python runtime…" : "Running…");

      ensureReady()
        .then(function (py) {
          firstRun = false;
          setStatus("Running…");

          // Redirect stdout/stderr into a buffer for this run.
          py.runPython(
            "import sys, io\n" +
              "__ts_buf = io.StringIO()\n" +
              "__ts_out, __ts_err = sys.stdout, sys.stderr\n" +
              "sys.stdout = sys.stderr = __ts_buf\n"
          );

          return py
            .runPythonAsync(editor.value)
            .then(
              function (result) {
                return { py: py, result: result, error: null };
              },
              function (error) {
                return { py: py, result: undefined, error: error };
              }
            );
        })
        .then(function (state) {
          var py = state.py;
          py.runPython("sys.stdout, sys.stderr = __ts_out, __ts_err\n");

          var buf = py.globals.get("__ts_buf");
          var text = buf.getvalue() || "";
          buf.destroy();

          if (state.error) {
            output.classList.add("ts-repl__output--error");
            if (text && text.slice(-1) !== "\n") {
              text += "\n";
            }
            text += state.error.message || String(state.error);
          } else if (state.result !== undefined && state.result !== null) {
            // String() triggers repr() — renders DataFrames as a table.
            var repr = String(state.result);
            if (repr) {
              if (text && text.slice(-1) !== "\n") {
                text += "\n";
              }
              text += repr;
            }
            if (state.result && typeof state.result.destroy === "function") {
              state.result.destroy();
            }
          }

          output.textContent = text || "(no output)";
          setStatus("");
          runBtn.disabled = false;
        })
        .catch(function (err) {
          // Reached only if the runtime/ts-shape failed to load.
          output.classList.add("ts-repl__output--error");
          output.textContent =
            "Could not load ts-shape in the browser.\n\n" +
            "A dependency may not be available for Pyodide. " +
            "Run the snippet locally instead:\n\n" +
            "    pip install ts-shape\n\n" +
            "Details:\n" +
            (err && err.message ? err.message : String(err));
          setStatus("");
          runBtn.disabled = false;
        });
    });

    // Track Material's light/dark palette toggle.
    function syncTheme() {
      root.setAttribute(
        "data-ts-repl-scheme",
        document.body.getAttribute("data-md-color-scheme") || "default"
      );
    }
    syncTheme();
    new MutationObserver(syncTheme).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
