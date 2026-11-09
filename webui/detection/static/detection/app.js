/* AutoAD web interface: charts, busy buttons and confirmations. */
(function () {
  "use strict";

  var VAR = /^var\((--[\w-]+)\)$/;
  var CONFIG = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
  };

  // Figures use "var(--token)" for colors; Plotly needs the actual values.
  function resolve(value, styles) {
    if (typeof value === "string") {
      var match = value.match(VAR);
      return match ? styles.getPropertyValue(match[1]).trim() || value : value;
    }
    if (Array.isArray(value)) {
      // Data arrays never hold tokens; only walk arrays of objects (traces, shapes).
      return value.length && typeof value[0] === "object" && value[0] !== null && !Array.isArray(value[0])
        ? value.map(function (item) { return resolve(item, styles); })
        : value;
    }
    if (value && typeof value === "object") {
      var out = {};
      Object.keys(value).forEach(function (key) { out[key] = resolve(value[key], styles); });
      return out;
    }
    return value;
  }

  function renderCharts() {
    var source = document.getElementById("chart-data");
    if (!source) return;
    var elements = document.querySelectorAll("[data-chart]");
    if (!window.Plotly) {
      elements.forEach(function (el) {
        el.textContent = "The charts could not be loaded: plotly.min.js is missing (pip install plotly).";
      });
      return;
    }
    var figures = JSON.parse(source.textContent);
    var styles = getComputedStyle(document.documentElement);
    elements.forEach(function (el) {
      var figure = figures[el.dataset.chart];
      if (!figure) return;
      var resolved = resolve(figure, styles);
      window.Plotly.react(el, resolved.data, resolved.layout, CONFIG);
      if (el.dataset.rowUrl && !el.dataset.bound) {
        el.dataset.bound = "1";
        el.on("plotly_click", function (event) {
          var point = event.points && event.points[0];
          if (!point || point.customdata == null) return;
          var row = Array.isArray(point.customdata) ? point.customdata[0] : point.customdata;
          window.location.href = el.dataset.rowUrl.replace("__row__", String(row));
        });
      }
    });
  }

  function busyForms() {
    document.querySelectorAll("form[data-busy]").forEach(function (form) {
      form.addEventListener("submit", function () {
        var button = form.querySelector("button[type=submit]");
        if (!button) return;
        // Disable after the submit has been dispatched, so the form is still sent.
        window.setTimeout(function () {
          button.disabled = true;
          button.textContent = form.dataset.busy;
        }, 0);
      });
    });
  }

  function confirmForms() {
    document.querySelectorAll("form[data-confirm]").forEach(function (form) {
      form.addEventListener("submit", function (event) {
        if (!window.confirm(form.dataset.confirm)) event.preventDefault();
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderCharts();
    busyForms();
    confirmForms();
  });
  if (window.matchMedia) {
    var dark = window.matchMedia("(prefers-color-scheme: dark)");
    if (dark.addEventListener) dark.addEventListener("change", renderCharts);
  }
})();
