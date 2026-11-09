"""Plotly figures for a scoring, as plain dicts.

Colors are CSS custom properties (``"var(--anomaly)"``); ``app.js`` resolves them in the
browser, so the charts follow the light or dark theme. Anomalies are the one thing that
should stand out: they get the status color *and* a diamond marker, normal rows stay
gray, and every chart with two series has a legend.
"""

from __future__ import annotations

import html

import numpy as np

from .tables import score_decimals

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
NORMAL = "var(--normal)"
ANOMALY = "var(--anomaly)"
SURFACE = "var(--surface)"
BAR = "var(--series-1)"

#: Placeholder in ``data-row-url``; app.js replaces it with the clicked row number.
ROW_PLACEHOLDER = "__row__"


def _axis(title: str | None = None, **extra) -> dict:
    axis = {
        "gridcolor": "var(--grid)",
        "linecolor": "var(--axis)",
        "zeroline": False,
        "tickfont": {"color": "var(--muted)"},
        "automargin": True,
    }
    if title:
        axis["title"] = {"text": title, "font": {"color": "var(--text-secondary)"}}
    axis.update(extra)
    return axis


def _layout(height: int = 320, **extra) -> dict:
    layout = {
        "height": height,
        "margin": {"l": 8, "r": 16, "t": 28, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "size": 12, "color": "var(--text-secondary)"},
        "hoverlabel": {
            "bgcolor": "var(--surface)",
            "bordercolor": "var(--border-strong)",
            "font": {"family": FONT, "color": "var(--text)"},
        },
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": 1.0,
            "yanchor": "bottom",
            "traceorder": "normal",
            "font": {"color": "var(--text)"},
        },
        "hovermode": "closest",
    }
    layout.update(extra)
    return layout


def _threshold_line(value: float, *, axis: str) -> tuple[dict, dict]:
    """A dashed reference line with a label: dashes mark it as a threshold, not a gridline."""
    line = {"color": "var(--text-secondary)", "width": 1, "dash": "dash"}
    if axis == "y":
        shape = {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": value, "y1": value, "line": line}
        note = {"xref": "paper", "x": 1, "y": value, "xanchor": "right", "yanchor": "bottom"}
    else:
        shape = {"type": "line", "yref": "paper", "y0": 0, "y1": 1, "x0": value, "x1": value, "line": line}
        note = {"yref": "paper", "y": 1, "x": value, "xanchor": "left", "yanchor": "top"}
    note.update({"text": "threshold", "showarrow": False, "font": {"color": "var(--text-secondary)", "size": 11}})
    return shape, note


def _marker(anomaly: bool) -> dict:
    if anomaly:
        # A surface-colored ring keeps overlapping diamonds apart.
        return {"color": ANOMALY, "size": 10, "symbol": "diamond", "line": {"color": SURFACE, "width": 1.5}}
    return {"color": NORMAL, "size": 6, "opacity": 0.55}


def plot_positions(labels: np.ndarray, max_points: int, seed: int = 0) -> tuple[np.ndarray, bool]:
    """Rows to draw: every anomaly plus, if there are too many rows, a sample of the rest."""
    n = len(labels)
    if n <= max_points:
        return np.arange(n), False
    anomalies = np.flatnonzero(labels)
    normal = np.flatnonzero(~labels)
    keep = max(max_points - len(anomalies), 0)
    sample = np.random.default_rng(seed).choice(normal, size=min(keep, len(normal)), replace=False)
    return np.sort(np.concatenate([anomalies, sample])), True


def _points(x, y, positions, labels, scores, *, gl: bool, hover: str) -> list:
    traces = []
    for name, anomaly in (("Normal", False), ("Anomaly", True)):
        chosen = positions[labels[positions] == anomaly]
        traces.append(
            {
                "type": "scattergl" if gl else "scatter",
                "mode": "markers",
                "name": name,
                "x": np.asarray(x)[chosen].tolist(),
                "y": np.asarray(y)[chosen].tolist(),
                "customdata": np.column_stack([chosen + 1, scores[chosen]]).tolist(),
                "hovertemplate": hover + f"<extra>{name}</extra>",
                "marker": _marker(anomaly),
            }
        )
    return traces


def score_by_row(scores: np.ndarray, labels: np.ndarray, threshold: float, max_points: int, decimals: int) -> dict:
    positions, sampled = plot_positions(labels, max_points)
    rows = np.arange(1, len(scores) + 1)
    data = _points(
        rows,
        scores,
        positions,
        labels,
        scores,
        gl=len(positions) > 5000,
        hover=f"Row %{{customdata[0]}}<br>Score %{{customdata[1]:.{decimals}f}}",
    )
    shape, note = _threshold_line(threshold, axis="y")
    layout = _layout(
        xaxis=_axis("Row in file", rangemode="tozero"),
        yaxis=_axis("Anomaly score"),
        shapes=[shape],
        annotations=[note],
    )
    return {"data": data, "layout": layout, "sampled": sampled}


def score_histogram(scores: np.ndarray, labels: np.ndarray, threshold: float, decimals: int) -> dict:
    n_bins = int(np.clip(np.sqrt(len(scores)), 10, 40))
    counts, edges = np.histogram(scores, bins=n_bins)
    anomaly_counts, _ = np.histogram(scores[labels], bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0]) * 0.92  # leaves a small surface gap between bins
    ranges = np.column_stack([edges[:-1], edges[1:]]).tolist()
    data = []
    for name, values, color in (
        ("Normal", counts - anomaly_counts, NORMAL),
        ("Anomaly", anomaly_counts, ANOMALY),
    ):
        data.append(
            {
                "type": "bar",
                "name": name,
                "x": centers.tolist(),
                "y": values.tolist(),
                "width": width,
                "customdata": ranges,
                "marker": {"color": color},
                "hovertemplate": (
                    f"%{{y}} rows<br>Score %{{customdata[0]:.{decimals}f}} to %{{customdata[1]:.{decimals}f}}"
                    f"<extra>{name}</extra>"
                ),
            }
        )
    shape, note = _threshold_line(threshold, axis="x")
    layout = _layout(
        barmode="stack",
        barcornerradius=3,
        xaxis=_axis("Anomaly score"),
        yaxis=_axis("Rows", rangemode="tozero"),
        shapes=[shape],
        annotations=[note],
    )
    return {"data": data, "layout": layout}


def row_map(coords: np.ndarray, labels: np.ndarray, scores: np.ndarray, max_points: int, decimals: int) -> dict:
    positions, sampled = plot_positions(labels, max_points)
    data = _points(
        coords[:, 0],
        coords[:, 1],
        positions,
        labels,
        scores,
        gl=len(positions) > 5000,
        hover=f"Row %{{customdata[0]}}<br>Score %{{customdata[1]:.{decimals}f}}",
    )
    layout = _layout(
        xaxis=_axis("Component 1", showticklabels=False),
        yaxis=_axis("Component 2", showticklabels=False),
    )
    return {"data": data, "layout": layout, "sampled": sampled}


def feature_importances(importances, top: int = 15) -> dict:
    """Horizontal bars, largest first. ``importances`` is ``AutoAD.feature_importances``."""
    head = importances.head(top)
    names = [html.escape(str(name)) for name in head["Feature"]]
    values = head["Importance"].astype(float).tolist()
    data = [
        {
            "type": "bar",
            "orientation": "h",
            "x": values,
            "y": names,
            "text": [f"{value:.3f}" for value in values],
            "textposition": "outside",
            "cliponaxis": False,
            "textfont": {"color": "var(--text-secondary)"},
            "marker": {"color": BAR},
            "hovertemplate": "%{y}<br>Importance %{x:.3f}<extra></extra>",
        }
    ]
    layout = _layout(
        height=28 * len(values) + 70,
        bargap=0.3,
        barcornerradius=4,
        showlegend=False,
        margin={"l": 8, "r": 48, "t": 8, "b": 8},
        xaxis=_axis("Importance", rangemode="tozero"),
        yaxis=_axis(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont={"color": "var(--text)"}),
    )
    return {"data": data, "layout": layout}


def dashboard(result: dict, model: dict, max_points: int) -> dict:
    """All figures of the results page, keyed by the ``data-chart`` name in the template."""
    scores = result["report"]["AD_score"].to_numpy(dtype=float)
    labels = np.asarray(result["labels"], dtype=bool)
    threshold = result["threshold"]
    decimals = score_decimals(scores)
    charts = {
        "scores": score_by_row(scores, labels, threshold, max_points, decimals),
        "histogram": score_histogram(scores, labels, threshold, decimals),
    }
    if result["coords"] is not None:
        charts["map"] = row_map(np.asarray(result["coords"]), labels, scores, max_points, decimals)
    try:
        importances = model["autoad"].feature_importances
    except AttributeError:
        importances = None
    if importances is not None:
        charts["importances"] = feature_importances(importances)
    return charts
