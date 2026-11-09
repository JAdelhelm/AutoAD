"""Table views of a scoring: rows with marked cells, findings per column, one row explained."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from django.core.paginator import Paginator

from autoad.preprocessing.type_inference import TypeInferenceTransformer, clean_missing

#: Finding keys of ``AutoAD.findings`` and how the interface names them.
FINDINGS = {
    "Tukey": "Outside Tukey's fences",
    "MAD": "High modified z-score",
    "Missing": "Missing or unreadable",
    "Unseen": "Category not seen in training",
}
SHOW_CHOICES = {
    "all": "All rows",
    "anomalies": "Anomalies",
    "findings": "Rows with findings",
    "duplicates": "Duplicates, by group",
}
ORDER_CHOICES = {"score": "Highest score first", "row": "File order"}


def format_value(value) -> str:
    """Compact text for a cell; missing values become an empty string."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # lists and other non-scalars
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def score_decimals(scores: np.ndarray) -> int:
    """Decimals that show the largest score with about four significant digits.

    One precision for all scores of a table keeps them comparable, and values near
    zero read as 0.0000 instead of 2.24e-17.
    """
    largest = float(np.max(np.abs(scores))) if len(scores) else 0.0
    if not largest or not math.isfinite(largest):
        return 4
    return int(min(max(3 - math.floor(math.log10(largest)), 0), 10))


def format_score(value: float, decimals: int = 4) -> str:
    text = f"{value:.{decimals}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def column_overview(frame: pd.DataFrame, sample_rows: int = 5000) -> list[dict]:
    """Per column: the type AutoAD would infer, distinct values, missing share, examples.

    Types are inferred on a sample, so they are a preview: the run itself uses all rows.
    """
    sample = frame if len(frame) <= sample_rows else frame.sample(n=sample_rows, random_state=0)
    try:
        types = TypeInferenceTransformer().fit(sample).column_types_
    except (TypeError, ValueError):
        types = {}
    overview = []
    for column in frame.columns:
        series = clean_missing(frame[column])
        examples = [format_value(value)[:30] for value in series.dropna().unique()[:3]]
        overview.append(
            {
                "name": column,
                "type": types.get(column, "unknown"),
                "distinct": int(series.nunique()),
                "missing": float(series.isna().mean()),
                "examples": ", ".join(examples),
            }
        )
    return overview


def _cell_kinds(findings: dict, positions: np.ndarray) -> list[dict]:
    """Per row (in ``positions`` order): column -> list of finding keys."""
    rows = [{} for _ in positions]
    for kind, flags in findings.items():
        if flags.shape[1] == 0:
            continue
        values = flags.to_numpy(dtype=bool)[positions]
        columns = list(flags.columns)
        for i, j in zip(*np.nonzero(values), strict=True):
            rows[i].setdefault(columns[j], []).append(kind)
    return rows


def ranks(scores: np.ndarray) -> np.ndarray:
    """1 for the highest score; ties keep file order."""
    order = np.argsort(-scores, kind="stable")
    result = np.empty(len(scores), dtype=int)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def rows_with_findings(findings: dict) -> np.ndarray:
    any_finding = None
    for flags in findings.values():
        row_any = flags.to_numpy(dtype=bool).any(axis=1) if flags.shape[1] else np.zeros(len(flags), dtype=bool)
        any_finding = row_any if any_finding is None else any_finding | row_any
    return any_finding


@dataclass
class Cell:
    text: str
    kinds: list = field(default_factory=list)

    @property
    def css(self) -> str:
        classes = []
        if "Tukey" in self.kinds or "MAD" in self.kinds:
            classes.append("cell-extreme")
        if "Missing" in self.kinds:
            classes.append("cell-missing")
        if "Unseen" in self.kinds:
            classes.append("cell-unseen")
        return " ".join(classes)

    @property
    def title(self) -> str:
        return "; ".join(FINDINGS[kind] for kind in self.kinds)


def row_table(result: dict, *, show: str, order: str, page: str | int | None, per_page: int) -> dict:
    report = result["report"]
    columns = result["columns"]
    scores = report["AD_score"].to_numpy(dtype=float)
    decimals = score_decimals(scores)
    labels = np.asarray(result["labels"], dtype=bool)
    rank = ranks(scores)

    duplicates = duplicate_info(result)
    if show == "anomalies":
        positions = np.flatnonzero(labels)
    elif show == "findings":
        positions = np.flatnonzero(rows_with_findings(result["findings"]))
    elif show == "duplicates":
        # Members of a group next to each other, so their values can be compared.
        positions = np.flatnonzero(duplicates["is_duplicate"])
        positions = positions[np.argsort(duplicates["group"][positions], kind="stable")]
    else:
        positions = np.arange(len(report))
    if order == "score" and show != "duplicates":
        positions = positions[np.argsort(rank[positions], kind="stable")]

    paginator = Paginator(positions, per_page)
    page_obj = paginator.get_page(page)
    shown = np.asarray(page_obj.object_list, dtype=int)
    kinds = _cell_kinds(result["findings"], shown)
    values = report[columns].iloc[shown]
    rows = []
    for i, position in enumerate(shown):
        rows.append(
            {
                "number": int(position) + 1,
                "score": format_score(scores[position], decimals),
                "rank": int(rank[position]),
                "is_anomaly": bool(labels[position]),
                "duplicate": _duplicate_of(duplicates, position),
                "cells": [
                    Cell(format_value(value), kinds[i].get(col, []))
                    for col, value in zip(columns, values.iloc[i], strict=True)
                ],
            }
        )
    return {
        "columns": columns,
        "rows": rows,
        "page": page_obj,
        "count": len(positions),
        "threshold": format_score(result["threshold"], decimals),
    }


def duplicate_info(result: dict) -> dict:
    """Arrays per row position from ``AutoAD.duplicates``: flag, group, closest row, score."""
    duplicates = result["duplicates"]
    report = result["report"]
    return {
        "is_duplicate": duplicates["is_duplicate"].to_numpy(dtype=bool),
        "group": duplicates["Duplicate_group"].fillna(0).to_numpy(dtype=int),
        "closest": report.index.get_indexer(duplicates["Closest_row"]),
        "score": duplicates["Duplicate_score"].to_numpy(dtype=float),
        "differing": duplicates["Differing_Columns"].to_numpy(dtype=object),
    }


def _duplicate_of(duplicates: dict, position: int) -> dict | None:
    if not duplicates["is_duplicate"][position]:
        return None
    return {"group": int(duplicates["group"][position]), "closest": int(duplicates["closest"][position]) + 1}


def _closest_row(result: dict, position: int) -> dict:
    """The closest other row of a row, for the row page."""
    duplicates = duplicate_info(result)
    scores = duplicates["score"]
    decimals = score_decimals(scores[np.isfinite(scores)])
    return {
        "is_duplicate": bool(duplicates["is_duplicate"][position]),
        "group": int(duplicates["group"][position]),
        "closest": int(duplicates["closest"][position]) + 1 if duplicates["closest"][position] >= 0 else None,
        "score": format_score(float(scores[position]), decimals) if np.isfinite(scores[position]) else "",
        "differing": duplicates["differing"][position],
    }


def duplicate_groups(result: dict, limit: int = 100) -> dict:
    """Groups of rows that duplicate each other, for the results page."""
    duplicates = duplicate_info(result)
    groups = duplicates["group"]
    positions = np.flatnonzero(duplicates["is_duplicate"])
    decimals = score_decimals(duplicates["score"][np.isfinite(duplicates["score"])])
    entries = []
    for number in np.unique(groups[positions])[:limit]:
        members = positions[groups[positions] == number]
        # Differing_Columns is relative to each row's closest row; usually the same text for all members.
        differing = sorted({text for text in duplicates["differing"][members] if text})
        entries.append(
            {
                "number": int(number),
                "rows": [int(position) + 1 for position in members],
                "identical": not differing,
                "differing": "; ".join(differing),
                "score": "0" if not differing else format_score(float(duplicates["score"][members].max()), decimals),
            }
        )
    return {"groups": entries, "count": int(len(np.unique(groups[positions]))), "rows": int(len(positions))}


def compared_columns(ad, columns: list) -> tuple[list, list]:
    """Columns the duplicate check compares, and the ones it ignores."""
    ignored = [col for col in columns if col in ad.excluded_columns_ or col in ad.dropped_columns_]
    return [col for col in columns if col not in ignored], ignored


def _column_status(ad, column: str) -> str:
    if column in ad.excluded_columns_:
        return "ignored"
    reason = ad.dropped_columns_.get(column)
    if reason:
        return "dropped: ID-like" if reason == "id" else f"dropped: {reason}"
    kind = ad.column_types_.get(column, "")
    if column in ad.column_groups_["binary"]:
        return f"{kind}, binary code"
    if column in ad.column_groups_["frequency"]:
        return f"{kind}, frequency"
    return kind


def column_findings(result: dict, ad) -> list[dict]:
    """One entry per input column: how it was used and how many rows have each finding."""
    n_rows = len(result["report"])
    findings = result["findings"]
    entries = []
    for column in result["columns"]:
        counts = {}
        any_rows = np.zeros(n_rows, dtype=bool)
        for kind, flags in findings.items():
            if column in flags.columns:
                flagged = flags[column].to_numpy(dtype=bool)
                counts[kind] = int(flagged.sum())
                any_rows |= flagged
            else:
                counts[kind] = None
        total = int(any_rows.sum())
        share = total / n_rows if n_rows else 0.0
        entries.append(
            {
                "name": column,
                "status": _column_status(ad, column),
                "counts": counts,
                "rows": total,
                "share": share,
                "width": f"{share * 100:.2f}",  # CSS width in percent, locale independent
            }
        )
    entries.sort(key=lambda entry: -entry["rows"])
    return entries


def _strip(stats: dict, value: float) -> dict:
    """Positions in percent for the range strip: P1-P99 line, Q1-Q3 box, median, value.

    The line ends at the 1st and 99th percentile rather than at the minimum and maximum,
    so that outliers in the training data do not set the scale.
    """
    low, high = stats["p01"], stats["p99"]
    span = high - low or abs(high) or 1.0
    start, end = low - 0.05 * span, high + 0.05 * span

    def position(x: float) -> float:
        return round(100 * (x - start) / (end - start), 2)

    strip = {
        "range_left": position(low),
        "range_width": round(position(high) - position(low), 2),
        "box_left": position(stats["q1"]),
        "box_width": max(round(position(stats["q3"]) - position(stats["q1"]), 2), 0.5),
        "median": position(stats["median"]),
        "value": None,
        "outside": "",
    }
    if math.isfinite(value):
        if value < start:
            strip["value"], strip["outside"] = 0.0, "below"
        elif value > end:
            strip["value"], strip["outside"] = 100.0, "above"
        else:
            strip["value"] = position(value)
    return strip


def explain_row(result: dict, model: dict, position: int) -> dict:
    """Everything the row page shows about one row (``position`` is 0-based)."""
    ad = model["autoad"]
    report = result["report"]
    statistics = model["statistics"]
    scores = report["AD_score"].to_numpy(dtype=float)
    decimals = score_decimals(scores)
    kinds = _cell_kinds(result["findings"], np.asarray([position]))[0]
    row = report.iloc[position]
    typed = ad.type_inference_.transform(report[list(ad.type_inference_.feature_names_in_)].iloc[[position]]).iloc[0]

    cells = []
    for column in result["columns"]:
        entry = {
            "name": column,
            "value": format_value(row[column]),
            "status": _column_status(ad, column),
            "findings": [FINDINGS[kind] for kind in kinds.get(column, [])],
            "css": Cell("", kinds.get(column, [])).css,
            "strip": None,
            "context": "",
        }
        if column in statistics["numeric"]:
            stats = statistics["numeric"][column]
            value = float(typed[column]) if pd.notna(typed[column]) else math.nan
            entry["strip"] = _strip(stats, value)
            entry["context"] = (
                f"median {format_value(stats['median'])}, middle half {format_value(stats['q1'])} to "
                f"{format_value(stats['q3'])}, 98 % between {format_value(stats['p01'])} and "
                f"{format_value(stats['p99'])}, range {format_value(stats['min'])} to {format_value(stats['max'])}"
            )
        elif column in statistics["categories"] and pd.notna(typed[column]):
            share = statistics["categories"][column].get(typed[column])
            entry["context"] = (
                "not seen in training" if share is None else f"{share:.1%} of the training rows have this value"
            )
        cells.append(entry)
    flagged_first = sorted(cells, key=lambda cell: not cell["findings"])
    rank = ranks(scores)
    by_rank = np.argsort(rank)
    current = int(rank[position]) - 1
    return {
        "number": position + 1,
        "score": format_score(scores[position], decimals),
        "threshold": format_score(result["threshold"], decimals),
        "rank": int(rank[position]),
        "is_anomaly": bool(result["labels"][position]),
        "duplicate": _closest_row(result, position),
        "cells": flagged_first,
        "previous": int(by_rank[current - 1]) + 1 if current > 0 else None,
        "next": int(by_rank[current + 1]) + 1 if current + 1 < len(scores) else None,
    }
