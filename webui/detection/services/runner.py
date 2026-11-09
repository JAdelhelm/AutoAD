"""Fit AutoAD, score tables with it, and store and load the results.

Two joblib files are written:

* the **model** of an analysis: the fitted ``AutoAD``, a PCA that projects its feature
  matrix to 2D for the map of rows, and descriptive statistics of the training data
  for the row explanations;
* the **result** of a scoring: AutoAD's report, the per-cell findings, the duplicates
  within the table, the 0/1 labels and the 2D coordinates.

Both are only ever read back by this module, from files the server wrote itself.
"""

from __future__ import annotations

import io
import logging
import os
import time
from functools import lru_cache
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from django.conf import settings as django_settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from sklearn.decomposition import PCA

from autoad import AutoAD

from ..models import Analysis, Scoring

logger = logging.getLogger(__name__)

DETECTORS = {
    "iforest": {"label": "Isolation Forest (IForest)", "module": "pyod.models.iforest", "class": "IForest"},
    "ecod": {"label": "ECOD", "module": "pyod.models.ecod", "class": "ECOD"},
    "knn": {"label": "k nearest neighbours (KNN)", "module": "pyod.models.knn", "class": "KNN"},
    "lof": {"label": "Local Outlier Factor (LOF)", "module": "pyod.models.lof", "class": "LOF"},
}
#: PyOD's defaults for n_neighbors.
NEIGHBOUR_DEFAULTS = {"knn": 5, "lof": 20}

ROLES = {"auto": "Auto", "numeric": "Number", "datetime": "Date", "exclude": "Ignore"}

DEFAULT_SETTINGS = {
    "detector": "iforest",
    "contamination": 0.1,
    "n_neighbors": None,
    "n_estimators": None,
    "roles": {},
    "drop_constant": True,
    "drop_id_columns": True,
    "max_binary_categories": 20,
    "tukey_factor": 1.5,
    "mad_threshold": 3.5,
    "missing_indicator": "count",
    "scaler": "robust",
    "duplicate_tolerance": 0.0,
}


class AnalysisError(Exception):
    """AutoAD could not run with these data or settings; the message is shown to the user."""


# ------------------------------------------------------------------ configuration


def detector_arguments(settings: dict, n_train: int) -> tuple[dict, list[str]]:
    """Keyword arguments for the PyOD detector, plus notes about adjusted values."""
    name = settings["detector"]
    kwargs = {"contamination": settings["contamination"]}
    notes = []
    if name == "iforest":
        if settings.get("n_estimators"):
            kwargs["n_estimators"] = settings["n_estimators"]
        kwargs["random_state"] = 0
    if name in NEIGHBOUR_DEFAULTS:
        if n_train < 3:
            raise AnalysisError("KNN and LOF need at least 3 training rows.")
        neighbours = settings.get("n_neighbors") or NEIGHBOUR_DEFAULTS[name]
        if neighbours >= n_train:
            notes.append(
                f"n_neighbors was reduced from {neighbours} to {n_train - 1}: it must be smaller than the number "
                "of training rows."
            )
            neighbours = n_train - 1
        kwargs["n_neighbors"] = neighbours
    return kwargs, notes


def build_detector(settings: dict, kwargs: dict):
    spec = DETECTORS[settings["detector"]]
    return getattr(import_module(spec["module"]), spec["class"])(**kwargs)


def autoad_arguments(settings: dict) -> dict:
    """Keyword arguments for ``AutoAD`` (without the detector) from the form settings."""
    roles = settings.get("roles") or {}

    def columns(role):
        return [col for col, value in roles.items() if value == role] or None

    return {
        "datetime_columns": columns("datetime"),
        "numerical_columns": columns("numeric"),
        "exclude_columns": columns("exclude"),
        "drop_constant": settings["drop_constant"],
        "drop_id_columns": settings["drop_id_columns"],
        "max_binary_categories": settings["max_binary_categories"],
        "tukey_factor": settings["tukey_factor"],
        "mad_threshold": settings["mad_threshold"],
        "missing_indicator": None if settings["missing_indicator"] == "none" else settings["missing_indicator"],
        "scaler": None if settings["scaler"] == "none" else settings["scaler"],
    }


def _describe(error: Exception) -> str:
    return str(error) or type(error).__name__


# ------------------------------------------------------------------ fitting and scoring


def fit_analysis(analysis: Analysis, frame: pd.DataFrame, settings: dict) -> Scoring:
    """Fit AutoAD on ``frame``, store the model and score the training data.

    Previous results of the analysis (including checks of new data) are replaced.
    """
    started = time.perf_counter()
    train = frame
    max_rows = django_settings.AUTOAD_WEB_MAX_TRAIN_ROWS
    if len(frame) > max_rows:
        train = frame.sample(n=max_rows, random_state=0).sort_index()

    kwargs, notes = detector_arguments(settings, len(train))
    try:
        ad = AutoAD(build_detector(settings, kwargs), **autoad_arguments(settings)).fit(train)
        features = ad.transform(train).to_numpy()
    except ValueError as error:  # AutoAD explains unusable data or settings this way
        raise AnalysisError(_describe(error)) from error
    except Exception as error:  # anything else from AutoAD, scikit-learn or PyOD
        logger.exception("Fitting AutoAD failed")
        raise AnalysisError(_describe(error)) from error

    model = {
        "autoad": ad,
        "projection": fit_projection(features, np.asarray(ad.detector_.labels_) == 1),
        "statistics": training_statistics(ad, train),
        "detector_kwargs": kwargs,
        "duplicate_tolerance": settings.get("duplicate_tolerance") or 0.0,
        "notes": notes,
    }
    result = compute_result(model, frame)
    fit_seconds = time.perf_counter() - started

    with transaction.atomic():
        for old in analysis.scorings.all():
            old.delete()
        if analysis.model_file:
            analysis.model_file.delete(save=False)
        analysis.model_file.save(f"{analysis.pk}.joblib", ContentFile(_dump(model)), save=False)
        analysis.settings = settings
        analysis.fitted_at = timezone.now()
        analysis.fit_seconds = fit_seconds
        analysis.train_rows = len(train)
        analysis.save()
        return save_scoring(analysis, result, is_training=True, name="Training data", file_name=analysis.file_name)


def check_new_data(analysis: Analysis, frame: pd.DataFrame, *, upload, name: str, read_options: dict) -> Scoring:
    """Score a new table with the fitted model of ``analysis``."""
    result = compute_result(load_model(analysis), frame)
    return save_scoring(
        analysis, result, is_training=False, name=name, file_name=upload.name, upload=upload, read_options=read_options
    )


def compute_result(model: dict, frame: pd.DataFrame) -> dict:
    ad = model["autoad"]
    try:
        report = ad.report(frame)
        findings = ad.findings(frame)
        duplicates = ad.duplicates(frame, tolerance=model.get("duplicate_tolerance", 0.0))
        features = ad.transform(frame).to_numpy() if model["projection"] is not None else None
    except (ValueError, TypeError) as error:  # e.g. columns that do not match the training data
        raise AnalysisError(_describe(error)) from error
    except Exception as error:
        logger.exception("Scoring with AutoAD failed")
        raise AnalysisError(_describe(error)) from error

    scores = report["AD_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise AnalysisError(
            "The detector returned infinite or missing scores. This happens with strongly correlated "
            "columns; try another detector."
        )
    # Same rule as PyOD's predict(), without transforming the data again.
    threshold = float(ad.detector_.threshold_)
    return {
        "columns": [str(col) for col in frame.columns],
        "report": report,
        "findings": findings,
        "duplicates": duplicates,
        "labels": scores > threshold,
        "threshold": threshold,
        "coords": project(model["projection"], features) if features is not None else None,
    }


def fit_projection(features: np.ndarray, anomalies: np.ndarray) -> dict | None:
    """PCA for the map of rows, fitted on the training rows the detector considers normal.

    Fitting without the anomalies keeps the two directions about the normal variation;
    the anomalies then show up wherever they leave it.
    """
    normal = features[~anomalies]
    base = normal if normal.shape[0] >= 2 else features
    if features.shape[1] < 2 or base.shape[0] < 2:
        return None
    pca = PCA(n_components=2, random_state=0).fit(base)
    spread = pca.transform(base).std(axis=0)
    return {"pca": pca, "scale": np.where(spread > 0, spread, 1.0)}


def project(projection: dict, features: np.ndarray) -> np.ndarray:
    """2D coordinates in units of the normal spread, log-compressed beyond 1.

    A single extreme row would otherwise squeeze all other rows into one point.
    """
    coords = projection["pca"].transform(features) / projection["scale"]
    return np.sign(coords) * np.log1p(np.abs(coords))


def training_statistics(ad: AutoAD, frame: pd.DataFrame) -> dict:
    """Percentiles per numeric column and category shares, to put a row's values in context."""
    typed = ad.type_inference_.transform(frame.drop(columns=ad.excluded_columns_))
    numeric = {}
    names = ("min", "p01", "q1", "median", "q3", "p99", "max")
    for col in ad.column_groups_["numeric"]:
        values = typed[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            quantiles = np.quantile(values, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
            numeric[col] = dict(zip(names, map(float, quantiles), strict=True))
    categories = {
        col: typed[col].value_counts(normalize=True).to_dict()
        for col in ad.column_groups_["binary"] + ad.column_groups_["frequency"]
    }
    return {"numeric": numeric, "categories": categories}


def summarize(result: dict) -> dict:
    findings = result["findings"]
    extreme = findings["Tukey"].any(axis=1) | findings["MAD"].any(axis=1)
    return {
        "rows_extreme": int(extreme.sum()),
        "rows_missing": int(findings["Missing"].any(axis=1).sum()),
        "rows_unseen": int(findings["Unseen"].any(axis=1).sum()),
        "rows_duplicate": int(result["duplicates"]["is_duplicate"].sum()),
        "duplicate_groups": int(result["duplicates"]["Duplicate_group"].nunique()),
    }


def save_scoring(
    analysis: Analysis,
    result: dict,
    *,
    is_training: bool,
    name: str,
    file_name: str,
    upload=None,
    read_options: dict | None = None,
) -> Scoring:
    scoring = Scoring(
        analysis=analysis,
        is_training=is_training,
        name=name,
        file_name=file_name,
        read_options=read_options or {},
        n_rows=len(result["report"]),
        n_anomalies=int(result["labels"].sum()),
        threshold=result["threshold"],
        summary=summarize(result),
    )
    if upload is not None:
        upload.seek(0)
        scoring.data_file.save(upload.name, upload, save=False)
    scoring.result_file.save(f"{scoring.pk}.joblib", ContentFile(_dump(result)), save=False)
    scoring.save()
    return scoring


# ------------------------------------------------------------------ storage


def _dump(obj) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer, compress=3)
    return buffer.getvalue()


@lru_cache(maxsize=8)
def _load(path: str, modified: int):
    return joblib.load(path)


def _load_file(field) -> dict:
    # Cached per file version. The returned objects are shared: do not modify them.
    path = field.path
    return _load(path, os.stat(path).st_mtime_ns)


def load_model(analysis: Analysis) -> dict:
    return _load_file(analysis.model_file)


def load_result(scoring: Scoring) -> dict:
    return _load_file(scoring.result_file)
