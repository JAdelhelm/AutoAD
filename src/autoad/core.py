"""The AutoAD estimator."""

from __future__ import annotations

import copy
import logging
import re

import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from autoad.ensemble import ScoreEnsemble
from autoad.pipeline import GROUPS, build_preprocessor
from autoad.preprocessing.type_inference import (
    CATEGORICAL,
    NUMERIC,
    TypeInferenceTransformer,
)

logger = logging.getLogger("autoad")

# "id", "ID", "customer_id", "id_customer", "Customer ID" - but not "paid" or "uuid".
_ID_NAME = re.compile(r"(?:^|[^a-z0-9])id$|^id(?:[^a-z0-9]|$)")
# Unnamed integer columns count as IDs only from this many rows on (1, 2, 3, ... in a
# tiny table is more likely a feature than an index).
_MIN_ROWS_FOR_UNNAMED_ID = 20


class AutoAD(TransformerMixin, BaseEstimator):
    """Automatic preprocessing and anomaly detection for tabular data.

    AutoAD learns a type for every column of a DataFrame, turns the table into a
    scaled numeric feature matrix and, if a detector is given, fits it on that matrix.

    Features per column type:

    * numeric: the value (missing values imputed with the median) plus, per row,
      ``Tukey_Total`` and ``MAD_Total``: how many of the row's numeric values lie
      outside Tukey's fences or have a modified z-score above ``mad_threshold``.
    * bool: 0/1 (missing values imputed with the most frequent value).
    * categorical: a binary code (``BinaryEncoder``) for columns with at most
      ``max_binary_categories`` distinct values, otherwise the relative frequency of
      the category in the training data (unseen categories get 0).
    * datetime: year, month, day and weekday, plus hour, minute and second if the
      column contains a time of day.
    * missing values: ``missing_count`` per row, computed before imputation.

    Features that are constant in the training data are removed (see ``drop_constant``),
    and the remaining matrix is scaled (``RobustScaler`` by default) before it reaches
    the detector. Missing values and categories that only appear in new data are listed
    by ``report`` in ``Missing_Columns`` and ``Unseen_Columns``.

    Parameters
    ----------
    detector : PyOD detector or "ensemble", optional
        Unfitted detector, e.g. ``pyod.models.iforest.IForest()``. It is cloned
        during ``fit``. ``"ensemble"`` is short for
        ``autoad.ScoreEnsemble(random_state=0)``, the mean score of Isolation Forest,
        ECOD and KNN. Without a detector AutoAD only builds the feature matrix.
    datetime_columns : list of str, optional
        Columns to parse as dates. Other text columns are parsed as dates only if
        at least 95 % of their values are recognised as dates.
    numerical_columns : list of str, optional
        Columns to treat as numbers even if they are stored as text.
    exclude_columns : list of str, optional
        Columns to ignore completely. They may be missing in later inputs.
    drop_constant : bool, default=True
        Drop columns with only one distinct value (missing counts as a value), and
        remove features that are constant in the training data, for example
        ``missing_count`` when the training data has no missing values, or
        ``MAD_Total`` when no training value is flagged. Such features carry no
        information for the detector, and some detectors (PyOD's PCA) return infinite
        scores for them.
    drop_id_columns : bool, default=True
        Drop columns that look like identifiers: unique, complete values and a name
        such as ``id`` or ``customer_id`` (integer or text columns), or an unnamed
        integer column with at least 20 rows and consecutive values (1, 2, 3, ...).
    max_binary_categories : int, default=20
        Categorical columns with more distinct values are frequency-encoded.
    tukey_factor : float, default=1.5
        IQR multiple for Tukey's fences.
    mad_threshold : float, default=3.5
        Threshold for the modified z-score.
    missing_indicator : {"count", "columns"} or None, default="count"
        ``"count"`` adds ``missing_count``, ``"columns"`` adds one ``<name>_MISSING``
        column per input column, ``None`` adds nothing.
    scaler : {"robust", "standard", "minmax"}, None or transformer, default="robust"
        Scaling of the feature matrix. ``None`` keeps the raw values.
    n_jobs : int, default=1
        Parallel jobs for the column transformer.

    Attributes
    ----------
    column_types_ : dict
        Type learned for every used column: numeric, bool, datetime or categorical.
    dropped_columns_ : dict
        Dropped columns and the reason ("constant" or "id").
    dropped_features_ : list of str
        Features removed because they are constant in the training data.
    known_categories_ : dict
        Categories seen during ``fit`` per categorical column.
    column_groups_ : dict
        Columns per feature group.
    preprocessor_ : sklearn.pipeline.Pipeline
        Fitted feature pipeline.
    detector_ : PyOD detector or None
        Fitted clone of ``detector``.
    feature_names_in_ : ndarray of str
        Columns seen during ``fit``.
    excluded_columns_ : list of str
        Columns ignored because of ``exclude_columns``.
    feature_names_out_ : ndarray of str
        Names of the features returned by ``transform``, see ``get_feature_names_out``.
    type_inference_ : TypeInferenceTransformer
        Fitted transformer that casts every column to its learned type.
    """

    def __init__(
        self,
        detector=None,
        *,
        datetime_columns=None,
        numerical_columns=None,
        exclude_columns=None,
        drop_constant=True,
        drop_id_columns=True,
        max_binary_categories=20,
        tukey_factor=1.5,
        mad_threshold=3.5,
        missing_indicator="count",
        scaler="robust",
        n_jobs=1,
    ):
        self.detector = detector
        self.datetime_columns = datetime_columns
        self.numerical_columns = numerical_columns
        self.exclude_columns = exclude_columns
        self.drop_constant = drop_constant
        self.drop_id_columns = drop_id_columns
        self.max_binary_categories = max_binary_categories
        self.tukey_factor = tukey_factor
        self.mad_threshold = mad_threshold
        self.missing_indicator = missing_indicator
        self.scaler = scaler
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ fitting

    def fit(self, X, y=None):
        """Learn column types, the feature pipeline and the detector. ``y`` is ignored."""
        frame = self._validate_input(X, reset=True)
        excluded = list(self.exclude_columns or [])
        unknown = [c for c in excluded if c not in frame.columns]
        if unknown:
            raise ValueError(f"exclude_columns not found in X: {unknown}")
        self.excluded_columns_ = excluded
        frame = frame.drop(columns=excluded)
        if frame.shape[1] == 0:
            raise ValueError("No columns left after exclude_columns.")

        self.type_inference_ = TypeInferenceTransformer(
            datetime_columns=self.datetime_columns, numerical_columns=self.numerical_columns
        ).fit(frame)
        typed = self.type_inference_.transform(frame)
        self.column_types_ = dict(self.type_inference_.column_types_)

        self.dropped_columns_ = self._find_columns_to_drop(typed)
        typed = typed.drop(columns=list(self.dropped_columns_))
        if typed.shape[1] == 0:
            raise ValueError(
                f"No usable columns left, all were dropped: {self.dropped_columns_}. "
                "Set drop_constant=False or drop_id_columns=False to keep them."
            )

        self.column_groups_ = self._group_columns(typed)
        self.preprocessor_ = build_preprocessor(
            self.column_groups_,
            tukey_factor=self.tukey_factor,
            mad_threshold=self.mad_threshold,
            missing_indicator=self.missing_indicator,
            scaler=self.scaler,
            n_jobs=self.n_jobs,
            drop_constant_features=self.drop_constant,
        )
        try:
            features = self.preprocessor_.fit_transform(typed)
        except ValueError as error:
            if "variance threshold" not in str(error):
                raise
            raise ValueError(
                "All features are constant in the training data, so there is nothing to "
                "learn from. Pass drop_constant=False to keep them."
            ) from error
        self.feature_names_out_ = np.asarray(features.columns, dtype=object)
        self.dropped_features_ = self._constant_feature_names()
        self.known_categories_ = {
            col: set(typed[col].dropna()) for col in self.column_groups_["binary"] + self.column_groups_["frequency"]
        }

        self.detector_ = None
        if self.detector is not None:
            self.detector_ = _new_detector(self.detector).fit(features.to_numpy())

        logger.info("AutoAD column types: %s", self.column_types_)
        if self.dropped_columns_:
            logger.info("AutoAD dropped columns: %s", self.dropped_columns_)
        if self.dropped_features_:
            logger.info("AutoAD dropped constant features: %s", self.dropped_features_)
        return self

    # ---------------------------------------------------------------- inference

    def transform(self, X):
        """Return the scaled feature matrix that the detector sees."""
        return self.preprocessor_.transform(self._prepare(X))

    def decision_function(self, X):
        """Anomaly score per row from the detector; higher means more anomalous."""
        return self._fitted_detector().decision_function(self.transform(X).to_numpy())

    def predict(self, X):
        """0/1 label per row from the detector (1 = anomaly), using its contamination."""
        return self._fitted_detector().predict(self.transform(X).to_numpy())

    def report(self, X) -> pd.DataFrame:
        """Return a copy of ``X`` with anomaly information, rows in their original order.

        Added columns:

        * ``AD_score``: detector score, higher means more anomalous (only with a detector)
        * ``Tukey_Total``, ``MAD_Total``: number of the row's numeric values flagged by
          Tukey's fences or by the modified z-score
        * ``Tukey_Columns``, ``MAD_Columns``: names of these columns, comma separated
        * ``Missing_Columns``: columns with a missing value in the row, including values
          that could not be read as the column's type
        * ``Unseen_Columns``: categorical columns whose value did not occur during ``fit``
        """
        typed = self._prepare(X)
        if isinstance(X, pd.DataFrame):
            result = X.copy()
        else:
            result = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        if self.detector_ is not None:
            features = self.preprocessor_.transform(typed)
            result["AD_score"] = self.detector_.decision_function(features.to_numpy())
        findings = self._findings(typed)
        for method in ("Tukey", "MAD"):
            result[f"{method}_Total"] = findings[method].sum(axis=1).to_numpy()
            result[f"{method}_Columns"] = _flagged_column_names(findings[method])
        result["Missing_Columns"] = _flagged_column_names(findings["Missing"])
        result["Unseen_Columns"] = _flagged_column_names(findings["Unseen"])
        return result

    def findings(self, X) -> dict:
        """Return the findings behind ``report`` per cell, as boolean DataFrames.

        Every DataFrame has the index of ``X`` and one column per input column the
        finding applies to. Columns dropped or excluded during ``fit`` are left out.

        * ``"Tukey"``, ``"MAD"``: numeric columns, True where the value lies outside
          Tukey's fences or has a modified z-score above ``mad_threshold``
        * ``"Missing"``: all used columns, True where the value is missing or could not
          be read as the column's type
        * ``"Unseen"``: categorical columns, True where the category did not occur
          during ``fit``
        """
        return self._findings(self._prepare(X))

    def duplicates(self, X, *, tolerance: float = 0.0) -> pd.DataFrame:
        """Find rows of ``X`` that duplicate another row of ``X``.

        The anomaly logic turned around: a KNN detector with 2 neighbours is fitted on
        the feature matrix of ``X`` and scores the same rows. Every row is its own first
        neighbour, so its score is the distance to the closest *other* row, and a low
        score marks a duplicate. Columns AutoAD does not use (ID-like, constant or in
        ``exclude_columns``) are ignored, so two orders that differ only in their order
        ID are duplicates. The detector of this estimator is not involved.

        Parameters
        ----------
        tolerance : float, default=0.0
            With 0, rows count as duplicates only if their values are identical in every
            used column. The values are compared as well as the features, because
            different values can share a feature (two categories with the same
            frequency, or a missing value and the median it is imputed with). A larger
            value also counts rows within this distance in the scaled feature space,
            near duplicates.

        Returns
        -------
        DataFrame with the index of ``X`` and the columns

        * ``Duplicate_score``: KNN score, the distance to the closest other row; 0 for
          rows identical to another row
        * ``Closest_row``: index label of that row
        * ``Differing_Columns``: used columns in which the two rows differ, comma
          separated; empty for identical rows
        * ``Duplicate_group``: number of the group of rows that duplicate each other,
          counted in the order of their first row; ``<NA>`` for other rows
        * ``is_duplicate``: True for every row in a group
        """
        if tolerance < 0:
            raise ValueError(f"tolerance must be at least 0, got {tolerance}.")
        typed = self._prepare(X)
        n_rows = len(typed)
        result = pd.DataFrame(index=typed.index)
        if n_rows < 2:
            result["Duplicate_score"] = np.nan
            result["Closest_row"] = pd.Series([pd.NA] * n_rows, index=typed.index, dtype=object)
            result["Differing_Columns"] = ""
            result["Duplicate_group"] = pd.array([pd.NA] * n_rows, dtype="Int64")
            result["is_duplicate"] = False
            return result

        features = self.preprocessor_.transform(typed).to_numpy()
        knn = KNN(n_neighbors=2).fit(features)
        scores = knn.decision_function(features)
        _, neighbours = knn.neigh_.kneighbors(features, n_neighbors=2)
        positions = np.arange(n_rows)
        # Usually the row itself comes first; with three or more identical rows it may
        # be left out, then both neighbours are other rows.
        closest = np.where(neighbours[:, 0] != positions, neighbours[:, 0], neighbours[:, 1])

        values = typed.to_numpy(dtype=object)
        partner = values[closest]
        same = (values == partner) | (pd.isna(values) & pd.isna(partner))
        differs = pd.DataFrame(~same, index=typed.index, columns=typed.columns)
        identical = ~differs.to_numpy().any(axis=1)
        pairs = identical | ((tolerance > 0) & (scores <= tolerance))

        # Rows linked by a pair form a group (a duplicate of a duplicate is one group).
        linked = np.flatnonzero(pairs)
        components = _connected_components(n_rows, linked, closest[linked])
        in_group = np.bincount(components)[components] > 1
        numbers = {}
        groups = [numbers.setdefault(components[i], len(numbers) + 1) if in_group[i] else pd.NA for i in positions]

        result["Duplicate_score"] = np.where(identical, 0.0, scores)
        result["Closest_row"] = typed.index[closest]
        result["Differing_Columns"] = _flagged_column_names(differs)
        result["Duplicate_group"] = pd.array(groups, dtype="Int64")
        result["is_duplicate"] = in_group
        return result

    @property
    def feature_importances(self) -> pd.DataFrame:
        """Importance per feature, for detectors with ``feature_importances_`` (e.g. IForest)."""
        check_is_fitted(self, "preprocessor_")
        importances = None
        if self.detector_ is not None:
            importances = getattr(self.detector_, "feature_importances_", None)
        if importances is None:
            raise AttributeError(
                "feature_importances needs a detector with a feature_importances_ "
                "attribute, for example pyod.models.iforest.IForest."
            )
        return pd.DataFrame({"Feature": self.feature_names_out_, "Importance": np.asarray(importances)}).sort_values(
            "Importance", ascending=False, ignore_index=True
        )

    def get_feature_names_out(self, input_features=None):
        """Names of the columns returned by ``transform``."""
        check_is_fitted(self, "preprocessor_")
        return self.feature_names_out_.copy()

    # ------------------------------------------------------------------ helpers

    def _validate_input(self, X, *, reset: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame, generated = X, False
        else:
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a DataFrame or a 2-dimensional array.")
            frame = pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])])
            generated = True
        if frame.shape[0] == 0:
            raise ValueError("X has no rows.")
        if frame.shape[1] == 0:
            raise ValueError("X has no columns.")
        columns = [str(c) for c in frame.columns]
        if len(set(columns)) != len(columns):
            raise ValueError("Column names must be unique.")
        if columns != list(frame.columns):
            frame = frame.set_axis(columns, axis=1)

        if reset:
            self.feature_names_in_ = np.asarray(columns, dtype=object)
            self.n_features_in_ = len(columns)
            self.generated_feature_names_ = generated
            return frame

        if generated and not self.generated_feature_names_:
            raise TypeError("AutoAD was fitted on a DataFrame; pass a DataFrame with the same columns.")
        known = set(self.feature_names_in_)
        excluded = set(self.excluded_columns_)
        used = [c for c in self.feature_names_in_ if c not in excluded]
        present = set(columns)
        missing = [c for c in used if c not in present]
        unexpected = [c for c in columns if c not in known]
        if missing or unexpected:
            raise ValueError(
                f"X does not have the columns seen during fit. Missing: {missing}. Unexpected: {unexpected}."
            )
        return frame[used]

    def _prepare(self, X) -> pd.DataFrame:
        check_is_fitted(self, "preprocessor_")
        frame = self._validate_input(X, reset=False)
        typed = self.type_inference_.transform(frame)
        return typed.drop(columns=list(self.dropped_columns_))

    def _fitted_detector(self):
        check_is_fitted(self, "preprocessor_")
        if self.detector_ is None:
            raise ValueError("AutoAD was fitted without a detector. Pass one, e.g. AutoAD(detector=IForest()).")
        return self.detector_

    def _constant_feature_names(self) -> list:
        selector = self.preprocessor_.named_steps["drop_constant"]
        if selector == "passthrough":
            return []
        names = self.preprocessor_.named_steps["features"].get_feature_names_out()
        return [str(name) for name in names[~selector.get_support()]]

    def _find_columns_to_drop(self, typed: pd.DataFrame) -> dict:
        dropped = {}
        for col in typed.columns:
            series = typed[col]
            if self.drop_constant and series.nunique(dropna=False) <= 1:
                dropped[col] = "constant"
            elif self.drop_id_columns and _looks_like_id(col, series, self.column_types_[col]):
                dropped[col] = "id"
        return dropped

    def _group_columns(self, typed: pd.DataFrame) -> dict:
        groups = {name: [] for name in GROUPS}
        for col in typed.columns:
            kind = self.column_types_[col]
            if kind == CATEGORICAL:
                many = typed[col].nunique() > self.max_binary_categories
                groups["frequency" if many else "binary"].append(col)
            else:
                groups[kind].append(col)
        return groups

    def _findings(self, typed: pd.DataFrame) -> dict:
        findings = {method: flags.astype(bool) for method, flags in self._univariate_flags(typed).items()}
        findings["Missing"] = typed.isna()
        findings["Unseen"] = pd.DataFrame(
            {col: typed[col].notna() & ~typed[col].isin(known) for col, known in self.known_categories_.items()},
            index=typed.index,
            dtype=bool,
        )
        return findings

    def _univariate_flags(self, typed: pd.DataFrame) -> dict:
        """Per-column 0/1 flags of both univariate methods."""
        numeric = self.column_groups_["numeric"]
        if not numeric:
            empty = pd.DataFrame(index=typed.index)
            return {"Tukey": empty, "MAD": empty}
        pipeline = self.preprocessor_.named_steps["features"].named_transformers_["numeric"]
        imputed = pipeline.named_steps["impute"].transform(typed[numeric])
        union = pipeline.named_steps["features"]
        flags = {}
        for method, key in (("Tukey", "tukey"), ("MAD", "mad")):
            detector = union.named_transformers[key].named_steps["flags"]
            values = np.asarray(detector.transform(imputed))
            flags[method] = pd.DataFrame(values, index=typed.index, columns=numeric)
        return flags


def _new_detector(detector):
    if isinstance(detector, str):
        if detector != "ensemble":
            raise ValueError(f'Unknown detector {detector!r}. Pass a PyOD detector or "ensemble".')
        return ScoreEnsemble(random_state=0)
    return _clone_detector(detector)


def _clone_detector(detector):
    try:
        return clone(detector)
    except (TypeError, RuntimeError, ValueError):
        return copy.deepcopy(detector)


def _looks_like_id(name, series: pd.Series, kind: str) -> bool:
    if len(series) < 2 or series.isna().any() or not series.is_unique:
        return False
    name_hint = bool(_ID_NAME.search(str(name).lower()))
    if kind == CATEGORICAL:
        return name_hint
    if kind != NUMERIC:
        return False
    values = series.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        return False
    if name_hint:
        return True
    if len(values) < _MIN_ROWS_FOR_UNNAMED_ID:
        return False
    steps = np.diff(values)
    return bool(np.all(steps == 1) or np.all(steps == -1))


def _connected_components(n_nodes: int, sources, targets) -> np.ndarray:
    """Component label per node of an undirected graph (union-find)."""
    parent = np.arange(n_nodes)

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for source, target in zip(sources, targets, strict=True):
        a, b = root(source), root(target)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return np.array([root(node) for node in range(n_nodes)])


def _flagged_column_names(flags: pd.DataFrame) -> list:
    names = np.asarray(flags.columns, dtype=object)
    return [", ".join(names[row.astype(bool)]) for row in flags.to_numpy()]
