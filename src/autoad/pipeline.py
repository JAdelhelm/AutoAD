"""Assemble the feature pipeline from the detected column groups."""

from __future__ import annotations

from category_encoders import BinaryEncoder
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from autoad.preprocessing.date_encoder import DateEncoder
from autoad.preprocessing.frequency_encoder import FrequencyEncoder
from autoad.preprocessing.missing import MissingValueMarker
from autoad.statistical.mad import MedianAbsoluteDeviation
from autoad.statistical.row_sum import RowSum
from autoad.statistical.tukey import TukeyTransformer

#: Column groups in the order their features appear in the feature matrix.
GROUPS = ("numeric", "bool", "binary", "frequency", "datetime")

SCALERS = {"robust": RobustScaler, "standard": StandardScaler, "minmax": MinMaxScaler}


def make_scaler(scaler):
    """Return a fresh scaler: "robust", "standard", "minmax", None or a transformer."""
    if scaler is None or scaler in ("none", "passthrough"):
        return "passthrough"
    if isinstance(scaler, str):
        if scaler not in SCALERS:
            raise ValueError(
                f"Unknown scaler {scaler!r}. Use one of {sorted(SCALERS)}, None or a scikit-learn transformer."
            )
        return SCALERS[scaler]()
    if hasattr(scaler, "fit") and hasattr(scaler, "transform"):
        return clone(scaler)
    raise TypeError(f"scaler must be a string, None or a scikit-learn transformer, got {scaler!r}.")


def _most_frequent():
    return SimpleImputer(strategy="most_frequent", keep_empty_features=True)


def _numeric_pipeline(tukey_factor, mad_threshold) -> Pipeline:
    # Impute once, then derive the values and both outlier counts from the same data.
    features = FeatureUnion(
        [
            ("values", "passthrough"),
            (
                "tukey",
                Pipeline(
                    [
                        ("flags", TukeyTransformer(factor=tukey_factor)),
                        ("total", RowSum(name="Tukey_Total")),
                    ]
                ),
            ),
            (
                "mad",
                Pipeline(
                    [
                        ("flags", MedianAbsoluteDeviation(threshold=mad_threshold)),
                        ("total", RowSum(name="MAD_Total")),
                    ]
                ),
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("features", features),
        ]
    )


def build_preprocessor(
    groups: dict,
    *,
    tukey_factor: float,
    mad_threshold: float,
    missing_indicator,
    scaler,
    n_jobs,
    drop_constant_features: bool = True,
) -> Pipeline:
    """Build the unfitted pipeline that turns typed columns into the scaled feature matrix.

    ``groups`` maps every name in :data:`GROUPS` to a list of columns. Missing values
    are marked on the raw columns, in parallel to (and therefore before) imputation.
    With ``drop_constant_features``, features without variance in the training data are
    removed before scaling (step ``"drop_constant"``): they carry no information for a
    detector, and some detectors (PyOD's PCA) return infinite scores for them.
    """
    transformers = []
    if groups["numeric"]:
        transformers.append(("numeric", _numeric_pipeline(tukey_factor, mad_threshold), groups["numeric"]))
    if groups["bool"]:
        transformers.append(("bool", _most_frequent(), groups["bool"]))
    if groups["binary"]:
        binary = Pipeline([("impute", _most_frequent()), ("encode", BinaryEncoder(handle_unknown="indicator"))])
        transformers.append(("binary", binary, groups["binary"]))
    if groups["frequency"]:
        frequency = Pipeline([("impute", _most_frequent()), ("encode", FrequencyEncoder())])
        transformers.append(("frequency", frequency, groups["frequency"]))
    if groups["datetime"]:
        dates = Pipeline(
            [
                ("encode", DateEncoder()),
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ]
        )
        transformers.append(("datetime", dates, groups["datetime"]))
    if missing_indicator is not None:
        columns = [col for group in GROUPS for col in groups[group]]
        transformers.append(("missing", MissingValueMarker(mode=missing_indicator), columns))

    features = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False, n_jobs=n_jobs)
    drop_constant = VarianceThreshold(threshold=0.0) if drop_constant_features else "passthrough"
    pipeline = Pipeline([("features", features), ("drop_constant", drop_constant), ("scale", make_scaler(scaler))])
    return pipeline.set_output(transform="pandas")
