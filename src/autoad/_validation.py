"""Small input helpers shared by the transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def remember_input(estimator, X) -> None:
    """Store ``n_features_in_`` and, for DataFrames, ``feature_names_in_`` on ``estimator``."""
    estimator.n_features_in_ = X.shape[1]
    if hasattr(X, "columns"):
        estimator.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
    elif hasattr(estimator, "feature_names_in_"):
        del estimator.feature_names_in_


def check_n_features(estimator, X) -> None:
    """Raise if ``X`` has a different number of columns than seen during fit."""
    if X.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(estimator).__name__} "
            f"was fitted with {estimator.n_features_in_} features."
        )


def to_float_array(X) -> np.ndarray:
    """Return ``X`` as a 2-D float array with NaN for missing values."""
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(dtype=float, na_value=np.nan)
    else:
        values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def as_frame(X) -> pd.DataFrame:
    """Return ``X`` as a DataFrame without copying an existing one."""
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
