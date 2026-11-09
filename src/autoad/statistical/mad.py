"""Modified z-score based on the median absolute deviation, column by column."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from autoad._validation import check_n_features, remember_input, to_float_array

# M = 0.6745 * (x - median) / MAD  (Iglewicz & Hoaglin)
_MAD_CONSTANT = 0.6745
# Fallback when MAD = 0: M = (x - median) / (1.253314 * MeanAD)
_MEAN_AD_CONSTANT = 1.253314


class MedianAbsoluteDeviation(TransformerMixin, BaseEstimator):
    """Flag values by their modified z-score, separately for every column.

    The modified z-score is ``M = 0.6745 * (x - median) / MAD``, where MAD is the
    median absolute deviation from the median. Values with ``|M| > threshold`` are
    flagged with 1. Median and MAD are learned per column during ``fit``.

    When the MAD of a column is 0 (more than half of its values are equal), the mean
    absolute deviation is used instead: ``M = (x - median) / (1.253314 * MeanAD)``.
    If that is 0 as well (a constant column), ``M`` is 0 and nothing is flagged.
    Missing values are ignored and never flagged.

    Parameters
    ----------
    threshold : float, default=3.5
        Values with ``|M|`` above this are flagged.
    scores_output : bool, default=False
        Return the scores ``M`` instead of 0/1 flags.

    Attributes
    ----------
    median_ : ndarray of shape (n_features_in_,)
    scale_ : ndarray of shape (n_features_in_,)
        ``M = (x - median_) / scale_``. NaN for constant columns.
    """

    def __init__(self, threshold=3.5, scores_output=False):
        self.threshold = threshold
        self.scores_output = scores_output

    def fit(self, X, y=None):
        values = to_float_array(X)
        remember_input(self, X)
        with warnings.catch_warnings():
            # All-NaN columns give NaN statistics; they are treated as constant.
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(values, axis=0)
            deviation = np.abs(values - median)
            mad = np.nanmedian(deviation, axis=0)
            mean_ad = np.nanmean(deviation, axis=0)
        scale = np.full(values.shape[1], np.nan)
        use_mad = mad > 0
        use_mean_ad = ~use_mad & (mean_ad > 0)
        scale[use_mad] = mad[use_mad] / _MAD_CONSTANT
        scale[use_mean_ad] = _MEAN_AD_CONSTANT * mean_ad[use_mean_ad]
        self.median_ = median
        self.scale_ = scale
        return self

    def modified_z_scores(self, X) -> np.ndarray:
        """Return the modified z-score of every value (NaN for missing values)."""
        check_is_fitted(self, "scale_")
        values = to_float_array(X)
        check_n_features(self, values)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = (values - self.median_) / self.scale_
        constant = np.isnan(self.scale_)
        scores[:, constant] = np.where(np.isnan(values[:, constant]), np.nan, 0.0)
        return scores

    def transform(self, X):
        scores = self.modified_z_scores(X)
        if self.scores_output:
            return scores
        with np.errstate(invalid="ignore"):
            return (np.abs(scores) > self.threshold).astype(np.int64)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "scale_")
        names = _check_feature_names_in(self, input_features)
        return np.asarray([f"{name}_MAD" for name in names], dtype=object)
