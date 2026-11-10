"""An ensemble that averages the scores of several detectors."""

from __future__ import annotations

import numpy as np
from pyod.models.base import BaseDetector
from sklearn.base import clone
from sklearn.utils.validation import check_array, check_is_fitted


class ScoreEnsemble(BaseDetector):
    """Average the standardised anomaly scores of several PyOD detectors.

    Without labels there is no way to tell which detector suits a table, and every
    detector fails on some data. Averaging evens this out: on the 47 datasets of ADBench
    the default ensemble scored slightly higher than its best member on average and
    stayed closer to the best single detector of each dataset.

    The scores of the detectors have different scales. Each detector's scores are
    therefore standardised with the mean and standard deviation of its scores on the
    training data before they are averaged.

    Parameters
    ----------
    detectors : list of PyOD detectors, optional
        Unfitted detectors; they are cloned during ``fit``. By default Isolation Forest,
        ECOD and KNN, three detectors that work in different ways (random splits, the
        tails of each feature, distances between rows). KNN uses 5 neighbours, or fewer
        if the training data has at most 5 rows.
    contamination : float, default=0.1
        Share of the training rows assumed to be anomalies; sets ``threshold_`` and so
        the labels of ``predict``. The ``contamination`` of the single detectors has no
        effect.
    random_state : int, optional
        Seed for the default Isolation Forest. Ignored if ``detectors`` is given.

    Attributes
    ----------
    detectors_ : list
        The fitted detectors.
    means_, scales_ : ndarray of shape (n_detectors,)
        Mean and standard deviation of each detector's training scores.
    decision_scores_ : ndarray of shape (n_samples,)
        Ensemble scores of the training data; higher means more anomalous.
    threshold_ : float
        Scores above it are labelled as anomalies.
    labels_ : ndarray of shape (n_samples,)
        0/1 labels of the training data.
    """

    def __init__(self, detectors=None, contamination=0.1, random_state=None):
        super().__init__(contamination=contamination)
        self.detectors = detectors
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit every detector on ``X``. ``y`` is ignored."""
        X = check_array(X)
        self._set_n_classes(y)
        if self.detectors is None:
            detectors = self._default_detectors(X.shape[0])
        else:
            detectors = [clone(detector) for detector in self.detectors]
        if not detectors:
            raise ValueError("detectors must contain at least one detector.")

        means, scales, standardised = [], [], []
        for detector in detectors:
            scores = np.asarray(detector.fit(X).decision_scores_, dtype=float)
            if not np.all(np.isfinite(scores)):
                raise ValueError(
                    f"{type(detector).__name__} returned scores that are not finite; leave it out of the ensemble."
                )
            scale = float(scores.std())
            means.append(float(scores.mean()))
            scales.append(scale if scale > 0 else 1.0)
            standardised.append((scores - means[-1]) / scales[-1])
        self.detectors_ = detectors
        self.means_ = np.asarray(means)
        self.scales_ = np.asarray(scales)
        self.decision_scores_ = np.mean(standardised, axis=0)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Mean standardised score per row; higher means more anomalous."""
        check_is_fitted(self, "detectors_")
        X = check_array(X)
        standardised = [
            (np.asarray(detector.decision_function(X), dtype=float) - mean) / scale
            for detector, mean, scale in zip(self.detectors_, self.means_, self.scales_, strict=True)
        ]
        return np.mean(standardised, axis=0)

    def _default_detectors(self, n_samples: int) -> list:
        from pyod.models.ecod import ECOD
        from pyod.models.iforest import IForest
        from pyod.models.knn import KNN

        return [
            IForest(random_state=self.random_state),
            ECOD(),
            KNN(n_neighbors=max(1, min(5, n_samples - 1))),
        ]
