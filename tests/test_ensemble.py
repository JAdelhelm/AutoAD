"""ScoreEnsemble: the mean of the standardised scores of several detectors."""

import numpy as np
import pytest
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from sklearn.base import clone

from autoad import AutoAD, ScoreEnsemble


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    X[:3] += 8  # three obvious anomalies
    return X


def test_score_is_the_mean_of_the_standardised_detector_scores(data):
    ensemble = ScoreEnsemble([ECOD(), KNN()]).fit(data)
    new = data[:20] + 0.5
    expected = np.mean(
        [
            (d.decision_function(new) - d.decision_scores_.mean()) / d.decision_scores_.std()
            for d in ensemble.detectors_
        ],
        axis=0,
    )
    assert np.allclose(ensemble.decision_function(new), expected)
    # Standardised training scores have mean 0 for every detector, and so has their
    # mean: no detector dominates because its scores happen to be larger.
    assert abs(ensemble.decision_scores_.mean()) < 1e-9


def test_default_detectors_and_labels(data):
    ensemble = ScoreEnsemble(contamination=0.05, random_state=0).fit(data)
    assert [type(d).__name__ for d in ensemble.detectors_] == ["IForest", "ECOD", "KNN"]
    assert ensemble.labels_.sum() == pytest.approx(10, abs=1)  # 5 % of 200 rows
    assert set(np.argsort(ensemble.decision_scores_)[::-1][:3]) == {0, 1, 2}
    assert np.array_equal(ensemble.predict(data), (ensemble.decision_function(data) > ensemble.threshold_).astype(int))


def test_is_reproducible_and_clonable(data):
    first = ScoreEnsemble(random_state=1).fit(data)
    second = clone(first).fit(data)
    assert np.allclose(first.decision_scores_, second.decision_scores_)
    given = [IForest(random_state=0)]
    ScoreEnsemble(given).fit(data)
    assert not hasattr(given[0], "decision_scores_")  # the caller's detectors stay unfitted


def test_small_training_data_limits_the_neighbours():
    X = np.random.default_rng(0).normal(size=(4, 2))
    ensemble = ScoreEnsemble(random_state=0).fit(X)
    assert ensemble.detectors_[2].n_neighbors == 3
    assert np.isfinite(ensemble.decision_function(X)).all()


def test_invalid_detectors_raise(data):
    with pytest.raises(ValueError, match="at least one"):
        ScoreEnsemble([]).fit(data)
    constant = np.hstack([data, np.ones((len(data), 1))])  # PyOD's PCA returns inf for a constant feature
    with pytest.raises(ValueError, match="PCA returned scores that are not finite"):
        ScoreEnsemble([PCA()]).fit(constant)


def test_autoad_accepts_the_ensemble_by_name(x_train, x_test):
    ad = AutoAD(detector="ensemble").fit(x_train)
    assert isinstance(ad.detector_, ScoreEnsemble)
    scores = ad.decision_function(x_test)
    assert set(np.argsort(scores)[::-1][:2]) == {3, 9}  # the two wrong salaries
    assert np.allclose(scores, AutoAD(detector="ensemble").fit(x_train).decision_function(x_test))
    with pytest.raises(AttributeError):
        ad.feature_importances  # noqa: B018
    with pytest.raises(ValueError, match="Unknown detector"):
        AutoAD(detector="forest").fit(x_train)
