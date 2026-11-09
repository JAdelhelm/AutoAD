"""Checks that AutoAD actually finds anomalies, not only that the columns look right."""

import numpy as np
import pandas as pd
import pytest
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from autoad import AutoAD

ANOMALIES = [10, 50, 150, 250]


@pytest.fixture
def employees_with_anomalies():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "age": rng.normal(40, 8, n).round(),
            "salary": rng.normal(60_000, 8_000, n).round(2),
            "tenure": rng.integers(0, 20, n),
            "department": rng.choice(["HR", "Engineering", "Finance", "Sales"], n),
            "hired": pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 3000, n), unit="D"),
        }
    )
    X.loc[10, "salary"] = 900_000
    X.loc[50, "age"] = 140
    X.loc[150, "salary"] = -50_000
    X.loc[250, ["age", "tenure"]] = [19, 45]
    return X


@pytest.mark.parametrize(
    "detector",
    [IForest(random_state=0, n_estimators=200), LOF(n_neighbors=20), KNN()],
    ids=["IForest", "LOF", "KNN"],
)
def test_injected_anomalies_get_the_highest_scores(employees_with_anomalies, detector):
    scores = AutoAD(detector).fit(employees_with_anomalies).decision_function(employees_with_anomalies)
    # Top 2 % of 300 rows. The random data has one natural outlier as well (row 178,
    # salary almost 4 standard deviations below the mean), so allow a little room.
    top = set(np.argsort(scores)[::-1][:6])
    assert set(ANOMALIES) <= top


def test_injected_anomalies_are_flagged_by_univariate_methods(employees_with_anomalies):
    report = AutoAD().fit(employees_with_anomalies).report(employees_with_anomalies)
    flagged = report.index[report["MAD_Total"] > 0]
    assert set(ANOMALIES) <= set(flagged)


def test_example_data_ranks_the_salary_outliers_first(x_train, x_test):
    # Regression for the example in examples/main.py: previously LOF gave all eight
    # normal rows exactly the same score, because features were neither scaled nor
    # were Tukey/MAD computed per column.
    scores = AutoAD(LOF()).fit(x_train).decision_function(x_test)
    top_two = set(np.argsort(scores)[::-1][:2])
    assert top_two == {3, 9}
    normal = np.delete(scores, [3, 9])
    assert len(np.unique(normal.round(6))) > 1
