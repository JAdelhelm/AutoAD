import numpy as np
import pandas as pd
import pytest

from autoad.statistical import MedianAbsoluteDeviation, RowSum, TukeyTransformer


@pytest.fixture
def two_scales():
    # One outlier in a small-valued column next to a column with large values.
    return pd.DataFrame(
        {
            "small": [1, 2, 3, 4, 5, 6, 7, 100],
            "large": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        }
    )


@pytest.mark.parametrize("method", [TukeyTransformer(), MedianAbsoluteDeviation()])
def test_statistics_are_computed_per_column(method, two_scales):
    # Regression: quantiles/medians used to be computed over all columns together, so
    # every value of the large column was flagged.
    flags = method.fit(two_scales).transform(two_scales)
    assert flags[:, 0].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert flags[:, 1].sum() == 0


def test_mad_zero_falls_back_to_mean_absolute_deviation():
    # More than half the values equal the median, so MAD = 0. Previously every value
    # that differs from the median was flagged; now only the extreme one is.
    X = pd.DataFrame({"v": [10, 10, 10, 10, 10, 11, 12, 10, 9]})
    mad = MedianAbsoluteDeviation().fit(X)
    scores = mad.modified_z_scores(X)
    assert np.all(np.isfinite(scores))
    assert mad.transform(X)[:, 0].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0]


def test_mad_constant_column_is_never_flagged():
    X = pd.DataFrame({"c": [3.0] * 6})
    mad = MedianAbsoluteDeviation().fit(X)
    assert mad.transform(X).sum() == 0
    assert np.all(mad.modified_z_scores(X) == 0)
    assert mad.transform(pd.DataFrame({"c": [3.0, 1000.0]})).sum() == 0


def test_mad_scores_output():
    X = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 100.0]})
    scores = MedianAbsoluteDeviation(scores_output=True).fit(X).transform(X)
    assert scores.dtype == float
    assert scores[-1, 0] > 3.5


def test_tukey_zero_iqr_uses_fallback_quantiles():
    # Q1 = Q3 = 0. Previously 1, 2, 3 and 50 were all flagged.
    X = pd.DataFrame({"v": [0] * 16 + [1, 2, 3, 50]})
    tukey = TukeyTransformer().fit(X)
    assert not tukey.skipped_[0]
    assert np.flatnonzero(tukey.transform(X)[:, 0]).tolist() == [19]


def test_tukey_skips_columns_without_spread():
    X = pd.DataFrame({"c": [7.0] * 10})
    tukey = TukeyTransformer().fit(X)
    assert tukey.skipped_[0]
    assert tukey.transform(pd.DataFrame({"c": [7.0, 1e9]})).sum() == 0


def test_tukey_without_fallback_skips_zero_iqr():
    X = pd.DataFrame({"v": [0] * 16 + [1, 2, 3, 50]})
    tukey = TukeyTransformer(fallback_quantiles=None).fit(X)
    assert tukey.skipped_[0]
    assert tukey.transform(X).sum() == 0


@pytest.mark.parametrize("method", [TukeyTransformer(), MedianAbsoluteDeviation()])
def test_missing_values_are_ignored_and_never_flagged(method):
    X = pd.DataFrame({"v": [1.0, 2.0, np.nan, 3.0, 4.0, 100.0]})
    flags = method.fit(X).transform(X)
    assert flags[:, 0].tolist() == [0, 0, 0, 0, 0, 1]


def test_feature_names(two_scales):
    assert TukeyTransformer().fit(two_scales).get_feature_names_out().tolist() == [
        "small_TUKEY",
        "large_TUKEY",
    ]
    assert MedianAbsoluteDeviation().fit(two_scales).get_feature_names_out().tolist() == [
        "small_MAD",
        "large_MAD",
    ]


def test_row_sum():
    flags = np.array([[1, 0, 1], [0, 0, 0]])
    total = RowSum(name="Total").fit(flags)
    assert total.transform(flags).ravel().tolist() == [2.0, 0.0]
    assert total.get_feature_names_out().tolist() == ["Total"]


def test_wrong_number_of_columns_raises(two_scales):
    tukey = TukeyTransformer().fit(two_scales)
    with pytest.raises(ValueError, match="features"):
        tukey.transform(two_scales[["small"]])
