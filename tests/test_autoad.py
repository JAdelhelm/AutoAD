import numpy as np
import pandas as pd
import pytest
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.base import clone

from autoad import AutoAD


def test_fit_returns_self(x_train):
    ad = AutoAD()
    assert ad.fit(x_train) is ad


def test_fit_transform_matches_fit_then_transform(x_train):
    pd.testing.assert_frame_equal(
        AutoAD().fit_transform(x_train), AutoAD().fit(x_train).transform(x_train)
    )


def test_constant_column_does_not_break_transform(x_train, x_test):
    # Regression: a dropped constant column made transform fail with
    # "Column names must be identical!".
    ad = AutoAD(IForest(random_state=0)).fit(x_train.assign(Const=1))
    assert ad.dropped_columns_["Const"] == "constant"
    out = ad.transform(x_test.assign(Const=1))
    assert list(out.columns) == list(ad.get_feature_names_out())


def test_small_but_varying_column_is_kept(x_train):
    # Regression: the old std <= 0.1 rule dropped columns such as rates.
    ad = AutoAD().fit(x_train.assign(Rate=np.linspace(0.01, 0.05, len(x_train))))
    assert "Rate" not in ad.dropped_columns_


def test_constant_text_column_is_dropped():
    X = pd.DataFrame({"v": [1.0, 5.0, 2.0, 8.0], "Rank": ["A"] * 4})
    ad = AutoAD().fit(X)
    assert ad.dropped_columns_ == {"Rank": "constant"}
    assert not any(name.startswith("Rank") for name in ad.get_feature_names_out())


def test_inputs_are_not_modified(x_train, x_test):
    train, test = x_train.copy(), x_test.copy()
    ad = AutoAD(IForest(random_state=0)).fit(x_train)
    ad.transform(x_test)
    ad.report(x_test)
    pd.testing.assert_frame_equal(x_train, train)
    pd.testing.assert_frame_equal(x_test, test)


def test_column_mismatch_names_missing_and_unexpected_columns(x_train, x_test):
    ad = AutoAD().fit(x_train)
    bad = x_test.drop(columns=["Age"]).assign(Extra=1)
    with pytest.raises(ValueError, match=r"Missing: \['Age'\]\. Unexpected: \['Extra'\]"):
        ad.transform(bad)


def test_column_order_does_not_matter(x_train, x_test):
    ad = AutoAD().fit(x_train)
    pd.testing.assert_frame_equal(ad.transform(x_test), ad.transform(x_test[x_test.columns[::-1]]))


def test_transform_uses_the_types_learned_in_fit(x_train, x_test):
    ad = AutoAD().fit(x_train)
    assert ad.column_types_["Hire Date"] == "datetime"
    names = list(ad.transform(x_test).columns)
    assert "Hire Date_YEAR" in names
    assert "Hire Date_HOUR" not in names  # pure dates: no constant time features


def test_id_column_is_dropped(x_train):
    assert AutoAD().fit(x_train).dropped_columns_ == {"ID": "id"}
    assert "ID" in AutoAD(drop_id_columns=False).fit(x_train).get_feature_names_out()


@pytest.mark.parametrize(
    "dtype, values, expected",
    [
        ("category", list("aabbccaabbcc"), "categorical"),
        ("string", list("aabbccaabbcc"), "categorical"),
        ("bool", [True, False] * 6, "bool"),
    ],
)
def test_category_string_and_bool_columns(dtype, values, expected):
    # Regression: category crashed, string was dropped silently, bool skipped the
    # missing-value marker.
    X = pd.DataFrame({"num": np.arange(12) * 1.7 % 5, "x": pd.Series(values, dtype=dtype)})
    ad = AutoAD().fit(X)
    out = ad.transform(X)
    assert ad.column_types_["x"] == expected
    assert any(name.startswith("x") for name in out.columns)
    assert not out.isna().any().any()


def test_missing_bool_value_is_counted():
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "flag": [True, False, None, True]})
    out = AutoAD(scaler=None).fit(X).transform(X)
    assert out["missing_count"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_new_missing_values_in_test_data_are_reported(x_train, x_test):
    # Age has no missing values in training, three in the test data.
    out = AutoAD().fit(x_train).transform(x_test)
    assert out["missing_count"].tolist() == [0, 0, 0, 1, 0, 0, 1, 0, 1, 0]


def test_unseen_categories_do_not_break_transform():
    train = pd.DataFrame({"Category": list("ABCABC"), "v": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0]})
    test = pd.DataFrame({"Category": ["A", "D", "E"], "v": [1.0, 2.0, 3.0]})
    out = AutoAD().fit(train).transform(test)
    assert out.shape[0] == 3
    assert not out.isna().any().any()


def test_index_is_kept(x_train):
    X = x_train.set_axis(range(100, 110), axis=0)
    assert AutoAD().fit_transform(X).index.tolist() == list(range(100, 110))


def test_report_keeps_rows_and_names_triggering_columns(x_train, x_test):
    report = AutoAD(IForest(random_state=0)).fit(x_train).report(x_test)
    assert report.index.tolist() == x_test.index.tolist()
    assert list(report.columns[: x_test.shape[1]]) == list(x_test.columns)
    for column in ["AD_score", "Tukey_Total", "MAD_Total", "Tukey_Columns", "MAD_Columns"]:
        assert column in report
    # Row 3 has a salary of 8,000,000 instead of 8,000.
    assert "Salary" in report.loc[3, "MAD_Columns"]
    assert report.loc[3, "MAD_Total"] >= 1


def test_report_without_detector_has_no_score(x_train):
    report = AutoAD().fit(x_train).report(x_train)
    assert "AD_score" not in report
    assert "Tukey_Total" in report


def test_decision_function_and_predict(x_train, x_test):
    ad = AutoAD(IForest(random_state=0)).fit(x_train)
    assert ad.decision_function(x_test).shape == (len(x_test),)
    assert set(np.unique(ad.predict(x_test))) <= {0, 1}


def test_scoring_needs_a_detector(x_train):
    ad = AutoAD().fit(x_train)
    with pytest.raises(ValueError, match="without a detector"):
        ad.decision_function(x_train)


def test_feature_importances(x_train):
    ad = AutoAD(IForest(random_state=0)).fit(x_train)
    importances = ad.feature_importances
    assert list(importances.columns) == ["Feature", "Importance"]
    assert len(importances) == len(ad.get_feature_names_out())


def test_feature_importances_need_a_supporting_detector(x_train):
    with pytest.raises(AttributeError, match="feature_importances_"):
        _ = AutoAD(LOF()).fit(x_train).feature_importances


def test_is_a_scikit_learn_estimator():
    ad = AutoAD(IForest(), scaler="standard", datetime_columns=["d"])
    copy = clone(ad)
    assert copy.get_params()["scaler"] == "standard"
    assert copy.get_params()["datetime_columns"] == ["d"]


def test_detector_passed_in_is_not_fitted(x_train):
    detector = IForest(random_state=0)
    AutoAD(detector).fit(x_train)
    assert not hasattr(detector, "decision_scores_")


def test_numpy_input():
    X = np.random.default_rng(0).normal(size=(30, 3))
    ad = AutoAD(IForest(random_state=0)).fit(X)
    out = ad.transform(X)
    assert out.shape[0] == 30
    assert "x0" in out.columns
    assert "x0" in ad.report(X)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no rows|no columns"):
        AutoAD().fit(pd.DataFrame())


def test_all_columns_dropped_raises():
    with pytest.raises(ValueError, match="No usable columns"):
        AutoAD().fit(pd.DataFrame({"c": [1, 1, 1]}))


def test_single_row_without_dropping():
    X = pd.DataFrame({"Name": ["Alice"], "Age": [25], "Sex": ["female"]})
    out = AutoAD(drop_constant=False).fit_transform(X)
    assert len(out) == 1


def test_features_are_robust_scaled_by_default(x_train):
    out = AutoAD().fit_transform(x_train)
    assert out["Salary"].median() == pytest.approx(0)
    raw = AutoAD(scaler=None).fit_transform(x_train)
    np.testing.assert_allclose(raw["Salary"], x_train["Salary"])


def test_unknown_scaler_raises(x_train):
    with pytest.raises(ValueError, match="Unknown scaler"):
        AutoAD(scaler="fancy").fit(x_train)


def test_missing_indicator_modes(x_train, x_test):
    columns = AutoAD(missing_indicator="columns").fit(x_train).transform(x_test)
    assert columns["Age_MISSING"].sum() == 3
    none = AutoAD(missing_indicator=None).fit(x_train).transform(x_test)
    assert not any("missing" in name.lower() for name in none.columns)


def test_exclude_columns(x_train, x_test):
    ad = AutoAD(exclude_columns=["Name"]).fit(x_train)
    assert not any(name.startswith("Name") for name in ad.get_feature_names_out())
    pd.testing.assert_frame_equal(ad.transform(x_test), ad.transform(x_test.drop(columns="Name")))


def test_high_cardinality_columns_are_frequency_encoded():
    X = pd.DataFrame({"city": [f"c{i % 30}" for i in range(90)], "v": np.arange(90) % 7})
    names = AutoAD().fit(X).get_feature_names_out().tolist()
    assert "city_FREQ" in names
