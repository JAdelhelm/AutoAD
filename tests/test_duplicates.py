import numpy as np
import pandas as pd
import pytest

from autoad import AutoAD


def orders(n_rows: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Order ID": np.arange(100, 100 + n_rows),
            "Amount": rng.normal(100, 15, n_rows).round(2),
            "Quantity": rng.integers(1, 9, n_rows),
            "Channel": rng.choice(["web", "store", "phone"], n_rows),
        }
    )


def test_rows_that_differ_only_in_their_id_are_duplicates():
    X = orders()
    X.loc[10, ["Amount", "Quantity", "Channel"]] = X.loc[3, ["Amount", "Quantity", "Channel"]].to_numpy()
    ad = AutoAD().fit(X)
    assert ad.dropped_columns_ == {"Order ID": "id"}

    result = ad.duplicates(X)
    assert list(result.columns) == [
        "Duplicate_score",
        "Closest_row",
        "Differing_Columns",
        "Duplicate_group",
        "is_duplicate",
    ]
    assert result.index.equals(X.index)
    assert result.index[result["is_duplicate"]].tolist() == [3, 10]
    assert result.loc[3, "Closest_row"] == 10
    assert result.loc[10, "Closest_row"] == 3
    assert result.loc[[3, 10], "Duplicate_score"].tolist() == [0.0, 0.0]
    assert result.loc[[3, 10], "Differing_Columns"].tolist() == ["", ""]
    assert result.loc[[3, 10], "Duplicate_group"].tolist() == [1, 1]
    assert result["Duplicate_group"].isna().sum() == len(X) - 2
    # The other rows are not duplicates, and their score is the distance to their closest row.
    assert (result.loc[~result["is_duplicate"], "Duplicate_score"] > 0).all()


def test_low_scores_mean_duplicates_but_nothing_is_flagged_without_them():
    # An inverted contamination threshold would always flag the lowest scores.
    X = orders()
    result = AutoAD().fit(X).duplicates(X)
    assert not result["is_duplicate"].any()


def test_three_identical_rows_form_one_group():
    X = orders()
    for row in (20, 40):
        X.loc[row, ["Amount", "Quantity", "Channel"]] = X.loc[5, ["Amount", "Quantity", "Channel"]].to_numpy()
    X.loc[30, ["Amount", "Quantity", "Channel"]] = X.loc[7, ["Amount", "Quantity", "Channel"]].to_numpy()
    result = AutoAD().fit(X).duplicates(X)
    groups = result.loc[result["is_duplicate"], "Duplicate_group"]
    assert groups.to_dict() == {5: 1, 7: 2, 20: 1, 30: 2, 40: 1}


def test_near_duplicates_need_a_tolerance():
    X = orders()
    X.loc[11, ["Amount", "Quantity", "Channel"]] = X.loc[2, ["Amount", "Quantity", "Channel"]].to_numpy()
    X.loc[11, "Amount"] += 0.01
    ad = AutoAD().fit(X)
    assert not ad.duplicates(X)["is_duplicate"].any()

    # Features are scaled by the IQR (about 20 for Amount), so 0.01 is a distance of about 0.0005.
    result = ad.duplicates(X, tolerance=0.002)
    assert result.index[result["is_duplicate"]].tolist() == [2, 11]
    assert result.loc[11, "Differing_Columns"] == "Amount"
    assert 0 < result.loc[11, "Duplicate_score"] <= 0.002
    # A larger tolerance also finds rows that are merely similar.
    assert ad.duplicates(X, tolerance=0.05)["is_duplicate"].sum() > 2


def test_equal_features_with_different_values_are_not_duplicates():
    # With frequency encoding, "x" and "y" (equally frequent) become the same feature.
    X = pd.DataFrame(
        {
            "value": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "city": ["x", "y", "z", "z", "x", "y"],
        }
    )
    ad = AutoAD(max_binary_categories=2).fit(X)
    assert ad.column_groups_["frequency"] == ["city"]
    result = ad.duplicates(X)
    assert result.loc[0, "Duplicate_score"] == pytest.approx(0.0, abs=1e-9)
    assert result.loc[0, "Differing_Columns"] == "city"
    assert not result["is_duplicate"].any()


def test_missing_values_are_equal_to_each_other_but_not_to_the_median():
    X = pd.DataFrame({"a": [1.0, np.nan, np.nan, 3.0, 2.0, 5.0], "b": [7, 8, 8, 9, 8, 6]})
    result = AutoAD().fit(X).duplicates(X)
    assert result.index[result["is_duplicate"]].tolist() == [1, 2]
    assert not result.loc[4, "is_duplicate"]  # a = 2 is the median that NaN is imputed with


def test_excluded_columns_are_ignored():
    X = orders()
    X["Comment"] = [f"note {i}" for i in range(len(X))]
    X.loc[10, ["Amount", "Quantity", "Channel"]] = X.loc[3, ["Amount", "Quantity", "Channel"]].to_numpy()
    assert AutoAD(exclude_columns=["Comment"]).fit(X).duplicates(X)["is_duplicate"].sum() == 2
    assert not AutoAD(max_binary_categories=100).fit(X).duplicates(X)["is_duplicate"].any()


def test_index_labels_and_single_row():
    X = orders().set_axis([f"r{i}" for i in range(60)], axis=0)
    X.loc["r9", ["Amount", "Quantity", "Channel"]] = X.loc["r1", ["Amount", "Quantity", "Channel"]].to_numpy()
    ad = AutoAD().fit(X)
    assert ad.duplicates(X).loc["r9", "Closest_row"] == "r1"

    single = ad.duplicates(X.iloc[[0]])
    assert single["is_duplicate"].tolist() == [False]
    assert single["Duplicate_group"].isna().all()


def test_negative_tolerance_is_rejected():
    X = orders()
    with pytest.raises(ValueError, match="tolerance"):
        AutoAD().fit(X).duplicates(X, tolerance=-1)
