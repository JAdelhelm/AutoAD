import warnings

import numpy as np
import pandas as pd
import pytest

from autoad.preprocessing import DateEncoder, FrequencyEncoder, MissingValueMarker
from autoad.preprocessing.type_inference import TypeInferenceTransformer


def types_of(frame, **kwargs):
    return TypeInferenceTransformer(**kwargs).fit(frame).column_types_


def test_numbers_stored_as_text_are_numeric_not_dates():
    # Regression: "2001", "2002", ... used to be parsed as dates.
    X = pd.DataFrame({"year": ["2001", "2002", "2003"], "mixed": [1, "2", 3]})
    t = TypeInferenceTransformer().fit(X)
    assert t.column_types_ == {"year": "numeric", "mixed": "numeric"}
    assert t.transform(X)["mixed"].tolist() == [1.0, 2.0, 3.0]


def test_date_strings_are_detected():
    X = pd.DataFrame({"d": ["2020-01-15", "2019-05-22", "2018-08-30", None]})
    assert types_of(X) == {"d": "datetime"}


def test_words_are_not_parsed_as_dates():
    X = pd.DataFrame({"w": ["today", "now", "March"], "t": ["Test1", "Test2", "Test3"]})
    assert types_of(X) == {"w": "categorical", "t": "categorical"}


def test_explicit_datetime_columns():
    X = pd.DataFrame({"d": ["15.01.2020", "22.05.2019", "30.08.2018"]})
    t = TypeInferenceTransformer(datetime_columns=["d"]).fit(X)
    assert t.column_types_ == {"d": "datetime"}
    out = t.transform(pd.DataFrame({"d": ["15.01.2020", "garbage"]}))
    assert out["d"].isna().tolist() == [False, True]


def test_explicit_numerical_columns():
    X = pd.DataFrame({"n": ["1", "2", "x"]})
    t = TypeInferenceTransformer(numerical_columns=["n"]).fit(X)
    assert t.column_types_ == {"n": "numeric"}
    assert t.transform(X)["n"].isna().tolist() == [False, False, True]


def test_placeholder_strings_become_missing():
    X = pd.DataFrame({"c": ["a", "", " ", "nan", "None", "b"]})
    out = TypeInferenceTransformer().fit(X).transform(X)
    assert out["c"].isna().tolist() == [False, True, True, True, True, False]


def test_bool_detection_with_missing_values():
    X = pd.DataFrame({"m": [False, True, False, ""]})
    t = TypeInferenceTransformer().fit(X)
    assert t.column_types_ == {"m": "bool"}
    assert t.transform(X)["m"].tolist()[:3] == [0.0, 1.0, 0.0]
    assert np.isnan(t.transform(X)["m"].iloc[3])


@pytest.mark.parametrize("dtype", ["category", "string", object])
def test_text_dtypes_are_categorical(dtype):
    X = pd.DataFrame({"c": pd.Series(list("aabbc"), dtype=dtype)})
    t = TypeInferenceTransformer().fit(X)
    assert t.column_types_ == {"c": "categorical"}
    assert t.transform(X)["c"].dtype == object


def test_transform_casts_to_the_types_learned_in_fit():
    train = pd.DataFrame({"n": [1.0, 2.0, 3.0], "d": pd.to_datetime(["2020-01-01"] * 3)})
    test = pd.DataFrame({"n": ["4", "abc", None], "d": ["2021-02-03", "not a date", None]})
    out = TypeInferenceTransformer().fit(train).transform(test)
    assert out["n"].isna().tolist() == [False, True, True]
    assert out["d"].isna().tolist() == [False, True, True]


def test_unknown_explicit_column_raises():
    with pytest.raises(ValueError, match="not found"):
        TypeInferenceTransformer(datetime_columns=["nope"]).fit(pd.DataFrame({"a": [1]}))


def test_importing_autoad_does_not_silence_warnings():
    # Regression: the old TypeInferenceTransformer called warnings.filterwarnings("ignore").
    import autoad  # noqa: F401

    assert not any(f[0] == "ignore" and f[1] is None and f[2] is Warning and f[3] is None for f in warnings.filters)


def test_date_encoder_adds_time_units_only_when_present():
    dates = pd.DataFrame({"d": pd.to_datetime(["2020-01-15", "2021-06-30", None])})
    stamps = pd.DataFrame({"t": pd.to_datetime(["2020-01-15 10:30", "2021-06-30 00:00"])})
    assert DateEncoder().fit(dates).get_feature_names_out().tolist() == [
        "d_YEAR",
        "d_MONTH",
        "d_DAY",
        "d_WKDAY",
    ]
    assert "t_HOUR" in DateEncoder().fit(stamps).get_feature_names_out()
    out = DateEncoder().fit(dates).transform(dates)
    assert out.loc[0, "d_YEAR"] == 2020 and np.isnan(out.loc[2, "d_YEAR"])


def test_frequency_encoder_gives_unseen_categories_zero():
    X = pd.DataFrame({"c": ["a", "a", "a", "b"]})
    out = FrequencyEncoder().fit(X).transform(pd.DataFrame({"c": ["a", "b", "new"]}))
    assert out[:, 0].tolist() == [0.75, 0.25, 0.0]


def test_missing_value_marker_watches_every_column():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    test = pd.DataFrame({"a": [np.nan, 2.0], "b": [None, "y"]})
    assert MissingValueMarker().fit(train).transform(test).ravel().tolist() == [2.0, 0.0]
    columns = MissingValueMarker(mode="columns").fit(train)
    assert columns.transform(test).tolist() == [[1.0, 1.0], [0.0, 0.0]]
    assert columns.get_feature_names_out().tolist() == ["a_MISSING", "b_MISSING"]
