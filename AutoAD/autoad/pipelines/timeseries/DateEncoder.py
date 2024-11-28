from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np

from sklearn import set_config

set_config(transform_output="pandas")


class DateEncoder(TransformerMixin):
    """
    The DateEncoder class is used to extract different time units.

    Steps - DateEncoder:
        (1) Iterate over each column in X (DataFrames).\n
        (2) Extraction of various time units.\n
        (3) Creation of a new DataFrame.\n
        (4) Reassignment of the new column labels.\n
        (5) Return of the transformed DataFrame.\n
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dfs = []
        self.column_names = []

        for column in X:
            # Explicitly cast it again to datetime
            X[column] = pd.to_datetime(X[column], errors="coerce")

            dt = X[column].dt

            new_column_names = [
                column + "_" + col
                for col in [
                    "YEAR",
                    "MONTH",
                    "DAY",
                    "HOUR",
                    "MINUTE",
                    "SECOND",
                    "WKDAY",
                ]
            ]
            df_dt = pd.concat(
                [
                    dt.year,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second,
                    dt.weekday,
                ],
                axis=1,
                keys=new_column_names,
            )

            dfs.append(df_dt)

        dfs_dt = pd.concat(dfs, axis=1)
        return dfs_dt

    def get_feature_names(self):
        return [c for sublist in self.column_names for c in sublist]
