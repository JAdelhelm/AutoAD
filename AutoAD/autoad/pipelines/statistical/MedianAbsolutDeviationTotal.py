# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn import set_config

set_config(transform_output="pandas")


class MedianAbsolutDeviationTotal(BaseEstimator, TransformerMixin):
    """
    The MedianAbsolutDeviationTotal (MAD) class aggregates the values w.r.t. MedianAbsolutDeviation to get a single  aggregated column.
    """

    def __init__(self):
        self.z_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X_copy = pd.DataFrame()

            # Erzeuge ein Total
            X_copy["MAD_Total"] = X.sum(axis=1)

            self.z_names = X_copy.columns

            return X_copy
        except Exception as e:
            print(f"Error: {e}")
            return X

    def get_feature_names(self, input_features=None):
        return self.z_names
