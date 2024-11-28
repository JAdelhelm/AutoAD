# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn import set_config

set_config(transform_output="pandas")


class TukeyTransformerTotal(BaseEstimator, TransformerMixin):
    """
    The TukeyTransformerTotal class aggregates the values w.r.t. TukeyTransformer to get a single  aggregated column.\n
    """

    def __init__(self):
        self.tukey_names = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X_copy = pd.DataFrame()

            X_copy["Tukey_Total"] = X.sum(axis=1)

            self.tukey_names = X_copy.columns

            return X_copy
        except Exception as e:
            print(f"Error: {e}")
            return X

    def get_feature_names(self, input_features=None):
        return self.tukey_names
