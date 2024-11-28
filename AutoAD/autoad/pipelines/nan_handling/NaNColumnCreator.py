# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np

from sklearn import set_config

set_config(transform_output="pandas")


class NaNColumnCreator(BaseEstimator, TransformerMixin):
    """
    NaNColumnCreator that creates new columns and marks NaN-Entries with 1

    Steps:
        (1) Checks if X has NaNs.\n
        (2) Marks the corresponding NaN-values with 1, else 0\n
    """

    def __init__(self):
        pass

    def filter_nans(self, X):
        nan_filter = X.isna()
        return nan_filter.map({False: 0, True: 1})

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.apply(self.filter_nans)

        return X_transformed
