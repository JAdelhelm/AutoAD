# %%


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from bitstring import BitArray


from sklearn import set_config

set_config(transform_output="pandas")


class NaNColumnCreatorTotal(BaseEstimator, TransformerMixin):
    """
    NaNColumnCreator that creates new columns and marks NaN-Entries with 1.

    Steps:
        (1) The numerical values are concatenated row-wise and converted into a string.\n
        (2) Subsequently, they are inverted so that binary sorting is done from left to right (Simpler interpretation of the numbers).\n
        (3) A 0 is added to the beginning due to the two's complement to obtain a positive number.\n
        !! Attention: There is no (Z-)scaling necessary here, as an ordinal order makes sense!\n
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.astype(int)
        try:
            X_Transformed = pd.DataFrame()

            transformed_rows = []

            for row in X.values:
                # Casten zum String + Inverse (Da man pro Zeile in der Spalte nur 0 und 1 hat)
                binary_str = "".join(map(str, row))[::-1]
                binary_str = "0" + binary_str
                integer_representation = BitArray(bin=binary_str).int
                transformed_rows.append(integer_representation)

            X_Transformed["NaNs-Binary"] = transformed_rows

            self.column_names_missing = X_Transformed.columns

            return X_Transformed
        except Exception as e:
            print(f"Error: {e}")
            return X

    def get_feature_names(self, input_features=None):
        return self.column_names_missing
