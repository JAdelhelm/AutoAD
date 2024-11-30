import pdb
import pandas as pd
from numpy import ndarray
from sklearn import set_config
import pyod
from sklearn.pipeline import FeatureUnion
from pandas.api.types import is_numeric_dtype

from .pipeline_configuration import PipelinesConfiguration

set_config(transform_output="pandas")


class AutoAD:
    def __init__(self) -> None:
        self.fitted_pipeline = None
        self.clf_ad_fitted = None

        self.X_fit = None
        self.X_transformed = None

        self.feature_importances_clf = None

    def fit(
        self,
        X: ndarray | pd.DataFrame,
        y=None,
        remove_columns_no_variance: bool = True,
        datetime_columns: list = None,
        numerical_columns: list = None,
        clf_ad: pyod.models = None,
        pipeline_type: str = "",
    ):
        """
        Fits the pipeline and anomaly detection method, based on incoming data X.
        """

        X = AutoAD.prepare_data_for_pipeline(
            X, numerical_columns, datetime_columns
        )

        preprocess_pipeline = AutoAD.create_pipeline(
            pipeline_type=pipeline_type
        )

        if remove_columns_no_variance is True:
            X = AutoAD.remove_variance(X)

        self.X_fit = X.copy(deep=True)

        self.fitted_pipeline = preprocess_pipeline.fit(X=X)
        if clf_ad is not None:
            X_ad_prepared = self.fitted_pipeline.transform(
                X=X
            )  # needs to be prepared, because of only numeric

            self.clf_ad_fitted = clf_ad.fit(X=X_ad_prepared)

        return self.fitted_pipeline

    def transform(self, X: ndarray | pd.DataFrame, y=None):
        if self.fitted_pipeline is not None:
            self.X_transformed = X

            if list(X.columns) != list(self.X_fit.columns):
                raise Exception(
                    f"Column names must be identical!\n\n{self.X_fit.columns}\n{X.columns}"
                )
            if self.clf_ad_fitted is not None:
                X_transformed = self.fitted_pipeline.transform(X=X)
                self.feature_importances(X_transformed)
                y_scores = self.clf_ad_fitted.decision_function(X_transformed)
                
                X["AD_score"] = y_scores
                X["MAD_Total"] = X_transformed["MAD_Total"]
                X["Tukey_Total"] = X_transformed["Tukey_Total"]

                return X.sort_values("AD_score", ascending=False)

            return self.fitted_pipeline.transform(X=X)
        else:
            raise Exception("Please fit the pipeline first.")

    def fit_transform(self, X: ndarray | pd.DataFrame, y=None):
        return AutoAD.create_pipeline().fit(X=X).transform(X=X)

    def feature_importances(self, X_transformed):
        try:
            feature_importance_df = pd.DataFrame({
                'Feature': X_transformed.columns,
                'Importance': self.clf_ad_fitted.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            self.feature_importances_clf = feature_importance_df
        except Exception as e:
            print("Couldnt get feature importances..")

    @staticmethod
    def create_pipeline(pipeline_type: str = "", n_jobs: int = -1):
        """
        Creates pipeline, based on provided type

        :str pipeline_type: Types available ("categorical", "numeric", timeseries)
        """
        pipeline_type = pipeline_type.lower()
        if pipeline_type == "categorical":
            pipelines = [
                ("Categorical", PipelinesConfiguration.categorical_pipeline()),
                (
                    "MissingIndicator",
                    PipelinesConfiguration.nan_marker_pipeline(),
                ),
            ]
        elif pipeline_type == "numeric":
            pipelines = [
                ("Numerical", PipelinesConfiguration.numeric_pipeline()),
                (
                    "MissingIndicator",
                    PipelinesConfiguration.nan_marker_pipeline(),
                ),
            ]
        elif pipeline_type == "timeseries":
            pipelines = [
                ("Timeseries", PipelinesConfiguration.timeseries_pipeline()),
                (
                    "MissingIndicator",
                    PipelinesConfiguration.nan_marker_pipeline(),
                ),
            ]
        else:
            pipelines = [
                ("Numerical", PipelinesConfiguration.numeric_pipeline()),
                ("Categorical", PipelinesConfiguration.categorical_pipeline()),
                ("Timeseries", PipelinesConfiguration.timeseries_pipeline()),
                (
                    "MissingIndicator",
                    PipelinesConfiguration.nan_marker_pipeline(),
                ),
            ]

        return FeatureUnion(transformer_list=pipelines, n_jobs=n_jobs)

    @staticmethod
    def prepare_data_for_pipeline(X, numerical_columns, datetime_columns):
        return PipelinesConfiguration.pre_pipeline(
            datetime_columns=datetime_columns,
            numerical_columns=numerical_columns,
        ).fit_transform(X=X)

    @staticmethod
    def remove_variance(X) -> pd.DataFrame:
        columns_with_no_variance = []

        for col in X.columns:
            if True is is_numeric_dtype(X[col]) and X[col].std() <= 0.1:
                columns_with_no_variance.append(col)

        return X.drop(columns_with_no_variance, axis=1)
