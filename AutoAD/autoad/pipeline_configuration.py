"""
Module for configuring and creating preprocessing pipelines.

This module provides the `PipelinesConfiguration` class which includes various methods to create
and configure pipelines for preprocessing numerical, categorical, timeseries, and pattern data.

Classes:
    PipelinesConfiguration: Configures pipelines for preprocessing different types of data.
    XPatternDropper: Drops specified columns from the data.

Imports:
    numpy as np
    pandas as pd
    sklearn and other relevant libraries for data preprocessing and transformation.
"""

import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import MissingIndicator

from .pipelines.dummy.TypeInferenceTransformer import TypeInferenceTransformer
from .pipelines.statistical.TukeyTransformer import TukeyTransformer
from .pipelines.statistical.TukeyTransformerTotal import TukeyTransformerTotal
from .pipelines.statistical.MedianAbsolutDeviation import (
    MedianAbsolutDeviation,
)
from .pipelines.statistical.MedianAbsolutDeviationTotal import (
    MedianAbsolutDeviationTotal,
)
from .pipelines.timeseries.DateEncoder import DateEncoder
from .pipelines.timeseries.TimeSeriesImputer import TimeSeriesImputer


class PipelinesConfiguration:
    """
    The PipelinesConfiguration class represents the class
    to configure pipelines for data preprocessing.

    There are different SchemaTransformer,
    to handle different datatypes as input.

    Methods
    -------
    pre_pipeline(datetime_columns=None, exclude_columns=None):
        Creates a preprocessing pipeline to prepare data for transformation.

    nan_marker_pipeline():
        Creates a pipeline that marks columns with NaN values.

    numeric_pipeline():
        Creates a pipeline for preprocessing numerical data.

    categorical_pipeline():
        Creates a pipeline for preprocessing categorical data.

    timeseries_pipeline():
        Creates a pipeline for preprocessing timeseries data.


    Parameters
    ----------
    datetime_columns : list
        List of Time-Columns that should be converted to timestamp data types.

    numerical_columns : list
        List of columns that should be dropped.
    """

    @staticmethod
    def drop_datetime_columns(X):
        datetime_cols = X.select_dtypes(
            include=["datetime64[ns]", "datetimetz"]
        ).columns
        return X.drop(columns=datetime_cols, errors="ignore")

    @staticmethod
    def drop_object_columns(X):
        object_cols = X.select_dtypes(include=["object", "string"]).columns
        return X.drop(columns=object_cols, errors="ignore")

    @staticmethod
    def pre_pipeline(datetime_columns, numerical_columns):

        original_preprocessor = Pipeline(
            steps=[
                (
                    "Preprocessing",
                    ColumnTransformer(
                        transformers=[
                            (
                                "Standard TypeCast Transformer",
                                TypeInferenceTransformer(
                                    datetime_columns=datetime_columns,
                                    numerical_columns=numerical_columns,
                                ),
                                make_column_selector(dtype_include=None),
                            )
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                        # Disable Prefix behaviour
                        verbose_feature_names_out=False,
                    ),
                )
            ]
        )

        return original_preprocessor

    @staticmethod
    def nan_marker_pipeline():
        nan_marker_preprocessor = Pipeline(
            steps=[
                (
                    "NanMarker",
                    ColumnTransformer(
                        transformers=[
                            (
                                "nan_marker_columns",
                                Pipeline(
                                    steps=[
                                        (
                                            "nan_marker",
                                            MissingIndicator(features="all"),
                                        )
                                    ]
                                ),
                                make_column_selector(
                                    dtype_include=(np.number, np.object_)
                                ),
                            ),
                        ],
                        remainder="passthrough",
                        n_jobs=-1,
                        verbose=True,
                        verbose_feature_names_out=False,
                    ),
                ),
                # Step to dynamically drop datetime columns
                (
                    "Drop_Datetime_Columns",
                    FunctionTransformer(
                        PipelinesConfiguration.drop_datetime_columns
                    ),
                ),
                # Step to dynamically drop object columns
                (
                    "Drop_Object_Columns",
                    FunctionTransformer(
                        PipelinesConfiguration.drop_object_columns
                    ),
                ),
                # Step to impute missing columns after they are marked
                (
                    "Impute Columns",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
            ]
        )

        return nan_marker_preprocessor

    @staticmethod
    def numeric_pipeline():
        return Pipeline(
            steps=[
                (
                    "Statistical methods",
                    ColumnTransformer(
                        transformers=[
                            (
                                "tukey",
                                Pipeline(
                                    steps=[
                                        (
                                            "Tukey_impute",
                                            IterativeImputer(
                                                initial_strategy="median"
                                            ),
                                        ),
                                        (
                                            "tukey",
                                            TukeyTransformer(factor=1.5),
                                        ),
                                        (
                                            "tukey_total",
                                            TukeyTransformerTotal(),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "z_mod",
                                Pipeline(
                                    steps=[
                                        (
                                            "z_mod_impute",
                                            IterativeImputer(
                                                initial_strategy="median"
                                            ),
                                        ),
                                        ("z_mod", MedianAbsolutDeviation()),
                                        (
                                            "z_mod_total",
                                            MedianAbsolutDeviationTotal(),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                            (
                                "pass_cols",
                                Pipeline(
                                    steps=[
                                        (
                                            "_pass_cols_",
                                            SimpleImputer(
                                                strategy="median",
                                                keep_empty_features=True,
                                            ),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.number),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                        verbose_feature_names_out=False,
                    ),
                ),
            ]
        )

    @staticmethod
    def categorical_pipeline():

        return Pipeline(
            steps=[
                (
                    "Preprocessing_Categorical",
                    ColumnTransformer(
                        transformers=[
                            (
                                "categorical",
                                Pipeline(
                                    steps=[
                                        (
                                            "C",
                                            SimpleImputer(
                                                strategy="most_frequent",
                                                keep_empty_features=True,
                                            ),
                                        ),
                                        (
                                            "BinaryEnc",
                                            BinaryEncoder(
                                                handle_unknown="indicator"
                                            ),
                                        ),
                                    ]
                                ),
                                make_column_selector(dtype_include=np.object_),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                    ),
                )
            ]
        )

    @staticmethod
    def timeseries_pipeline():
        """
        Columns in datetime format will be dropped, because most ml-algorithms can only handle numeric data.
        Open Task: Add MissingIndicator to TimeSeries
        """

        timeseries_preprocessor = Pipeline(
            steps=[
                (
                    "Preprocessing_Timeseries",
                    ColumnTransformer(
                        transformers=[
                            (
                                "timeseries",
                                Pipeline(
                                    steps=[
                                        (
                                            "T",
                                            TimeSeriesImputer(
                                                impute_method="ffill"
                                            ),
                                        ),
                                        ("num_time_dates", DateEncoder()),
                                    ]
                                ),
                                make_column_selector(
                                    dtype_include=(
                                        np.dtype("datetime64[ns]"),
                                        np.datetime64,
                                        "datetimetz",
                                    )
                                ),
                            ),
                        ],
                        remainder="drop",
                        n_jobs=-1,
                        verbose=True,
                        verbose_feature_names_out=False,
                    ),
                ),
                # Step to dynamically drop datetime columns
                (
                    "Drop_Datetime_Columns",
                    FunctionTransformer(
                        PipelinesConfiguration.drop_datetime_columns
                    ),
                ),
                # Step to dynamically drop object columns
                (
                    "Drop_Object_Columns",
                    FunctionTransformer(
                        PipelinesConfiguration.drop_object_columns
                    ),
                ),
            ]
        )

        return timeseries_preprocessor
