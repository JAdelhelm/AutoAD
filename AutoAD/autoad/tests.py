import pandas as pd
import numpy as np
import pytest
from .autoad import AutoAD
import pdb

# Create some dummy data
X_train = pd.DataFrame(
    {
        "ID": [1, 2, 3, 4],
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Rank": ["A", "A", "A", "A"],
        "Age": [25, 30, 35, np.nan],
        "Salary": [50000.00, 60000.50, 75000.75, 80000],
        "Hire Date": pd.to_datetime(
            ["2020-01-15", "2019-05-22", "2018-08-30", "2021-04-12"]
        ),
        "Is Manager": [False, True, False, ""],
    }
)

X_test = pd.DataFrame(
    {
        "ID": [1, 2, 3, 4],
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Rank": ["A", "B", "C", "D"],
        "Age": [25, 30, 35, np.nan],
        "Salary": [50000.00, 60000.50, 75000.75, 8000000],  # Outlier in Salary
        "Hire Date": pd.to_datetime(
            ["2020-01-15", "2019-05-22", "2018-08-30", "2021-04-12"]
        ),
        "Is Manager": [False, True, False, ""],
    }
)


import unittest


class TestPipelineConsistency(unittest.TestCase):

    # def test_inconsistent_column_1(self):
    #     """Open Issue"""
    #     df_inconsistent = pd.DataFrame({"ID": [1, 2, 3, "42"]})
    #     pipeline = AutoAD()
    #     pipeline.fit_transform(df_inconsistent)

    # def test_inconsistent_column_2(self):
    #     """Open Issue"""
    #     df_inconsistent = pd.DataFrame(
    #         {"Name": ["Alice", "Bob", "Charlie", 42]}
    #     )
    #     pipeline = AutoAD()
    #     pipeline.fit_transform(df_inconsistent)

    def test_remove_variance_train_test(self):
        """
        Remove columns in X_train and X_test, if no variance in X_test,
        if remove_columns_no_variance = True
        """
        pipeline = AutoAD()
        pipeline.fit(X=X_train, y=None, remove_columns_no_variance=True)
        pipeline.transform(X=X_test, y=None)

        self.assertEqual(
            list(pipeline.X_fit.columns), list(pipeline.X_transformed.columns)
        )

    def test_dtype_numerical(self):
        df_numerical = pd.DataFrame({"ID": [1, 2, 3, 4]})
        pipeline = AutoAD()
        transformed_numerical = pipeline.fit_transform(df_numerical)
        self.assertEqual(
            transformed_numerical.shape[0],
            df_numerical.shape[0],
            "Row count mismatch for numerical DataFrame.",
        )

    def test_column_consistency_after_transformation(self):
        """
        Test that the output columns after transformation still retain the original names
        in some form even if the transformation alters them.
        """
        pipeline = AutoAD()
        pipeline.fit(X=X_train, y=None, remove_columns_no_variance=False)
        transformed_test = pipeline.transform(X=X_test, y=None)

        # Verify that all original columns are present, possibly renamed
        for col in X_train.columns:
            self.assertTrue(
                any(
                    col in transformed_col
                    for transformed_col in transformed_test.columns
                ),
                f"Original column {col} not found in transformed columns.",
            )

    def test_transform_column_count(self):
        """
        Test that the number of columns in the transformed data is greater than or equal
        to the original data (indicating possible feature expansion).
        """
        pipeline = AutoAD()
        pipeline.fit(X=X_train, y=None, remove_columns_no_variance=True)
        transformed_test = pipeline.transform(X=X_test, y=None)

        # Check that the number of columns in the transformed data is >= the original data
        self.assertGreaterEqual(
            len(transformed_test.columns),
            len(X_train.columns),
            "The number of columns in the transformed data is less than expected.",
        )

    def test_transform_with_outliers(self):
        """
        Test how the pipeline handles outliers, such as an unusually high salary in the test data.
        """
        pipeline = AutoAD()
        pipeline.fit(X=X_train, y=None, remove_columns_no_variance=True)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertIn(
            8000000,
            transformed_test["Salary"].values,
            "Outlier in 'Salary' column has been incorrectly removed or altered.",
        )

    def test_pipeline_with_datetime_columns(self):
        """
        Test that the pipeline correctly processes datetime columns such as 'Hire Date'.
        """
        pipeline = AutoAD()
        pipeline.fit(
            X=X_train,
            y=None,
            remove_columns_no_variance=True,
            datetime_columns=["Hire Date"],
        )
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertTrue(
            any("Hire Date" in col for col in transformed_test.columns),
            "'Hire Date' column is missing after transformation.",
        )

    def test_pipeline_with_categorical_columns(self):
        """
        Test that the pipeline correctly processes categorical columns such as 'Rank'.
        """
        pipeline = AutoAD()
        pipeline.fit(X=X_train, y=None, remove_columns_no_variance=False)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertTrue(
            any("Rank" in col for col in transformed_test.columns),
            "'Rank' column is missing after transformation or was not properly processed as categorical.",
        )

    def test_pipeline_with_dropped_columns(self):
        """
        Test that columns are properly dropped when specified as having no variance or for other reasons.
        """
        pipeline = AutoAD()
        pipeline.fit(X=X_train, y=None, remove_columns_no_variance=True)
        transformed_test = pipeline.transform(X=X_test, y=None)

        self.assertNotIn(
            "Rank",
            transformed_test.columns,
            "'Rank' column was not dropped as expected due to lack of variance in training data.",
        )

    def test_AutoAD_empty_df(self):
        pipeline = AutoAD()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="at least one array or dtype is required"):
            X_output = pipeline.fit_transform(empty_df)

    def test_AutoAD_single_row(self):
        pipeline = AutoAD()
        df_single = pd.DataFrame(
            {"Name": ["Alice"], "Age": [25], "Sex": ["female"]}
        )
        X_output = pipeline.fit_transform(df_single)

        assert isinstance(X_output, pd.DataFrame)
        assert len(X_output) == 1, "Output should contain one row"

    def test_AutoAD_test_default_behavior(self):
        pipeline = AutoAD()
        X_output = pipeline.fit_transform(X_train)
        # pdb.set_trace()

        assert isinstance(X_output, pd.DataFrame)







    def test_pipeline_handling_mixed_dtypes(self):
        """
        Test how the pipeline handles a mix of numerical, categorical, and text data.
        """
        mixed_dtype_df = pd.DataFrame(
            {
                "ID": [1, 2, 3, 4],
                "Category": ["A", "B", "A", "C"],
                "Value": [100.5, 200.75, np.nan, 300.1],
                "Description": ["Test1", "Test2", "Test3", "Test4"],
            }
        )
        pipeline = AutoAD()
        transformed_data = pipeline.fit_transform(mixed_dtype_df)

        # Ensure no unexpected errors occurred
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(mixed_dtype_df)

    def test_pipeline_with_missing_values(self):
        """
        Test the pipeline's handling of missing values in various columns.
        """
        df_missing_values = pd.DataFrame(
            {
                "Feature1": [1, 2, np.nan, 4],
                "Feature2": [np.nan, "B", "C", "D"],
                "Feature3": [100.5, 200.75, 300.1, np.nan],
            }
        )
        pipeline = AutoAD()
        transformed_data = pipeline.fit_transform(df_missing_values)

        # Verify that missing values are handled correctly
        assert isinstance(transformed_data, pd.DataFrame)
        assert not transformed_data.isnull().values.any(), "Missing values not handled properly."

    def test_pipeline_retains_index(self):
        """
        Test that the pipeline retains the index of the original DataFrame.
        """
        df_indexed = pd.DataFrame(
            {
                "Index": [10, 20, 30, 40],
                "Feature1": [1, 2, 3, 4],
            }
        ).set_index("Index")
        pipeline = AutoAD()
        transformed_data = pipeline.fit_transform(df_indexed)

        assert isinstance(transformed_data, pd.DataFrame)
        assert transformed_data.index.equals(df_indexed.index), "Index mismatch after transformation."



    def test_pipeline_with_unseen_categories(self):
        """
        Test the pipeline's behavior when encountering unseen categories in test data.
        """
        train_data = pd.DataFrame({"Category": ["A", "B", "C"]})
        test_data = pd.DataFrame({"Category": ["A", "D", "E"]})

        pipeline = AutoAD()
        pipeline.fit(train_data)
        pdb.set_trace()
        transformed_test = pipeline.transform(test_data)

        # Ensure unseen categories are handled without errors
        assert isinstance(transformed_test, pd.DataFrame)
        assert "Category" in transformed_test.columns



    def test_pipeline_feature_scaling(self):
        """
        Test that numerical features are correctly scaled if scaling is enabled.
        """
        df_scaling = pd.DataFrame(
            {
                "Feature1": [1, 2, 3, 4],
                "Feature2": [100, 200, 300, 400],
            }
        )
        pipeline = AutoAD(scale_numerical_features=True)
        transformed_data = pipeline.fit_transform(df_scaling)

        # Ensure numerical features are scaled between 0 and 1
        assert transformed_data["Feature1"].min() >= 0 and transformed_data["Feature1"].max() <= 1
        assert transformed_data["Feature2"].min() >= 0 and transformed_data["Feature2"].max() <= 1

    def test_pipeline_with_custom_preprocessing(self):
        """
        Test the pipeline's ability to accept custom preprocessing steps.
        """
        def custom_preprocessor(df):
            df["CustomFeature"] = df["ID"] * 2
            return df

        df_custom = pd.DataFrame({"ID": [1, 2, 3, 4]})
        pipeline = AutoAD(custom_preprocessor=custom_preprocessor)
        transformed_data = pipeline.fit_transform(df_custom)

        # Ensure custom preprocessing has been applied
        assert "CustomFeature" in transformed_data.columns
        assert (transformed_data["CustomFeature"] == df_custom["ID"] * 2).all()














if __name__ == "__main__":
    # Running the tests
    pytest.main()
