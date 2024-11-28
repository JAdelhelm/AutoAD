# %%

############## example data #############

import pandas as pd
import numpy as np
from sklearn import set_config

set_config(transform_output="pandas")

X_train = pd.DataFrame(
    {
        "ID": [1, 2, 3, 4],
        "Name": ["Alice", "Alice", "Alice", "Alice"],
        "Rank": ["A", "B", "C", "D"],
        "Age": [25, 30, 35, 40],
        "Salary": [50000.00, 60000.50, 75000.75, 8_000],
        "Hire Date": pd.to_datetime(
            ["2020-01-15", "2019-05-22", "2018-08-30", "2021-04-12"]
        ),
        "Is Manager": [False, True, False, ""],
    }
)
X_test = pd.DataFrame(
    {
        "ID": [1, 2, 3, 4],
        "Name": ["Alice", "Alice", "Alice", "Bob"],
        "Rank": ["A", "B", "C", "D"],
        "Age": [25, 30, 35, np.nan],
        "Salary": [50000.00, 60000.50, 75000.75, 8_000_000],
        "Hire Date": pd.to_datetime(
            ["2020-01-15", "2019-05-22", "2018-08-30", "2021-04-12"]
        ),
        "Is Manager": [False, True, False, ""],
    }
)


########################################
import pdb
from autoad.autoad import AutoAD
from pyod.models.iforest import IForest


pipeline_ad = AutoAD()


pipeline_ad.fit(X=X_train, clf_ad=IForest(), pipeline_type="")
X_transformed = pipeline_ad.transform(X=X_test)
X_transformed
