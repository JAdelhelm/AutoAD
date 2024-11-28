# %%

############## example data #############

import pandas as pd
import numpy as np
import pdb
from sklearn import set_config

from autoad.autoad import AutoAD
from pyod.models.lof import LOF

set_config(transform_output="pandas")

X_train  = pd.read_csv("./X_train.csv")
X_test = pd.read_csv("./X_test.csv")

pipeline_ad = AutoAD()


pipeline_ad.fit(X=X_train, clf_ad=LOF())
X_transformed = pipeline_ad.transform(X=X_test)
X_transformed
