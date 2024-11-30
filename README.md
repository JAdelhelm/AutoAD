# AutoAD -  Automated preprocessing-pipeline for the detection of anomalies





## Basic Usage

```bash
conda create -n auto_ad python=3.11
conda activate auto_ad

cd AutoAD

pip install -r requirements.txt
```
---

````python
import pandas as pd
import numpy as np
from sklearn import set_config

set_config(transform_output="pandas")

X_train = pd.DataFrame({

    'ID': [1, 2, 3, 4],                 
    'Name': ['Alice', 'Alice', 'Alice', "Alice"],  
    'Rank': ['A','B','C','D'],
    'Age': [25, 30, 35, 40],                 
    'Salary': [50000.00, 60000.50, 75000.75, 8_000], 
    'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']), 
    'Is Manager': [False, True, False, ""]  
})
X_test = pd.DataFrame({

    'ID': [1, 2, 3, 4],                 
    'Name': ['Alice', 'Alice', 'Alice', "Bob"],  
    'Rank': ['A','B','C','D'],
    'Age': [25, 30, 35, np.nan],                 
    'Salary': [50000.00, 60000.50, 75000.75, 8_000_000], 
    'Hire Date': pd.to_datetime(['2020-01-15', '2019-05-22', '2018-08-30', '2021-04-12']), 
    'Is Manager': [False, True, False, ""]  
})

########################################
import pdb
from autoad.autoad import AutoAD
from pyod.models.iforest import IForest


pipeline_ad = AutoAD()


pipeline_ad.fit(X=X_train, clf_ad=IForest())
X_transformed = pipeline_ad.transform(X=X_test)
X_transformed
````

## Highlights ⭐


#### 📌 Implementation of univariate methods / *Detection of univariate anomalies*
Both methods (MOD Z-Value and Tukey Method) are resilient against outliers, ensuring that the position measurement will not be biased. They also support multivariate anomaly detection algorithms in identifying univariate anomalies.

#### 📌 BinaryEncoder instead of OneHotEncoder for nominal columns / *Big Data and Performance*
   Newest research shows similar results for encoding nominal columns with significantly fewer dimensions.
   - (John T. Hancock and Taghi M. Khoshgoftaar. "Survey on categorical data for neural networks." In: Journal of Big Data 7.1 (2020), pp. 1–41.), Tables 2, 4
   - (Diogo Seca and João Mendes-Moreira. "Benchmark of Encoders of Nominal Features for Regression." In: World Conference on Information Systems and Technologies. 2021, pp. 146–155.), P. 151

#### 📌 Transformation of time series data and standardization of data with RobustScaler / *Normalization for better prediction results*

#### 📌 Labeling of NaN values in an extra column instead of removing them / *No loss of information*



---





## Pipeline - Built-in Logic
<!-- ![Logic of Pipeline](./AutoAD/autoad/img/decision_rules.png) -->
![Logic of Pipeline](https://raw.githubusercontent.com/JAdelhelm/AutoPrep/main/AutoPrep/img/decision_rules.png) 

- I used sklearn's Pipeline and Transformer concept to create this preprocessing pipeline
    - Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - Transformer: https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html





---

<!--
### Reference
- https://www.researchgate.net/publication/379640146_Detektion_von_Anomalien_in_der_Datenqualitatskontrolle_mittels_unuberwachter_Ansatze (German Thesis)
-->



















