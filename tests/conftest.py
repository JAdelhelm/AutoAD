from pathlib import Path

import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def x_train() -> pd.DataFrame:
    """10 employees, no missing values (examples/X_train.csv)."""
    return pd.read_csv(EXAMPLES / "X_train.csv")


@pytest.fixture
def x_test() -> pd.DataFrame:
    """The same employees with two salary outliers (rows 3 and 9) and three missing ages."""
    return pd.read_csv(EXAMPLES / "X_test.csv")
