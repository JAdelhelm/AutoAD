"""AutoAD in a nutshell: learn from known-good data, then check new data.

The bundled files contain the same ten employees twice. In ``X_test.csv`` two
salaries are wrong (8,000,000 instead of 8,000, and 900,000) and three ages are
missing.

Run from the repository root::

    pip install -e .
    python examples/main.py
"""

import sys
from pathlib import Path

import pandas as pd
from pyod.models.lof import LOF

try:
    from autoad import AutoAD
except ImportError:  # running from a source checkout without installing the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from autoad import AutoAD

HERE = Path(__file__).resolve().parent


def main() -> None:
    X_train = pd.read_csv(HERE / "X_train.csv")
    X_test = pd.read_csv(HERE / "X_test.csv")

    # 1. Configure and fit. Any PyOD detector works. With only ten training rows, LOF
    #    compares each row with its 3 nearest neighbours.
    ad = AutoAD(
        detector=LOF(n_neighbors=3),
        datetime_columns=["Hire Date"],  # optional: dates are also detected automatically
    )
    ad.fit(X_train)

    # 2. Check what AutoAD learned about the columns.
    print("Column types:", ad.column_types_)
    print("Dropped columns:", ad.dropped_columns_)
    # Features without variance in the ten training rows carry no information:
    # no missing values (missing_count) and no value flagged by the MAD (MAD_Total).
    print("Dropped constant features:", ad.dropped_features_)
    features = ad.get_feature_names_out()
    print(f"The detector sees {len(features)} features:", ", ".join(features))

    # 3. Check the new data. report() returns X_test plus the anomaly information,
    #    predict() a 0/1 label per row: 1 for rows that score higher than the top 10 %
    #    of the training data (the detector's contamination, PyOD's default).
    report = ad.report(X_test)
    report["is_anomaly"] = ad.predict(X_test)

    # The *_Columns explain each row: extreme values (MAD), missing values and
    # categories that did not occur in the training data.
    columns = ["ID", "Name", "Salary", "AD_score", "is_anomaly", "MAD_Columns", "Missing_Columns", "Unseen_Columns"]
    print("\nNew data, most unusual rows first:")
    print(report.sort_values("AD_score", ascending=False)[columns].to_string(index=False))


if __name__ == "__main__":
    main()
