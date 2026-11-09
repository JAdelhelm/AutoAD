import tempfile

import numpy as np
import pandas as pd
from django.test import override_settings

#: Rows with injected anomalies in :func:`make_frame` (0-based).
ANOMALY_ROWS = (7, 42)
#: Row 60 repeats row 10 under a new order ID (0-based).
DUPLICATE_ROWS = (10, 60)


def make_frame(n_rows: int = 120, seed: int = 0) -> pd.DataFrame:
    """Plausible orders with two clearly wrong rows."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "Order ID": np.arange(1000, 1000 + n_rows),
            "Amount": rng.normal(100, 10, n_rows).round(2),
            "Quantity": rng.integers(1, 6, n_rows),
            "Channel": rng.choice(["web", "store", "phone"], n_rows),
            "Ordered": pd.date_range("2025-01-01", periods=n_rows, freq="D").strftime("%Y-%m-%d"),
        }
    )
    for row, values in zip(ANOMALY_ROWS, ([10_000.0, 500], [-5_000.0, 400]), strict=True):
        if row < n_rows:
            frame.loc[row, ["Amount", "Quantity"]] = values
    original, copy = DUPLICATE_ROWS
    if copy < n_rows:
        columns = [col for col in frame.columns if col != "Order ID"]
        frame.loc[copy, columns] = frame.loc[original, columns].to_numpy()
    return frame


def csv_bytes(frame: pd.DataFrame, **kwargs) -> bytes:
    return frame.to_csv(index=False, **kwargs).encode("utf-8")


def configure_data(n_columns: int, **overrides) -> dict:
    """POST data of the configure form with default settings and every role on "auto"."""
    data = {
        "detector": "iforest",
        "contamination": "0.05",
        "n_neighbors": "",
        "n_estimators": "",
        "drop_constant": "on",
        "drop_id_columns": "on",
        "max_binary_categories": "20",
        "tukey_factor": "1.5",
        "mad_threshold": "3.5",
        "missing_indicator": "count",
        "scaler": "robust",
        "duplicate_tolerance": "0",
    }
    data.update({f"role_{i}": "auto" for i in range(n_columns)})
    data.update(overrides)
    return data


class TemporaryMediaMixin:
    """Store uploads and results of a test class in a temporary directory."""

    @classmethod
    def setUpClass(cls):
        cls._media = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls._media_override = override_settings(MEDIA_ROOT=cls._media.name)
        cls._media_override.enable()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._media_override.disable()
        cls._media.cleanup()
