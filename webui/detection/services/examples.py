"""Example tables to try the interface without a file of your own.

Every example is generated here with a fixed seed, so it needs no data files and is the
same on every machine. An example consists of a training table that is mostly normal
with a few planted problems, and a second table with new problems to check against the
fitted model. Both go through the same code as an uploaded CSV file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Example:
    key: str
    title: str
    description: str
    #: What the planted problems are, shown next to the description.
    look_for: str
    build: Callable[[bool], pd.DataFrame]
    #: Values of the configure form that differ from the defaults.
    settings: dict = field(default_factory=dict)

    @property
    def file_name(self) -> str:
        return f"example_{self.key}.csv"

    @property
    def check_file_name(self) -> str:
        return f"example_{self.key}_new.csv"

    def training_csv(self) -> bytes:
        return _csv(self.build(False))

    def check_csv(self) -> bytes:
        return _csv(self.build(True))


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _orders(new: bool) -> pd.DataFrame:
    """Orders of an online shop."""
    n_rows = 120 if new else 400
    rng = np.random.default_rng(11 if new else 1)
    quantity = rng.integers(1, 6, n_rows)
    unit_price = rng.normal(40, 6, n_rows).clip(15, None)
    start = "2025-07-01" if new else "2025-01-01"
    frame = pd.DataFrame(
        {
            "Order ID": np.arange(50_000, 50_000 + n_rows) + (400 if new else 0),
            "Ordered": pd.date_range(start, periods=n_rows, freq="10h").strftime("%Y-%m-%d %H:%M"),
            "Channel": rng.choice(["web", "app", "phone"], n_rows, p=[0.6, 0.3, 0.1]),
            "Payment": rng.choice(["card", "paypal", "invoice"], n_rows, p=[0.5, 0.35, 0.15]),
            "Country": rng.choice(["DE", "AT", "CH", "NL"], n_rows, p=[0.7, 0.12, 0.1, 0.08]),
            "Quantity": quantity,
            "Amount": (quantity * unit_price).round(2),
            "Shipping": rng.choice([0.0, 4.9, 9.9], n_rows, p=[0.3, 0.6, 0.1]),
        }
    )
    if new:
        frame.loc[5, ["Quantity", "Amount"]] = [3, 12_500.0]  # decimal point slipped
        frame.loc[33, "Payment"] = "crypto"  # category the model never saw
        frame.loc[61, "Amount"] = np.nan
        frame.loc[90, ["Quantity", "Amount"]] = [250, 9_800.0]  # bulk order
    else:
        frame.loc[17, ["Quantity", "Amount"]] = [2, 8_450.0]  # decimal point slipped
        frame.loc[112, ["Quantity", "Amount"]] = [180, 7_300.0]  # bulk order
        frame.loc[203, "Amount"] = -129.9  # refund booked as an order
        frame.loc[288, "Shipping"] = 149.0
        frame.loc[[64, 240, 351], "Amount"] = np.nan
        # The same order entered twice under a new order ID.
        columns = [col for col in frame.columns if col != "Order ID"]
        frame.loc[300, columns] = frame.loc[150, columns].to_numpy()
    return frame


def _sensors(new: bool) -> pd.DataFrame:
    """Hourly readings of three machines. Pressure follows temperature."""
    n_rows = 150 if new else 600
    rng = np.random.default_rng(12 if new else 2)
    temperature = rng.normal(70, 4, n_rows)
    pressure = 2.0 + 0.05 * (temperature - 70) + rng.normal(0, 0.04, n_rows)
    start = "2025-04-01" if new else "2025-03-01"
    frame = pd.DataFrame(
        {
            "Timestamp": pd.date_range(start, periods=n_rows, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
            "Machine": rng.choice(["M-01", "M-02", "M-03"], n_rows),
            "Temperature": temperature.round(1),
            "Pressure": pressure.round(3),
            "Vibration": rng.gamma(4.0, 0.05, n_rows).round(3),
            "Power": rng.normal(12, 0.8, n_rows).round(2),
            "Status": rng.choice(["ok", "ok", "ok", "maintenance"], n_rows),
        }
    )
    if new:
        frame.loc[20, "Temperature"] = 131.0
        frame.loc[75, ["Temperature", "Pressure"]] = [64.0, 2.31]  # each normal, together not
        frame.loc[110, "Status"] = "error"  # category the model never saw
        frame.loc[[40, 41, 42], "Vibration"] = np.nan  # sensor offline
    else:
        frame.loc[45, "Temperature"] = 118.5  # overheating
        frame.loc[180, "Vibration"] = 1.9  # bearing damage
        frame.loc[333, "Power"] = 0.0  # machine off, still reporting
        # Each value alone is normal, the combination is not: hot machine, low pressure.
        frame.loc[260, ["Temperature", "Pressure"]] = [78.0, 1.72]
        frame.loc[415, ["Temperature", "Pressure"]] = [63.0, 2.29]
        frame.loc[[500, 501], "Pressure"] = np.nan
    return frame


def _employees(new: bool) -> pd.DataFrame:
    """Staff master data, with the columns of ``examples/X_train.csv`` in the repository."""
    n_rows = 60 if new else 250
    rng = np.random.default_rng(13 if new else 3)
    years = rng.integers(0, 16, n_rows)
    age = (22 + years + rng.integers(0, 20, n_rows)).clip(None, 64)
    rank = rng.choice(["A", "B", "C", "D"], n_rows, p=[0.4, 0.3, 0.2, 0.1])
    base = pd.Series(rank).map({"A": 42_000, "B": 55_000, "C": 70_000, "D": 90_000}).to_numpy()
    hired = pd.Timestamp("2025-06-30") - pd.to_timedelta(years * 365 + rng.integers(0, 365, n_rows), unit="D")
    frame = pd.DataFrame(
        {
            "ID": np.arange(1, n_rows + 1) + (250 if new else 0),
            "Rank": rank,
            "Age": age,
            "Salary": (base + years * 1_200 + rng.normal(0, 3_000, n_rows)).round(2),
            "Hire Date": hired.strftime("%Y-%m-%d"),
            "Is Manager": rng.random(n_rows) < 0.15,
            "Department": rng.choice(["HR", "Engineering", "Finance", "Marketing", "Sales"], n_rows),
            "Years at Company": years,
        }
    )
    if new:
        frame.loc[8, "Salary"] = 8_000_000.0
        frame.loc[[15, 30, 44], "Age"] = np.nan
        frame.loc[52, "Department"] = "Legal"  # category the model never saw
    else:
        frame.loc[12, "Salary"] = 5_400.0  # a zero is missing
        frame.loc[77, "Salary"] = 940_000.0
        frame.loc[140, "Age"] = 17
        frame.loc[199, "Years at Company"] = 48
        frame.loc[[31, 166], "Age"] = np.nan
    return frame


EXAMPLES: dict[str, Example] = {
    example.key: example
    for example in (
        Example(
            key="orders",
            title="Shop orders",
            description="400 orders of an online shop with channel, payment, country, quantity and amount.",
            look_for="Wrong amounts, a bulk order, missing amounts and one order entered twice under a new ID.",
            build=_orders,
            settings={"contamination": 0.02},
        ),
        Example(
            key="sensors",
            title="Machine sensors",
            description="600 hourly readings of three machines: temperature, pressure, vibration and power.",
            look_for="Spikes of single sensors, and readings that are normal alone but do not fit together.",
            build=_sensors,
            settings={"contamination": 0.02},
        ),
        Example(
            key="employees",
            title="Employees",
            description="250 staff records with rank, age, salary, hire date and department.",
            look_for="Implausible salaries, ages and years of service, and missing ages.",
            build=_employees,
            settings={"contamination": 0.03},
        ),
    )
}


def for_file_name(file_name: str) -> Example | None:
    """The example an analysis was started from, recognised by its file name."""
    return next((example for example in EXAMPLES.values() if example.file_name == file_name), None)
