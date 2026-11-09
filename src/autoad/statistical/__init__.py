"""Robust univariate outlier methods."""

from autoad.statistical.mad import MedianAbsoluteDeviation
from autoad.statistical.row_sum import RowSum
from autoad.statistical.tukey import TukeyTransformer

__all__ = ["MedianAbsoluteDeviation", "RowSum", "TukeyTransformer"]
