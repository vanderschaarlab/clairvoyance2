from typing import Any

import numpy as np
import pandas as pd


def isnan(value: Any) -> bool:
    if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
        raise TypeError(f"Value of type {type(value)} is not supported")
    try:
        isnan_ = bool(np.isnan(value))  # numpy.bool_ --> bool
    except TypeError:  # pylint: disable=broad-except
        isnan_ = False
    return isnan_


def equal_or_nans(a: Any, b: Any) -> bool:
    a_isnan = isnan(a)
    b_isnan = isnan(b)
    if a_isnan:
        return True if b_isnan else False
    else:
        return a == b
