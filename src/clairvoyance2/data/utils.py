from typing import Mapping, Tuple, Union

import numpy as np
import pandas as pd


def all_items_are_of_types(series: pd.Series, of_types: Union[type, Tuple[type, ...]]) -> bool:
    return not series.apply(lambda x, t=of_types: not isinstance(x, t)).any()


def _extract_np_dtype_from_pd(_type: type) -> type:
    """Helper to handle `pandas` references to `numpy` dtypes, e.g. `dtype('int64')`.
    If `_type` is a `pandas` reference to a `numpy` dtype, will return the underlying `numpy` dtype.
    Otherwise will return `_type` as passed.
    """
    try:
        _type = _type.type  # type: ignore
    except AttributeError:
        pass
    return _type


# For the purposes of data type comparisons in this library,
# we assume the following numpy types be equivalent to Python types.
NP_EQUIVALENT_TYPES_MAP: Mapping[type, type] = {
    np.int_: int,
    np.int64: int,
    np.float_: float,
    np.float64: float,
    np.object_: object,
}


def _np_dtype_to_python_type(dtype: type) -> type:
    if dtype in NP_EQUIVALENT_TYPES_MAP:
        return NP_EQUIVALENT_TYPES_MAP[dtype]
    else:
        return dtype


def python_type_from_np_pd_dtype(dtype: type) -> type:
    return _np_dtype_to_python_type(_extract_np_dtype_from_pd(dtype))
