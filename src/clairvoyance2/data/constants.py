import warnings
from typing import Sequence, Union

import numpy as np
import pandas as pd

# Types.

T_ContainerInitializable = Union[pd.DataFrame, np.ndarray]
T_ContainerInitializable_AsTuple = tuple([pd.DataFrame, np.ndarray])  # NOTE: Must match the above.

T_NumericDtype = Union[int, float]
T_NumericDtype_AsTuple = tuple([int, float])  # NOTE: Must match the above.

T_CategoricalDtype = Union[int, float, str]
T_CategoricalDtype_AsTuple = tuple([int, float, str])  # NOTE: Must match the above.

T_FeatureContainer = Union[pd.Series]  # pyright: ignore  # NOTE: May expand this.

T_FeatureIndexDtype = Union[int, str]
T_FeatureIndexDtype_AsTuple = tuple([int, str])  # NOTE: Must match the above. For pandas check, so str -> object.

T_SamplesIndexDtype = Union[int]  # pyright: ignore
T_SamplesIndexDtype_AsTuple = tuple([int])  # NOTE: Must match the above.

# Abbreviation: TS = TimeSeries
T_TSIndexDtype = Union[int, float, np.datetime64]
T_TSIndexDtype_AsTuple = tuple([int, float, np.datetime64])  # NOTE: Must match the above.

T_ElementsObjectType = Union[str]  # pyright: ignore
T_ElementsObjectType_AsTuple = tuple([str])  # pyright: ignore

with warnings.catch_warnings():
    # This is to suppress (expected) FutureWarnings for index types like pd.Int64Index.
    warnings.filterwarnings("ignore", message=r".*Use pandas.Index.*", category=FutureWarning)

    if "NumericIndex" not in dir(pd):
        T_TSIndexClass = Union[
            pd.RangeIndex,
            pd.DatetimeIndex,
            pd.Int64Index,
            pd.UInt64Index,
            pd.Float64Index,
        ]
    else:
        T_TSIndexClass = Union[  # type: ignore
            pd.RangeIndex,
            pd.DatetimeIndex,
            pd.Int64Index,
            pd.UInt64Index,
            pd.Float64Index,
            pd.NumericIndex,  # Future-proofing.
        ]
    T_TSIndexClass_AsTuple = (
        pd.RangeIndex,
        pd.DatetimeIndex,
        pd.Int64Index,
        pd.UInt64Index,
        pd.Float64Index,
        pd.NumericIndex if "NumericIndex" in dir(pd) else pd.Index,  # Future-proofing.
        # NOTE: Other candidates: TimedeltaIndex, PeriodIndex.
    )

    if "NumericIndex" not in dir(pd):
        T_FeatureIndexClass = Union[
            pd.Int64Index,
            pd.UInt64Index,
        ]
    else:
        T_FeatureIndexClass = Union[  # type: ignore
            pd.Int64Index,
            pd.UInt64Index,
            pd.NumericIndex,  # Future-proofing.
        ]
    T_FeatureIndexClass_AsTuple = (
        (
            pd.Int64Index,
            pd.UInt64Index,
            pd.NumericIndex if "NumericIndex" in dir(pd) else pd.Index,  # Future-proofing.
        ),
    )

    if "NumericIndex" not in dir(pd):
        T_SampleIndexClass = Union[
            pd.RangeIndex,
            pd.Int64Index,
            pd.UInt64Index,
        ]
    else:
        T_SampleIndexClass = Union[  # type: ignore
            pd.RangeIndex,
            pd.Int64Index,
            pd.UInt64Index,
            pd.NumericIndex,
        ]
    T_SampleIndexClass_AsTuple = (
        pd.RangeIndex,
        pd.Int64Index,
        pd.UInt64Index,
        pd.NumericIndex if "NumericIndex" in dir(pd) else pd.Index,  # Future-proofing.
    )

T_SampleIndex_Compatible = Union[Sequence[T_SamplesIndexDtype], T_SampleIndexClass]

TIndexDiff = Union[float, int, pd.Timedelta]

# Values.

DEFAULT_PADDING_INDICATOR = -999.0
SAMPLE_INDEX_NAME = "s_idx"
TIME_INDEX_NAME = "t_idx"
