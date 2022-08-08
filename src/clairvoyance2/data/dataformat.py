import warnings
from collections.abc import (
    Sequence as SequenceABC,  # Otherwise name clash with typing.Sequence
)
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from ..utils.dev import raise_not_implemented
from . import DEFAULT_PADDING_INDICATOR
from . import df_constraints as dfc
from .dataformat_base import Copyable, CustomGetItemMixin, WrappedDF
from .feature import CategoricalDtype
from .has_features_mixin import HasFeaturesMixin
from .has_missing_mixin import HasMissingMixin, TMissingIndicator
from .to_numpy_mixin import ToNumpyMixin
from .utils import TIndexDiff, check_index_regular

TFeatureIndex = Union[int, str]
TCategoricalDef = Union[Iterable[TFeatureIndex], Mapping[TFeatureIndex, Tuple[CategoricalDtype, ...]]]

TSamplesIndexDtype = Union[int]  # pyright: ignore

TInitContainer = Union[pd.DataFrame, np.ndarray]
TStaticSamplesContainer = Union[pd.Series]  # pyright: ignore

# Abbreviation: TS = TimeSeries
T_TS_IndexDtype = Union[int, float, np.datetime64]
T_TS_StepContainer = Union[pd.Series]  # pyright: ignore # NOTE: May expand this.


with warnings.catch_warnings():
    # This is to suppress (expected) FutureWarnings for index types like pd.Int64Index.
    warnings.filterwarnings("ignore", message=r".*Use pandas.Index.*", category=FutureWarning)

    T_TS_Index = Union[
        pd.RangeIndex,
        pd.DatetimeIndex,
        pd.Int64Index,
        pd.UInt64Index,
        pd.Float64Index,
        pd.Index,
        "pd.NumericIndex",
    ]

    _DF_CONSTRAINTS_FEATURES = dfc.IndexConstraints(
        (
            pd.Int64Index,
            pd.UInt64Index,
            pd.NumericIndex if "NumericIndex" in dir(pd) else pd.Index,  # Future-proofing.
        ),
        dtypes=(int, object),
        dtype_object_constrain_types=(str,),
        enforce_monotonic_increasing=False,
        enforce_unique=True,
        enforce_not_multi_index=True,
    )
    _DF_CONSTRAINTS_SAMPLES = dfc.IndexConstraints(
        types=(
            pd.RangeIndex,
            pd.Int64Index,
            pd.UInt64Index,
            pd.NumericIndex if "NumericIndex" in dir(pd) else pd.Index,  # Future-proofing.
        ),
        dtypes=(int,),
        dtype_object_constrain_types=None,
        enforce_monotonic_increasing=True,
        enforce_unique=True,
        enforce_not_multi_index=True,
    )
    _DF_CONSTRAINTS_TS_INDEX = dfc.IndexConstraints(
        types=(
            pd.RangeIndex,
            pd.DatetimeIndex,
            pd.Int64Index,
            pd.UInt64Index,
            pd.Float64Index,
            pd.NumericIndex if "NumericIndex" in dir(pd) else pd.Index,  # Future-proofing.
            # NOTE: Other candidates: TimedeltaIndex, PeriodIndex.
            # NOTE: Must match T_TS_Index.
        ),
        dtypes=(int, float, np.datetime64),  # NOTE: Must match TSIndexDtype.
        dtype_object_constrain_types=None,
        enforce_monotonic_increasing=True,
        enforce_unique=True,
        enforce_not_multi_index=True,
    )

_DF_CONSTRAINT_DATAPOINTS = dfc.ElementConstraints(
    dtypes=(float, int, object),  # NOTE: Others candidates: bool, other numeric types (like np.int32).
    dtype_object_constrain_types=(str,),  # NOTE: could expand to broader "categorical" types.
    enforce_homogenous_type_per_column=True,
)


# TODO: Define an ABC?
class TimeSeries(HasFeaturesMixin, HasMissingMixin, ToNumpyMixin, CustomGetItemMixin, Copyable, WrappedDF, SequenceABC):
    _DF_CONSTRAINTS = dfc.Constraints(
        on_index=_DF_CONSTRAINTS_TS_INDEX,
        on_columns=_DF_CONSTRAINTS_FEATURES,
        on_elements=_DF_CONSTRAINT_DATAPOINTS,
    )

    def __init__(
        self,
        data: TInitContainer,
        categorical_features: TCategoricalDef = tuple(),
        missing_indicator: TMissingIndicator = np.nan,
    ) -> None:
        # TODO: More ways to initialize features?
        WrappedDF.__init__(self, data=data)
        HasMissingMixin.__init__(self, missing_indicator=missing_indicator)
        self.set_categorical_def(categorical_features)
        self.validate()

    # --- Sequence Interface ---

    def __len__(self) -> int:
        return len(self._data)

    def _getitem_key_index(
        self,
        key_index: Union[T_TS_IndexDtype, slice],
        single_item: bool,
    ):
        if single_item:
            assert not isinstance(key_index, slice)
            return self._data.loc[key_index, :]
        else:
            return TimeSeries(
                self._data.loc[key_index, :], self._categorical_def, missing_indicator=self.missing_indicator
            )

    def _getitem_key_column(
        self,
        key_column: Union[TFeatureIndex, slice],
    ):
        new_data = self._data.loc[:, key_column]
        if isinstance(new_data, pd.Series):
            new_data = pd.DataFrame(data=new_data, columns=[key_column])
        new_categorical_def = {col: val for col, val in self._categorical_def.items() if col in new_data.columns}
        return TimeSeries(new_data, new_categorical_def, missing_indicator=self.missing_indicator)

    # Override to introduce typehints.
    def __getitem__(  # pylint: disable=useless-super-delegation
        self, key: Union[T_TS_IndexDtype, slice, Tuple[Union[T_TS_IndexDtype, slice], Union[TFeatureIndex, slice]]]
    ) -> Union["TimeSeries", T_TS_StepContainer]:
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[T_TS_StepContainer]:
        for _, row in self._data.iterrows():
            yield row

    def __contains__(
        self, value: Union[T_TS_IndexDtype, T_TS_StepContainer, Tuple[T_TS_IndexDtype, T_TS_StepContainer]]
    ) -> bool:
        # TODO: This should probably be changed, some likely unnecessary functionality.
        if isinstance(value, pd.Series):
            # Check by row content.
            for row in self:
                if (value == row).all():
                    return True
            return False
        elif isinstance(value, tuple):
            # Check by tuple of (index, row content)
            if not (len(value) == 2 and isinstance(value[0], self._index_dtypes) and isinstance(value[1], pd.Series)):
                raise TypeError(
                    f"Comparison value provided as a tuple must be of form: Tuple[{self._index_dtypes}, pd.Series]"
                )
            value_index, value_series = value
            if value_index in self._data.index:
                return (self[value_index] == value_series).all()
            else:
                return False
        else:
            # Check by index.
            if not isinstance(value, self._index_dtypes):
                raise TypeError(
                    f"Comparison value is of inappropriate type: must be a pd.Series or "
                    f"one of {self._index_dtypes}, was {type(value)}"
                )
            return value in self._data.index

    def __reversed__(self) -> Iterator[T_TS_StepContainer]:
        for _, row in self._data[::-1].iterrows():
            yield row

    def index(self, value, start=0, stop=None):
        raise NotImplementedError

    def count(self, value):
        raise NotImplementedError

    # --- Sequence Interface (End) ---

    def _to_numpy_time_series(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        # TODO: Currently assumes that the values are all float, may wish different handling in case there are ints.
        array = self._data.to_numpy()  # Note we make a copy.
        if padding_indicator in array:
            raise ValueError(
                f"Value `{padding_indicator}` found in time series array, choose a different padding indicator"
            )
        n_timesteps, _ = array.shape
        max_len = max_len if max_len is not None else n_timesteps
        if max_len > n_timesteps:
            array = np.pad(
                array, [(0, max_len - n_timesteps), (0, 0)], mode="constant", constant_values=padding_indicator
            )
        elif max_len < n_timesteps:
            array = array[:max_len, :]
        return array

    def is_regular(self) -> Tuple[bool, Optional[TIndexDiff]]:
        return check_index_regular(index=self._data.index)

    @property
    def n_timesteps(self) -> int:
        return len(self)

    def validate(self):
        WrappedDF.validate(self)
        self._init_features()


# Abbreviation: TSS = TimeSeriesSamples
T_TSS_InitContainer = Union[TimeSeries, pd.DataFrame, np.ndarray]


def _make_nested_df(data: Sequence[TimeSeries], index: Sequence[TSamplesIndexDtype]) -> pd.DataFrame:
    nested_df = pd.DataFrame(index=index, columns=data[index[0]].df.columns, dtype=object)
    for c in nested_df.columns:
        for idx, ts in zip(index, data):
            nested_df.at[idx, c] = ts.df[c]
    return nested_df


class TimeSeriesSamples(
    HasFeaturesMixin, HasMissingMixin, ToNumpyMixin, CustomGetItemMixin, Copyable, WrappedDF, SequenceABC
):
    _DF_CONSTRAINTS = dfc.Constraints(
        on_index=_DF_CONSTRAINTS_SAMPLES,
        on_columns=_DF_CONSTRAINTS_FEATURES,
        on_elements=None,
    )

    def __init__(
        self,
        data: Sequence[T_TSS_InitContainer],
        categorical_features: Optional[TCategoricalDef] = None,
        missing_indicator: TMissingIndicator = np.nan,
    ) -> None:
        if len(data) == 0:
            # TODO: Handle this case properly.
            raise ValueError("Must provide at least one time-series sample, cannot be empty")

        _list_data: List[TimeSeries] = list()
        _first_ts = None
        for container in data:
            if isinstance(container, TimeSeries):
                if _first_ts is None:
                    _first_ts = container  # Take features from first TS.
                if categorical_features is None:
                    categorical_features_ready: TCategoricalDef = _first_ts._categorical_def
                else:
                    categorical_features_ready = categorical_features
                container.set_categorical_def(categorical_features_ready)
                _list_data.append(container)
            elif isinstance(container, (pd.DataFrame, np.ndarray)):
                if categorical_features is None:
                    categorical_features_ready = tuple()
                else:
                    categorical_features_ready = categorical_features
                _list_data.append(
                    TimeSeries(
                        data=container,
                        categorical_features=categorical_features_ready,
                        missing_indicator=missing_indicator,
                    )
                )
            else:
                raise TypeError(
                    f"Must provide an iterable of elements like {T_TSS_InitContainer}, " f"found {type(container)}"
                )

        self._data_internal = tuple(_list_data)

        WrappedDF.__init__(self, self._data)
        HasMissingMixin.__init__(self, missing_indicator=missing_indicator)
        self.set_categorical_def(categorical_features_ready)
        # TODO: Check all nested dataframes definitely have same features?

        self.validate()

    @property
    def has_missing(self) -> bool:
        return any(
            [bool(x._data.isnull().sum().sum() > 0) for x in self._data_internal]  # pylint: disable=protected-access
        )

    @property
    def _data_internal(self) -> Tuple[TimeSeries, ...]:
        return self._data_internal_

    @_data_internal.setter
    def _data_internal(self, value: Tuple[TimeSeries, ...]) -> None:
        self._data_internal_ = value
        self._set_data_with_index(value, index=range(len(value)))

    def _set_data_with_index(self, value: Tuple[TimeSeries, ...], index: Sequence[TSamplesIndexDtype]) -> None:
        self._data: pd.DataFrame = _make_nested_df(value, index)

    def _df_for_features(self) -> pd.DataFrame:
        return self._data_internal_[0].df

    # --- Sequence Interface ---

    def _get_single_ts(self, key: TSamplesIndexDtype):
        return self._data_internal[self._data.index.get_loc(key)]

    def __len__(self) -> int:
        return len(self._data_internal)

    def _getitem_key_index(
        self,
        key_index: Union[TSamplesIndexDtype, slice],
        single_item: bool,
    ):
        if single_item:
            assert not isinstance(key_index, slice)
            return self._get_single_ts(key_index)
        else:
            new_keys = [i for i in self._data.loc[key_index, :].index]
            data: Tuple[TimeSeries, ...] = tuple([self._get_single_ts(idx) for idx in new_keys])
            tss = TimeSeriesSamples(data, self._categorical_def, missing_indicator=self.missing_indicator)
            tss._set_data_with_index(data, new_keys)  # pylint: disable=protected-access
            return tss

    def _getitem_key_column(
        self,
        key_column: Union[TFeatureIndex, slice],
    ):
        new_data = [d._data.loc[:, key_column] for d in self._data_internal]  # pylint: disable=protected-access
        if isinstance(new_data[0], pd.Series):
            new_data = [pd.DataFrame(data=d, columns=[key_column]) for d in self._data_internal]
        new_categorical_def = {col: val for col, val in self._categorical_def.items() if col in new_data[0].columns}
        return TimeSeriesSamples(new_data, new_categorical_def, missing_indicator=self.missing_indicator)

    # Override to introduce typehints.
    def __getitem__(  # pylint: disable=useless-super-delegation
        self,
        key: Union[TSamplesIndexDtype, slice, Tuple[Union[TSamplesIndexDtype, slice], Union[TFeatureIndex, slice]]],
    ):
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[TimeSeries]:
        for ts in self._data_internal:
            yield ts

    def __contains__(self, value) -> bool:
        if isinstance(value, int):
            return value in self._data.index
        else:
            raise NotImplementedError(f"Only lookup by sample key is supported in {self.__class__.__name__}")

    def __reversed__(self) -> Iterator[TimeSeries]:
        for ts in reversed(self._data_internal):
            yield ts

    def index(self, value: TimeSeries, start=0, stop=None):
        raise NotImplementedError

    def count(self, value: TimeSeries):
        raise NotImplementedError

    # --- Sequence Interface (End) ---

    def _to_numpy_time_series(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        if max_len is None:
            max_len = max(self.n_timesteps_per_sample)
        arrays = []
        for ts in self._data_internal:
            arrays.append(ts.to_numpy(padding_indicator=padding_indicator, max_len=max_len))
        return np.asarray(arrays)

    def plot(self, n: Optional[int] = None) -> Any:
        for idx, ts in enumerate(self._data_internal):
            print(f"Plotting {idx}-th sample.")
            ts.plot()
            if n is not None and idx + 1 >= n:
                break

    @property
    def df(self) -> pd.DataFrame:
        # Override getter in this class just because that is necessary to override setter.
        return self._data

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        raise AttributeError(f"May not set .df on {self.__class__.__name__}")

    @property
    def n_samples(self) -> int:
        return len(self._data)

    @property
    def n_timesteps_per_sample(self) -> Sequence[int]:
        return [len(ts) for ts in self]

    def is_regular(self) -> Tuple[bool, Optional[TIndexDiff]]:
        diff_list = []
        for ts in self._data_internal:
            is_regular, diff = check_index_regular(index=ts._data.index)  # pylint: disable=protected-access
            diff_list.append(diff)
            if is_regular is False:
                return False, None
        if len(diff_list) == 0:
            return True, None
        else:
            return all([x == diff_list[0] for x in diff_list]), diff_list[0]

    def is_aligned(self) -> bool:
        raise_not_implemented("is_aligned() check on timeseries samples.")

    def validate(self):
        WrappedDF.validate(self)
        self._init_features()

    def to_multi_index_dataframe(self) -> pd.DataFrame:
        # TODO: Copy?
        return pd.concat([x.df for x in self._data_internal], axis=0, keys=self._data.index)

    @property
    def sample_indices(self) -> Sequence[TSamplesIndexDtype]:
        return list(self._data.index)


class StaticSamples(
    HasFeaturesMixin, HasMissingMixin, ToNumpyMixin, CustomGetItemMixin, Copyable, WrappedDF, SequenceABC
):
    _DF_CONSTRAINTS = dfc.Constraints(
        on_index=_DF_CONSTRAINTS_SAMPLES,
        on_columns=_DF_CONSTRAINTS_FEATURES,
        on_elements=_DF_CONSTRAINT_DATAPOINTS,
    )

    def __init__(
        self,
        data: TInitContainer,
        categorical_features: TCategoricalDef = tuple(),
        missing_indicator: TMissingIndicator = np.nan,
    ) -> None:
        WrappedDF.__init__(self, data=data)
        HasMissingMixin.__init__(self, missing_indicator=missing_indicator)
        self.set_categorical_def(categorical_features)
        self.validate()

    # --- Sequence Interface ---

    def __len__(self) -> int:
        return len(self._data)

    def _getitem_key_index(
        self,
        key_index: Union[TSamplesIndexDtype, slice],
        single_item: bool,
    ):
        if single_item:
            assert not isinstance(key_index, slice)
            return self._data.loc[key_index, :]
        else:
            return StaticSamples(
                self._data.loc[key_index, :], self._categorical_def, missing_indicator=self.missing_indicator
            )

    def _getitem_key_column(
        self,
        key_column: Union[TFeatureIndex, slice],
    ):
        new_data = self._data.loc[:, key_column]
        if isinstance(new_data, pd.Series):
            new_data = pd.DataFrame(data=new_data, columns=[key_column])
        new_categorical_def = {col: val for col, val in self._categorical_def.items() if col in new_data.columns}
        return StaticSamples(new_data, new_categorical_def, missing_indicator=self.missing_indicator)

    # Override to introduce typehints.
    def __getitem__(  # pylint: disable=useless-super-delegation
        self,
        key: Union[TSamplesIndexDtype, slice, Tuple[Union[TSamplesIndexDtype, slice], Union[TFeatureIndex, slice]]],
    ) -> Union["StaticSamples", TStaticSamplesContainer]:
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[TStaticSamplesContainer]:
        for _, row in self._data.iterrows():
            yield row

    def __contains__(
        self, value: Union[T_TS_IndexDtype, TStaticSamplesContainer, Tuple[T_TS_IndexDtype, TStaticSamplesContainer]]
    ) -> bool:
        if isinstance(value, int):
            return value in self._data.index
        else:
            raise NotImplementedError(f"Only lookup by sample key is supported in {self.__class__.__name__}")

    def __reversed__(self) -> Iterator[TStaticSamplesContainer]:
        for _, row in self._data[::-1].iterrows():
            yield row

    def index(self, value, start=0, stop=None):
        raise NotImplementedError

    def count(self, value):
        raise NotImplementedError

    # --- Sequence Interface (End) ---

    def _to_numpy_static(self) -> np.ndarray:
        return self._data.to_numpy()  # Note we make a copy.

    @property
    def n_samples(self) -> int:
        return len(self._data)

    def validate(self):
        WrappedDF.validate(self)
        self._init_features()


# Next steps:
# TODO: TimeToEvent - a version of StaticSamples with some constraints.
# TODO: Think whether implementing TemporalDataset class is needed.
