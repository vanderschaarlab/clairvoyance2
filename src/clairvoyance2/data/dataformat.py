import warnings
from typing import Any, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from ..utils.dev import raise_not_implemented
from . import df_constraints as dfc
from .constants import (
    DEFAULT_PADDING_INDICATOR,
    T_CategoricalDef_Arg,
    T_ContainerInitializable,
    T_ContainerInitializable_AsTuple,
    T_FeatureIndexClass_AsTuple,
    T_FeatureIndexDtype,
    T_FeatureIndexDtype_AsTuple,
    T_SampleIndex_Compatible,
    T_SampleIndexClass,
    T_SampleIndexClass_AsTuple,
    T_SamplesIndexDtype,
    T_SamplesIndexDtype_AsTuple,
    T_TSIndexClass_AsTuple,
    T_TSIndexDtype,
    T_TSIndexDtype_AsTuple,
)
from .dataformat_base import (
    BaseContainer,
    Copyable,
    SupportsNewLike,
    _process_init_from_ndarray,
)
from .has_features_mixin import HasFeaturesMixin
from .has_missing_mixin import HasMissingMixin, TMissingIndicator
from .internal_utils import TIndexDiff, check_index_regular
from .to_tensor_like_mixin import ToTensorLikeMixin
from .update_from import UpdateFromArrayExtension, UpdateFromSequenceOfArraysExtension

# pylint: disable=useless-super-delegation
# ^ In some methods, "useless" super delegation used to add type hints.

# pylint: disable=abstract-method
# ^ Pylint detects some expected NotImplementedError methods as abstract.


with warnings.catch_warnings():
    # This is to suppress (expected) FutureWarnings for index types like pd.Int64Index.
    warnings.filterwarnings("ignore", message=r".*Use pandas.Index.*", category=FutureWarning)

    _DF_CONSTRAINTS_FEATURES = dfc.IndexConstraints(
        types=T_FeatureIndexClass_AsTuple,
        dtypes=dfc.cast_to_index_constraints_dtypes(T_FeatureIndexDtype_AsTuple),
        dtype_object_constrain_types=(str,),
        enforce_monotonic_increasing=False,
        enforce_unique=True,
        enforce_not_multi_index=True,
    )
    _DF_CONSTRAINTS_SAMPLES = dfc.IndexConstraints(
        types=T_SampleIndexClass_AsTuple,
        dtypes=dfc.cast_to_index_constraints_dtypes(T_SamplesIndexDtype_AsTuple),
        dtype_object_constrain_types=None,
        enforce_monotonic_increasing=True,
        enforce_unique=True,
        enforce_not_multi_index=True,
    )
    _DF_CONSTRAINTS_TS_INDEX = dfc.IndexConstraints(
        types=T_TSIndexClass_AsTuple,
        dtypes=dfc.cast_to_index_constraints_dtypes(T_TSIndexDtype_AsTuple),
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


class TimeSeries(
    UpdateFromArrayExtension,
    HasFeaturesMixin,
    HasMissingMixin,
    ToTensorLikeMixin,
    Copyable,
    SupportsNewLike,
    BaseContainer[T_TSIndexDtype, T_FeatureIndexDtype],
):
    _df_constraints = dfc.Constraints(
        on_index=_DF_CONSTRAINTS_TS_INDEX,
        on_columns=_DF_CONSTRAINTS_FEATURES,
        on_elements=_DF_CONSTRAINT_DATAPOINTS,
    )

    def __init__(
        self,
        data: T_ContainerInitializable,
        categorical_features: T_CategoricalDef_Arg = tuple(),
        missing_indicator: TMissingIndicator = np.nan,
    ) -> None:
        # TODO: More ways to initialize features?
        BaseContainer.__init__(self, data=data)
        HasMissingMixin.__init__(self, missing_indicator=missing_indicator)
        self.set_categorical_def(categorical_features)
        self.validate()

    # --- Sequence Interface ---

    def _getitem_index(self, index_key):
        new_data = self._getitem_index_helper(index_key)
        return self.new_like(like=self, data=new_data)

    def _getitem_column(self, column_key):
        new_data, new_categorical_def = self._getitem_column_helper(column_key)
        return self.new_like(like=self, data=new_data, categorical_features=new_categorical_def)

    def __getitem__(self, key) -> "TimeSeries":
        return super().__getitem__(key)

    # --- Sequence Interface (End) ---

    def apply_time_indexing(self, key, inplace: bool = False) -> Union["TimeSeries", None]:
        # TODO: Experimental / not finalized.
        if not inplace:
            return self.new_like(like=self, data=self._data.loc[key, :])
        else:
            self._data = self._data.loc[key, :]
            return None

    @staticmethod
    def _to_numpy_helper(array: np.ndarray, padding_indicator: float, max_len: Optional[int] = None) -> np.ndarray:
        if padding_indicator in array:
            raise ValueError(
                f"Value `{padding_indicator}` found in time series array, choose a different padding indicator"
            )
        n_timesteps, *_ = array.shape
        max_len = max_len if max_len is not None else n_timesteps
        if max_len > n_timesteps:
            if array.ndim == 1:
                pad_shape: Any = [0, max_len - n_timesteps]
            else:
                pad_shape = [(0, max_len - n_timesteps), (0, 0)]
            array = np.pad(array, pad_shape, mode="constant", constant_values=padding_indicator)
        elif max_len < n_timesteps:
            array = array[:max_len]
        if array.ndim == 1:
            array = np.expand_dims(array, axis=-1)
        return array

    def _to_numpy_time_series(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        # TODO: Currently assumes that the values are all float, may wish different handling in case there are ints.
        array = self._data.to_numpy()  # Note we make a copy.
        return self._to_numpy_helper(array, padding_indicator, max_len)

    def _to_numpy_time_index(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        array = self.time_index.values.copy()
        return self._to_numpy_helper(array, padding_indicator, max_len)

    @property
    def time_index(self):
        return self._data.index

    def is_regular(self) -> Tuple[bool, Optional[TIndexDiff]]:
        return check_index_regular(index=self.time_index)

    @property
    def n_timesteps(self) -> int:
        return len(self)

    def validate(self):
        BaseContainer.validate(self)
        self._init_features()

    @staticmethod
    def new_like(like: "TimeSeries", **kwargs) -> "TimeSeries":
        kwargs = SupportsNewLike.process_kwargs(
            kwargs, dict(categorical_features=like.categorical_def, missing_indicator=like.missing_indicator)
        )
        return TimeSeries(**kwargs)  # type: ignore  # Mypy complains about kwargs but it's fine.


# Abbreviation: TSS = TimeSeriesSamples
T_TSS_ContainerInitializable = Union[TimeSeries, T_ContainerInitializable]


class TimeSeriesSamples(
    UpdateFromSequenceOfArraysExtension,
    HasFeaturesMixin,
    HasMissingMixin,
    ToTensorLikeMixin,
    Copyable,
    SupportsNewLike,
    BaseContainer[T_SamplesIndexDtype, T_FeatureIndexDtype],
):
    _df_constraints = dfc.Constraints(
        on_index=_DF_CONSTRAINTS_SAMPLES,
        on_columns=_DF_CONSTRAINTS_FEATURES,
        on_elements=None,
    )

    def __init__(
        self,
        data: Sequence[T_TSS_ContainerInitializable],
        sample_indices: Optional[T_SampleIndex_Compatible] = None,
        categorical_features: Optional[T_CategoricalDef_Arg] = None,
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
                    categorical_features_ready: T_CategoricalDef_Arg = _first_ts.categorical_def
                else:
                    categorical_features_ready = categorical_features
                container.set_categorical_def(categorical_features_ready)
                _list_data.append(container)
            elif isinstance(container, T_ContainerInitializable_AsTuple):
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
                    f"Must provide a sequence of elements like {T_TSS_ContainerInitializable}, found {type(container)}"
                )

        if sample_indices is None:
            sample_indices = list(range(len(_list_data)))
        if len(sample_indices) != len(_list_data):
            raise ValueError(
                f"Length of `sample_indices` provided to {self.__class__.__name__} constructor "
                "did not match the length of `data` (number of samples)"
            )
        self._set_data(_list_data, sample_indices)

        BaseContainer.__init__(self, self._data)
        HasMissingMixin.__init__(self, missing_indicator=missing_indicator)
        self.set_categorical_def(categorical_features_ready)
        # TODO: Check all nested dataframes definitely have same features?

        self.validate()

    @staticmethod
    def _make_nested_df(data: Sequence[TimeSeries], index: T_SampleIndex_Compatible) -> pd.DataFrame:
        assert len(data) > 0
        nested_df = pd.DataFrame(index=index, columns=data[0].df.columns, dtype=object)
        for c in nested_df.columns:
            for idx, ts in zip(index, data):
                nested_df.at[idx, c] = ts.df[c]
        return nested_df

    @property
    def has_missing(self) -> bool:
        return any([bool(ts.df.isnull().sum().sum() > 0) for ts in self])

    def _set_data(self, value: Sequence[TimeSeries], index: T_SampleIndex_Compatible) -> None:
        self._internal: Sequence[TimeSeries] = value
        self._df_tracker: List[int] = [id(x.df) for x in self._internal]
        self._data: pd.DataFrame = self._make_nested_df(value, index)

    def _refresh_data(self) -> None:
        for idx, ts in enumerate(self._internal):
            # Check if any of the .df on TimeSeries objects in ._internal have been reassigned.
            if id(ts.df) != self._df_tracker[idx]:
                # If so, repopulate the Series inside the appropriate row of self._data DataFrame.
                self._df_tracker[idx] = id(ts.df)  # Update the tracker itself with the new id.
                for c in self._data.columns:
                    self._data.at[idx, c] = ts.df[c]

    def _df_for_features(self) -> pd.DataFrame:
        return self._internal[0].df

    # --- Sequence Interface ---

    def apply_time_indexing(self, key, inplace: bool = False) -> Union["TimeSeriesSamples", None]:
        # TODO: Experimental / not finalized.
        if not inplace:
            ts_list = []
            for ts in self:
                ts_list.append(ts.apply_time_indexing(key, inplace=False))  # pylint: disable=protected-access
            return self.new_like(like=self, data=ts_list)
        else:
            for ts in self:
                ts_list.append(ts.apply_time_indexing(key, inplace=True))  # pylint: disable=protected-access
            return None

    def _get_single_ts(self, key: T_SamplesIndexDtype):
        return self._internal[self.sample_index.get_loc(key)]

    def __len__(self) -> int:
        return len(self._internal)

    def _getitem_index(self, index_key) -> Union["TimeSeriesSamples", TimeSeries]:
        selection: pd.DataFrame = self._data.loc[index_key, :]
        if isinstance(selection, pd.Series):
            assert not isinstance(index_key, slice)
            return self._get_single_ts(index_key)
        new_keys = [i for i in selection.index]
        data: Tuple[TimeSeries, ...] = tuple([self._get_single_ts(idx) for idx in new_keys])
        return self.new_like(like=self, data=data, sample_indices=new_keys)

    def _getitem_column(self, column_key) -> "TimeSeriesSamples":
        new_data = [ts.df.loc[:, column_key] for ts in self]
        if isinstance(new_data[0], pd.Series):
            new_data = [pd.DataFrame(data=ts.df.loc[:, column_key], columns=[column_key]) for ts in self]
        new_categorical_def = {col: val for col, val in self.categorical_def.items() if col in new_data[0].columns}
        return self.new_like(like=self, data=new_data, categorical_features=new_categorical_def)

    def __getitem__(self, key) -> Union["TimeSeriesSamples", TimeSeries]:
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[TimeSeries]:
        for ts in self._internal:
            yield ts

    def __reversed__(self) -> Iterator[TimeSeries]:
        for ts in reversed(self._internal):
            yield ts

    # --- Sequence Interface (End) ---

    def _to_numpy_time_series(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        if max_len is None:
            max_len = max(self.n_timesteps_per_sample)
        arrays = []
        for ts in self:
            arrays.append(ts.to_numpy(padding_indicator=padding_indicator, max_len=max_len))
        return np.asarray(arrays)

    def _to_numpy_time_index(
        self, padding_indicator: float = DEFAULT_PADDING_INDICATOR, max_len: Optional[int] = None
    ) -> np.ndarray:
        if max_len is None:
            max_len = max(self.n_timesteps_per_sample)
        arrays = []
        for ts in self:
            arrays.append(ts.to_numpy_time_index(padding_indicator=padding_indicator, max_len=max_len))
        return np.asarray(arrays)

    def plot(self, n: Optional[int] = None) -> Any:
        for idx, ts in enumerate(self):
            print(f"Plotting {idx}-th sample.")
            ts.plot()
            if n is not None and idx + 1 >= n:
                break

    @property
    def empty(self) -> bool:
        return False

    @property
    def df(self) -> pd.DataFrame:
        self._refresh_data()
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
        for ts in self:
            is_regular, diff = check_index_regular(index=ts.time_index)
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
        BaseContainer.validate(self)
        self._init_features()

    def to_multi_index_dataframe(self) -> pd.DataFrame:
        return pd.concat([ts.df for ts in self], axis=0, keys=self.sample_index)

    @property
    def sample_index(self) -> T_SampleIndexClass:
        return self._data.index

    @property
    def sample_indices(self) -> Sequence[T_SamplesIndexDtype]:
        return list(self.sample_index)

    @staticmethod
    def new_like(like: "TimeSeriesSamples", **kwargs) -> "TimeSeriesSamples":
        kwargs = SupportsNewLike.process_kwargs(
            kwargs,
            dict(
                sample_indices=like.sample_indices,
                categorical_features=like.categorical_def,
                missing_indicator=like.missing_indicator,
            ),
        )
        return TimeSeriesSamples(**kwargs)  # type: ignore  # Mypy complains about kwargs but it's fine.


class StaticSamples(
    HasFeaturesMixin,
    HasMissingMixin,
    ToTensorLikeMixin,
    Copyable,
    SupportsNewLike,
    BaseContainer[T_SamplesIndexDtype, T_FeatureIndexDtype],
):
    _df_constraints = dfc.Constraints(
        on_index=_DF_CONSTRAINTS_SAMPLES,
        on_columns=_DF_CONSTRAINTS_FEATURES,
        on_elements=_DF_CONSTRAINT_DATAPOINTS,
    )

    def __init__(
        self,
        data: T_ContainerInitializable,
        sample_indices: Optional[T_SampleIndex_Compatible] = None,
        categorical_features: T_CategoricalDef_Arg = tuple(),
        missing_indicator: TMissingIndicator = np.nan,
    ) -> None:
        if sample_indices is not None:
            if len(sample_indices) != len(data):
                raise ValueError(
                    f"Length of `sample_indices` provided to {self.__class__.__name__} constructor "
                    "did not match the length of `data` (number of samples)"
                )
            if isinstance(data, np.ndarray):
                data = _process_init_from_ndarray(data)
            data.set_index(pd.Index(sample_indices), inplace=True)

        BaseContainer.__init__(self, data=data)
        HasMissingMixin.__init__(self, missing_indicator=missing_indicator)
        self.set_categorical_def(categorical_features)
        self.validate()

    # --- Sequence Interface ---

    def _getitem_index(self, index_key):
        new_data = self._getitem_index_helper(index_key)
        return self.new_like(like=self, data=new_data, sample_indices=new_data.index)

    def _getitem_column(self, column_key):
        new_data, new_categorical_def = self._getitem_column_helper(column_key)
        return self.new_like(like=self, data=new_data, categorical_features=new_categorical_def)

    def __getitem__(self, key) -> "StaticSamples":
        return super().__getitem__(key)

    # --- Sequence Interface (End) ---

    def _to_numpy_static(self) -> np.ndarray:
        return self._data.to_numpy()  # Note we make a copy.

    @property
    def n_samples(self) -> int:
        return len(self._data)

    def validate(self):
        BaseContainer.validate(self)
        self._init_features()

    @property
    def sample_index(self) -> T_SampleIndexClass:
        return self._data.index

    @property
    def sample_indices(self) -> Sequence[T_SamplesIndexDtype]:
        return list(self.sample_index)

    @staticmethod
    def new_like(like: "StaticSamples", **kwargs) -> "StaticSamples":
        kwargs = SupportsNewLike.process_kwargs(
            kwargs,
            dict(
                sample_indices=like.sample_indices,
                categorical_features=like.categorical_def,
                missing_indicator=like.missing_indicator,
            ),
        )
        return StaticSamples(**kwargs)  # type: ignore  # Mypy complains about kwargs but it's fine.


THasTimeIndex = TypeVar("THasTimeIndex", TimeSeries, TimeSeriesSamples)


def time_index_equal(a: THasTimeIndex, b: THasTimeIndex) -> bool:
    if isinstance(a, TimeSeries):
        return _time_index_equal__time_series(a, b)
    elif isinstance(a, TimeSeriesSamples):
        return _time_index_equal__time_series_samples(a, b)
    else:
        raise TypeError(f"Unexpected type encountered: {type(a)}")


def _time_index_equal__time_series(a: TimeSeries, b: TimeSeries) -> bool:
    if len(a.time_index) != len(b.time_index):
        return False
    else:
        return (a.time_index == b.time_index).all()


def _time_index_equal__time_series_samples(a: TimeSeriesSamples, b: TimeSeriesSamples) -> bool:
    if a.sample_indices != b.sample_indices:
        return False
    return all(_time_index_equal__time_series(a_, b_) for a_, b_ in zip(a, b))


# Next steps:
# TODO: TimeToEvent - a version of StaticSamples with some constraints.
# TODO: Think whether implementing TemporalDataset class is needed.
