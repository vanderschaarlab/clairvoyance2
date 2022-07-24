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

from . import df_constraints as dfc
from .feature import CategoricalDtype
from .has_features_mixin import HasFeaturesMixin
from .has_missing_mixin import HasMissingMixin, TMissingIndicator

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


# TODO: Unit test.
class WrappedDF:
    _DF_CONSTRAINTS: dfc.Constraints

    def __init__(self, data) -> None:
        if isinstance(data, np.ndarray):
            data = _process_init_from_ndarray(data)
        dfc.ConstraintsChecker(self._DF_CONSTRAINTS).check(data)

        self._data: pd.DataFrame = data

        # Convenience.
        assert (
            self._DF_CONSTRAINTS.on_index is not None
            and self._DF_CONSTRAINTS.on_index.dtypes is not None
            and len(self._DF_CONSTRAINTS.on_index.dtypes) > 0
        )
        self._index_dtypes: Tuple[type, ...] = tuple(self._DF_CONSTRAINTS.on_index.dtypes)

        WrappedDF.validate(self)  # In case derived classes override.

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        self._data = value
        self.validate()

    def __repr__(self) -> str:
        return self._data.__repr__()

    def _repr_html_(self) -> Optional[str]:
        return self._data._repr_html_()  # pylint: disable=protected-access

    def __str__(self) -> str:
        return self._data.__str__()

    def plot(self) -> Any:
        return self._data.plot()

    def validate(self) -> None:
        dfc.ConstraintsChecker(self._DF_CONSTRAINTS).check(self._data)


def _process_init_from_ndarray(array: np.ndarray) -> pd.DataFrame:
    if array.ndim != 2:
        raise ValueError(f"TimeSeries can be constructed from a 2D array only, found {array.ndim} dimensions.")
    return pd.DataFrame(data=array)


def _validate_nonslice_key_type(key, allowed_types) -> None:
    if not isinstance(key, allowed_types):
        raise TypeError(f"Key is of inappropriate type: must be a slice or one of {allowed_types}, was {type(key)}")


# TODO: Define an ABC?
# TODO: Interfaces: MutableSequence, (Mutable)Mapping?
class TimeSeries(HasFeaturesMixin, HasMissingMixin, WrappedDF, SequenceABC):
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
    # TODO: There is quite a bit of shared implementation around here, need to refactor in some way, compare with
    #       implementation in classes StaticSamples, TimeSeriesSamples.

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Union[T_TS_IndexDtype, slice]) -> Union["TimeSeries", T_TS_StepContainer]:
        # TODO: two slices key, similar to np array.
        if not isinstance(key, slice):
            # If key is not a slice, check it's the right type.
            # When the key is a slice this check cannot be sensibly done, so rely on pandas to fail as necessary.
            _validate_nonslice_key_type(key, self._index_dtypes)
            # In this case, looking up one row, so will return the row as TSTimestepType.
            return self._data.loc[key, :]
        else:
            # In this case, looking up 0+ rows via a slice, so will return a TimeSeries.
            return TimeSeries(self._data.loc[key, :], self._categorical_def, self.missing_indicator)

    def __iter__(self) -> Iterator[T_TS_StepContainer]:
        for _, row in self._data.iterrows():
            yield row

    def __contains__(
        self, value: Union[T_TS_IndexDtype, T_TS_StepContainer, Tuple[T_TS_IndexDtype, T_TS_StepContainer]]
    ) -> bool:
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


class TimeSeriesSamples(HasFeaturesMixin, HasMissingMixin, WrappedDF, SequenceABC):
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

    def _get_single_ts(self, key: TSamplesIndexDtype):
        return self._data_internal[self._data.index.get_loc(key)]

    # --- Sequence Interface ---

    def __len__(self) -> int:
        return len(self._data_internal)

    def __getitem__(self, key: Union[TSamplesIndexDtype, slice]) -> Union[TimeSeries, "TimeSeriesSamples"]:
        # TODO: two slices key, similar to np array.
        if not isinstance(key, slice):
            _validate_nonslice_key_type(key, self._index_dtypes)
            return self._get_single_ts(key)
        else:
            new_keys = [i for i in self._data.loc[key, :].index]
            data: Tuple[TimeSeries, ...] = tuple([self._get_single_ts(idx) for idx in new_keys])
            tss = TimeSeriesSamples(
                data, categorical_features=self._categorical_def, missing_indicator=self.missing_indicator
            )
            tss._set_data_with_index(data, new_keys)
            return tss

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

    def validate(self):
        WrappedDF.validate(self)
        self._init_features()


class StaticSamples(HasFeaturesMixin, HasMissingMixin, WrappedDF, SequenceABC):
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

    def __getitem__(self, key: Union[TSamplesIndexDtype, slice]) -> Union["StaticSamples", TStaticSamplesContainer]:
        # TODO: two slices key, similar to np array.
        if not isinstance(key, slice):
            _validate_nonslice_key_type(key, self._index_dtypes)
            return self._data.loc[key, :]
        else:
            return StaticSamples(self._data.loc[key, :], self._categorical_def, self.missing_indicator)

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

    @property
    def n_samples(self) -> int:
        return len(self._data)

    def validate(self):
        WrappedDF.validate(self)
        self._init_features()


TDataset = Union[TimeSeriesSamples, Tuple[TimeSeriesSamples, StaticSamples]]  # NOTE: This will evolve.

# Next steps:
# TODO: TimeToEvent - a version of StaticSamples with some constraints.
# TODO: Think whether implementing TemporalDataset class is needed.
