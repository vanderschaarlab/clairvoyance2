import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from . import df_constraints as dfc

# TODO: May get rid of this Generic[TIndexItem, TColumnItem], not hugely useful.
TIndexItem = TypeVar("TIndexItem")
TColumnItem = TypeVar("TColumnItem")

TIndexIndexers = Union[TIndexItem, slice, Iterable]
TColumnIndexers = Union[TColumnItem, slice, Iterable]


class CustomGetItemMixin(Generic[TIndexItem, TColumnItem]):
    _data: pd.DataFrame

    # Define this in inheriting classes.
    def _getitem_index(self, index_key: TIndexIndexers):
        raise NotImplementedError

    # Define this in inheriting classes.
    def _getitem_column(self, column_key: TColumnIndexers):
        raise NotImplementedError

    # A helper for the straight-forward (_data is simple DF) case.
    def _getitem_index_helper(self, index_key: TIndexIndexers) -> pd.DataFrame:
        new_data: pd.DataFrame = self._data.loc[index_key, :]
        if isinstance(new_data, pd.Series):
            # Handle the case of where single item indexer leads to a pd.Series being returned, by indexing such that
            # a pd.DataFrame is returned.
            new_data = self._data.loc[[index_key], :]
        return new_data

    # A helper for the straight-forward (_data is simple DF) case.
    def _getitem_column_helper(self, column_key: TColumnIndexers) -> pd.DataFrame:
        new_data: pd.DataFrame = self._data.loc[:, column_key]
        if isinstance(new_data, pd.Series):
            # Handle the case of where single item indexer leads to a pd.Series being returned, by indexing such that
            # a pd.DataFrame is returned.
            new_data = self._data.loc[:, [column_key]]
        return new_data

    def _getitem_index_then_column_key(self, index_key, column_key):
        # First create a new temporary object with the new columns:
        temp: "CustomGetItemMixin" = self._getitem_column(column_key=column_key)
        # Then _getitem_key_index() on index:
        return temp._getitem_index(index_key)  # pylint: disable=protected-access

    def __getitem__(self, key: Union[TIndexIndexers, Tuple[TIndexIndexers, TColumnIndexers]]):
        if isinstance(key, tuple) and len(key) == 2:
            index_key, column_key = key
            try:
                return self._getitem_index_then_column_key(index_key, column_key)
            except KeyError:
                # Fall back to getitem on index.
                return self._getitem_index(key)
        else:
            return self._getitem_index(key)


# TODO: Unit test.
class BaseContainer(CustomGetItemMixin[TIndexItem, TColumnItem], Sequence):
    _df_constraints: dfc.Constraints

    def __init__(self, data) -> None:
        if isinstance(data, np.ndarray):
            data = _process_init_from_ndarray(data)
        dfc.ConstraintsChecker(self._df_constraints).check(data)

        self._data: pd.DataFrame = data

        # Convenience.
        assert (
            self._df_constraints.on_index is not None
            and self._df_constraints.on_index.dtypes is not None
            and len(self._df_constraints.on_index.dtypes) > 0
        )
        self._index_dtypes: Tuple[type, ...] = tuple(self._df_constraints.on_index.dtypes)
        if (
            self._df_constraints.on_columns is not None
            and self._df_constraints.on_columns.dtypes is not None
            and len(self._df_constraints.on_columns.dtypes) > 0
        ):
            self._column_dtypes: Optional[Tuple[type, ...]] = tuple(self._df_constraints.on_columns.dtypes)
        else:
            self._column_dtypes = None

        BaseContainer.validate(self)  # In case derived classes override.

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

    @property
    def empty(self) -> bool:
        return self._data.empty

    def validate(self) -> None:
        dfc.ConstraintsChecker(self._df_constraints).check(self._data)

    # --- Sequence Interface ---

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key):  # pylint: disable=useless-super-delegation
        return super().__getitem__(key)

    def __iter__(self) -> Iterator:
        for idx in self._data.index:
            yield self[idx]

    def __contains__(self, value) -> bool:
        return value in self._data.index

    def __reversed__(self) -> Iterator:
        for idx in self._data.index[::-1]:
            yield self[idx]

    def index(self, value, start=0, stop=None):
        raise NotImplementedError

    def count(self, value):
        raise NotImplementedError

    # --- Sequence Interface (End) ---


def _process_init_from_ndarray(array: np.ndarray) -> pd.DataFrame:
    if array.ndim != 2:
        raise ValueError(f"TimeSeries can be constructed from a 2D array only, found {array.ndim} dimensions.")
    return pd.DataFrame(data=array)


class Copyable:
    def copy(self):
        # Default implementation of copy.
        # May wish to have custom version in derived classes.
        return copy.deepcopy(self)


class SupportsNewLike(ABC):
    @staticmethod
    def process_kwargs(kwargs: Dict[str, Any], kwargs_default: Dict[str, Any]) -> Dict[str, Any]:
        kwargs_default.update(kwargs)
        return kwargs_default

    @staticmethod
    @abstractmethod
    def new_like(like: Any, **kwargs) -> "SupportsNewLike":
        ...

    @staticmethod
    @abstractmethod
    def new_empty_like(like: Any, **kwargs) -> "SupportsNewLike":
        ...
