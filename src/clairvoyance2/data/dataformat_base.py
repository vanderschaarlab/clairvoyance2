import copy
from typing import Any, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from . import df_constraints as dfc


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
        if (
            self._DF_CONSTRAINTS.on_columns is not None
            and self._DF_CONSTRAINTS.on_columns.dtypes is not None
            and len(self._DF_CONSTRAINTS.on_columns.dtypes) > 0
        ):
            self._column_dtypes: Optional[Tuple[type, ...]] = tuple(self._DF_CONSTRAINTS.on_columns.dtypes)
        else:
            self._column_dtypes = None

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


TIndexItem = TypeVar("TIndexItem")
TColumnItem = TypeVar("TColumnItem")


# TODO: Unit test.
class CustomGetItemMixin(Generic[TIndexItem, TColumnItem]):
    # Requires the following to be defined (from WrappedDF):
    _index_dtypes: Tuple[type, ...]
    _column_dtypes: Optional[Tuple[type, ...]]

    # Define this in inheriting classes.
    def _getitem_key_index(self, key_index: Union[TIndexItem, slice], single_item: bool):
        raise NotImplementedError

    # Define this in inheriting classes.
    def _getitem_key_column(self, key_column: Union[TColumnItem, slice]):
        raise NotImplementedError

    def _getitem_single_item_key(self, key):
        return self._getitem_key_index(key_index=key, single_item=True)

    def _getitem_single_slice_key(self, key):
        return self._getitem_key_index(key_index=key, single_item=False)

    def _getitem_tuple_of_two_keys(self, key_index, key_column):
        # First create a new temporary object with the new columns:
        temp_ = self._getitem_key_column(key_column=key_column)
        # Then __getitem__() on index:
        return temp_.__getitem__(key_index)

    def __getitem__(self, key: Union[TIndexItem, slice, Tuple[Union[TIndexItem, slice], Union[TColumnItem, slice]]]):
        if isinstance(key, tuple) and len(key) == 1:
            # If a tuple with one item, cast the key as the item.
            key = key[0]
        if not isinstance(key, slice):
            # If key is not a slice...
            if not isinstance(key, tuple):
                # Key is single index item, check it's the right type.
                _validate_nonslice_key_type(key, self._index_dtypes)
                # In this case, looking up one row.
                return self._getitem_single_item_key(key)
            else:
                # Key is a tuple...
                if len(key) != 2:
                    # Ensure it's a tuple of 2.
                    raise KeyError("The key, if tuple, must be a tuple of 2 elements")
                else:
                    key_index, key_column = key
                    # Key is single index item, check it's the right type:
                    if not isinstance(key_index, slice):
                        _validate_nonslice_key_type(key_index, self._index_dtypes)
                    if not isinstance(key_column, slice) and self._column_dtypes is not None:
                        _validate_nonslice_key_type(key_column, self._column_dtypes)
                    return self._getitem_tuple_of_two_keys(key_index, key_column)
        else:
            # When the key is a slice type check cannot be sensibly done, so rely on pandas to fail as necessary.
            # In this case, looking up 0+ rows via a slice, so will return a TimeSeries.
            return self._getitem_single_slice_key(key)


def _process_init_from_ndarray(array: np.ndarray) -> pd.DataFrame:
    if array.ndim != 2:
        raise ValueError(f"TimeSeries can be constructed from a 2D array only, found {array.ndim} dimensions.")
    return pd.DataFrame(data=array)


def _validate_nonslice_key_type(key, allowed_types) -> None:
    if len(allowed_types) > 0:
        if not isinstance(key, allowed_types):
            raise TypeError(f"Key is of inappropriate type: must be a slice or one of {allowed_types}, was {type(key)}")


class Copyable:
    def copy(self):
        # Default implementation of copy.
        # May wish to have custom version in derived classes.
        return copy.deepcopy(self)
