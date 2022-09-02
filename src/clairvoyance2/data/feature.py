from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Mapping, NoReturn, Optional, Sequence, Tuple

import numpy as np

from ..utils.common import NP_EQUIVALENT_TYPES_MAP, python_type_from_np_pd_dtype
from .constants import (
    T_CategoricalDtype,
    T_CategoricalDtype_AsTuple,
    T_FeatureContainer,
    T_NumericDtype_AsTuple,
)
from .internal_utils import all_items_are_of_types

_DEBUG = False


class FeatureType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()


PD_SERIES_OBJECT_DTYPE_ALLOWED_TYPES = (str,)

FEATURE_DTYPE_MAP: Mapping[FeatureType, Tuple[type, ...]] = {
    FeatureType.NUMERIC: T_NumericDtype_AsTuple,
    FeatureType.CATEGORICAL: T_CategoricalDtype_AsTuple,
    # TODO: Think - should we perhaps allow float for CATEGORICAL?
}


def _infer_dtype(data: T_FeatureContainer) -> type:
    if data.dtype != object:
        return python_type_from_np_pd_dtype(data.dtype)
    else:
        if all([isinstance(x, str) for x in data]):
            return str
        else:
            return object


def _isinstance_compat_np_pd_dtypes(o: Any, _type: type) -> bool:
    return issubclass(python_type_from_np_pd_dtype(type(o)), python_type_from_np_pd_dtype(_type))


def _infer_categories(data: T_FeatureContainer, dtype: type) -> Sequence[T_CategoricalDtype]:
    # TODO: Maybe raise NotImplementedError for unknown dtypes rather than falling back to list(unique)
    # TODO: Maybe trigger a warning when "too many" categories.
    unique = data.unique()
    if unique.dtype in NP_EQUIVALENT_TYPES_MAP:
        assert NP_EQUIVALENT_TYPES_MAP[unique.dtype] == dtype
        result: Sequence[T_CategoricalDtype] = [NP_EQUIVALENT_TYPES_MAP[unique.dtype](x) for x in unique]
    else:
        result = list(unique)
    return result


# TODO: ABC for Feature, then separate out NumericFeature as its own class?


class Feature:
    def __init__(
        self,
        data: T_FeatureContainer,
        feature_type: FeatureType,
        infer_dtype: bool = True,
        dtype: Optional[type] = None,
    ) -> None:
        if feature_type == FeatureType.CATEGORICAL:
            try:
                if self._init_as_categorical_feature:  # type: ignore
                    pass
            except AttributeError as ex:
                # Trying to initialise Feature(feature_type=FeatureType.CATEGORICAL, ...) rather than using
                # CategoricalFeature(), raise exception.
                raise ValueError(
                    "Initialize CategoricalFeature rather than Feature(feature_type=FeatureType.CATEGORICAL, ...)"
                ) from ex

        if infer_dtype:
            if dtype is not None:
                raise ValueError("Must not provide `dtype` when `infer_dtype` is True")
            dtype = _infer_dtype(data)
        else:
            if dtype is None:
                raise ValueError("Must provide `dtype` argument if `infer_dtype` is False")

        if dtype not in FEATURE_DTYPE_MAP[feature_type]:
            raise TypeError(f"Incompatible dtype {dtype} with feature type {feature_type}")

        self.feature_type: FeatureType = feature_type
        self.dtype: type = dtype
        self._data: T_FeatureContainer = data

        self._validate()

    @property
    def numeric_compatible(self) -> bool:
        return self.feature_type == FeatureType.NUMERIC

    @property
    def data(self) -> T_FeatureContainer:
        return self._data

    @data.setter
    def data(self, value: T_FeatureContainer):
        self._data = value
        self._validate()

    @staticmethod
    def _raise_validation_type_error(dtype, data_dtype) -> NoReturn:
        raise TypeError(f"Feature dtype '{dtype}' does not match feature data dtype '{data_dtype}'")

    def _validate(self) -> bool:
        if self.dtype in PD_SERIES_OBJECT_DTYPE_ALLOWED_TYPES:
            if not all_items_are_of_types(self.data, self.dtype):
                self._raise_validation_type_error(self.dtype, self._data.dtype)
        elif self._data.dtype != self.dtype:
            self._raise_validation_type_error(self.dtype, self._data.dtype)
        return True

    def _members_repr(self) -> str:  # pragma: no cover
        return f"feature_type={self.feature_type}, dtype={self.dtype}"

    def _build_repr(self, members_repr: str) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({members_repr}, data={str(object.__repr__(self.data))})"

    def __repr__(self) -> str:
        return self._build_repr(members_repr=self._members_repr())

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Feature):
            return self.feature_type == __o.feature_type and self.dtype == __o.dtype and (self._data == __o._data).all()
        else:
            return False


class CategoricalFeature(Feature):
    def __init__(
        self,
        data: T_FeatureContainer,
        infer_dtype: bool = True,
        dtype: Optional[type] = None,
        infer_categories: bool = True,
        categories: Optional[Sequence[T_CategoricalDtype]] = None,
    ) -> None:
        self._init_as_categorical_feature = True
        super().__init__(data=data, feature_type=FeatureType.CATEGORICAL, infer_dtype=infer_dtype, dtype=dtype)

        if infer_categories:
            if categories is not None:
                raise ValueError("Must not provide `categories` when `infer_categories` is True")
            self.categories = _infer_categories(data, self.dtype)
        else:
            if categories is None:
                raise ValueError("Must provide `categories` argument if `infer_categories` is False")
            self.categories = categories

        self._validate()

    def _process_categories_input(self, value: Sequence[T_CategoricalDtype]) -> Sequence[T_CategoricalDtype]:
        if any(not _isinstance_compat_np_pd_dtypes(x, self.dtype) for x in value):
            raise TypeError(f"The elements of categories must all be of type {self.dtype}")
        return tuple(sorted(set(value)))

    @property
    def numeric_compatible(self) -> bool:
        return self.dtype in (float, int)

    @property
    def categories(self) -> Sequence[T_CategoricalDtype]:
        return self._categories

    @categories.setter
    def categories(self, value: Sequence[T_CategoricalDtype]):
        self._categories: Sequence[T_CategoricalDtype] = self._process_categories_input(value)
        self._validate()

    def update_data_and_categories(self, data: T_FeatureContainer, categories: Sequence[T_CategoricalDtype]) -> None:
        self._categories = self._process_categories_input(categories)
        self._data = data
        self._validate()

    def _validate(self) -> bool:
        super()._validate()
        called_in_parent_init = False
        try:
            self._categories
        except AttributeError:
            called_in_parent_init = True
        if not called_in_parent_init:
            # Arrive here when _validate() called through parent class, no action needed.
            if not np.isin(self._data.unique(), test_elements=self._categories, assume_unique=True).all():
                # NOTE: categories *can* include some "extra" classes not in self._data.
                raise TypeError(
                    f"Not all elements of categorical feature data are of expected categories: {self._categories}"
                )
        return True

    def _members_repr(self) -> str:  # pragma: no cover
        return f"{super()._members_repr()}, categories={self.categories}"

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, CategoricalFeature):
            return super().__eq__(__o) and self.categories == __o.categories
        else:
            return False


# TODO: A factory method that would return appropriate Feature class automatically.
class FeatureCreatorABC(ABC):
    @staticmethod
    @abstractmethod
    def create_feature() -> Feature:  # pragma: no cover
        ...

    @staticmethod
    @abstractmethod
    def auto_create_feature_from_data() -> Feature:  # pragma: no cover
        ...


def _rule_auto_determine_categorical_feature(
    max_categories_frac: float, max_categories_count: int, data: T_FeatureContainer
) -> bool:
    # TODO: This function could perhaps be made smarter.

    if data.dtype == object and all_items_are_of_types(data, str):
        return True

    if data.dtype not in FEATURE_DTYPE_MAP[FeatureType.CATEGORICAL]:
        return False

    if not 0.0 <= max_categories_frac <= 1.0:
        raise ValueError("`max_categories_frac` must be between 0. and 1.")
    if max_categories_count > len(data):
        max_categories_count = len(data)
    max_categories_count_from_frac = round(max_categories_frac * len(data), ndigits=None)
    categories = _infer_categories(data, dtype=data.dtype)

    result = (len(categories) <= max_categories_count_from_frac) and len(categories) < max_categories_count

    if _DEBUG:  # pragma: no cover
        print("max_categories_count:", max_categories_count)
        print(
            "max_categories_count_from_frac:", max_categories_count_from_frac, f"({max_categories_frac} of {len(data)})"
        )
        print("actual categories count:", len(categories))
        print("consider CategoricalFeature?:", result)

    return result


class FeatureCreator(FeatureCreatorABC):
    @staticmethod
    def auto_create_feature_from_data(  # type: ignore # pylint: disable=arguments-differ
        data: T_FeatureContainer, max_categories_frac: float = 0.2, max_categories_count: int = 100
    ) -> Feature:
        if _rule_auto_determine_categorical_feature(max_categories_frac, max_categories_count, data):
            # Assume categorical feature if [<= max_num_categories] and [not each element a different category].
            return CategoricalFeature(data, infer_dtype=True, infer_categories=True)
        else:
            return Feature(data, feature_type=FeatureType.NUMERIC, infer_dtype=True)

    @staticmethod
    def create_feature(*args, **kwargs) -> Feature:  # type: ignore # pylint: disable=arguments-differ
        return Feature(*args, **kwargs)


class CategoricalFeatureCreator(FeatureCreator):
    @staticmethod
    def create_feature(*args, **kwargs) -> CategoricalFeature:  # type: ignore # pylint: disable=arguments-differ
        return CategoricalFeature(*args, **kwargs)
