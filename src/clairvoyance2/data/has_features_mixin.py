from typing import Iterable, List, Mapping, Tuple, Union

import pandas as pd

from .feature import (
    CategoricalDtype,
    CategoricalFeatureCreator,
    Feature,
    FeatureCreator,
    FeatureType,
)

TFeatureIndex = Union[int, str]
TCategoricalDef = Union[Iterable[TFeatureIndex], Mapping[TFeatureIndex, Tuple[CategoricalDtype, ...]]]


# TODO: Add a way to automatically infer feature types (using `auto_create_feature_from_data()`).
class HasFeaturesMixin:
    @property
    def _categorical_def(self) -> Mapping[TFeatureIndex, Tuple[CategoricalDtype, ...]]:
        try:
            self._categorical_def_: Mapping[TFeatureIndex, Tuple[CategoricalDtype, ...]]  # Define type.
            self._categorical_def_  # pylint: disable=pointless-statement  # Attempt to retrieve.
        except AttributeError:
            self._categorical_def_ = dict()
        return self._categorical_def_

    @_categorical_def.setter
    def _categorical_def(self, value: Mapping[TFeatureIndex, Tuple[CategoricalDtype, ...]]) -> None:
        self.set_categorical_def(value)

    def _df_for_features(self) -> pd.DataFrame:
        self._data: pd.DataFrame
        return self._data

    @staticmethod
    def _raise_feature_not_found_in_data(feature_name):
        raise ValueError(f"Categorical feature named '{feature_name}' not found in the data")

    def set_categorical_def(self, value: TCategoricalDef) -> None:
        self._categorical_def_ = dict()
        if isinstance(value, Mapping):
            for cat_index, cat_value in value.items():
                if cat_index not in self._df_for_features().columns:
                    self._raise_feature_not_found_in_data(cat_index)
                self._categorical_def_[cat_index] = tuple(cat_value)
        else:
            for cat_index in value:
                if cat_index not in self._data.columns:
                    self._raise_feature_not_found_in_data(cat_index)
                self._categorical_def_[cat_index] = tuple()
        self._features: Mapping[TFeatureIndex, Feature] = self._init_features()  # Regenerate features.

    @property
    def features(self) -> Mapping[TFeatureIndex, Feature]:
        try:
            self._features  # pylint: disable=pointless-statement  # Attempt to retrieve.
        except AttributeError:
            # Create for the first time.
            self._features = self._init_features()
        # NOTE: May wish to have self._features = self._init_features() here...
        return self._features

    def _init_features(self) -> Mapping[TFeatureIndex, Feature]:
        features: List[Feature] = []
        for c in self._df_for_features().columns:
            if c in self._categorical_def:
                # print(c)
                # print(self._df_for_features()[c])
                cat_value = self._categorical_def[c]
                # print(cat_value)
                features.append(
                    CategoricalFeatureCreator.create_feature(
                        data=self._df_for_features()[c],
                        infer_dtype=True,
                        infer_categories=(len(cat_value) == 0),
                        categories=None if len(cat_value) == 0 else cat_value,
                    )
                )
            else:
                features.append(
                    FeatureCreator.create_feature(
                        data=self._df_for_features()[c],
                        feature_type=FeatureType.NUMERIC,
                        infer_dtype=True,
                    )
                )
        return dict(zip(self._df_for_features().columns, features))

    @property
    def feature_types(self) -> Mapping[TFeatureIndex, FeatureType]:
        return {k: v.feature_type for k, v in self._features.items()}

    @property
    def has_categorical_features(self) -> bool:
        return any(f == FeatureType.CATEGORICAL for f in self.feature_types.values())

    @property
    def has_numeric_features(self) -> bool:
        return any(f == FeatureType.NUMERIC for f in self.feature_types.values())

    @property
    def all_categorical_features(self) -> bool:
        return all(f == FeatureType.CATEGORICAL for f in self.feature_types.values())

    @property
    def all_numeric_features(self) -> bool:
        return all(f == FeatureType.NUMERIC for f in self.feature_types.values())

    @property
    def n_features(self) -> int:
        return len(self._features)
