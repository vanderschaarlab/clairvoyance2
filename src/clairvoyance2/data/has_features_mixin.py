from typing import List, Mapping, Optional, Sequence

import pandas as pd

from .constants import T_CategoricalDef, T_CategoricalDef_Arg, T_FeatureIndexDtype
from .feature import CategoricalFeatureCreator, Feature, FeatureCreator, FeatureType


# TODO: Add a way to automatically infer feature types (using `auto_create_feature_from_data()`).
class HasFeaturesMixin:
    @property
    def categorical_def(self) -> T_CategoricalDef:
        try:
            self._categorical_def: T_CategoricalDef  # Define type.
            _ = self._categorical_def  # Attempt to retrieve.
        except AttributeError:
            self._categorical_def = dict()
        return self._categorical_def

    @categorical_def.setter
    def categorical_def(self, value: T_CategoricalDef) -> None:
        self.set_categorical_def(value)

    def _df_for_features(self) -> pd.DataFrame:
        self._data: pd.DataFrame
        return self._data

    @staticmethod
    def _raise_feature_not_found_in_data(feature_name):
        raise ValueError(f"Categorical feature named '{feature_name}' not found in the data")

    def set_categorical_def(self, value: T_CategoricalDef_Arg) -> None:
        self._set_categorical_def(value, check=True)

    def _set_categorical_def(self, value: T_CategoricalDef_Arg, check: bool = True) -> None:
        self._categorical_def = dict()
        if isinstance(value, Mapping):
            for cat_index, cat_value in value.items():
                if check:
                    if cat_index not in self._df_for_features().columns:
                        self._raise_feature_not_found_in_data(cat_index)
                self._categorical_def[cat_index] = tuple(cat_value)
        else:
            for cat_index in value:
                if check:
                    if cat_index not in self._data.columns:
                        self._raise_feature_not_found_in_data(cat_index)
                self._categorical_def[cat_index] = tuple()
        self._features: Mapping[T_FeatureIndexDtype, Feature] = self._init_features()  # Regenerate features.

    @property
    def features(self) -> Mapping[T_FeatureIndexDtype, Feature]:
        try:
            _ = self._features  # Attempt to retrieve.
        except AttributeError:
            # Create for the first time.
            self._features = self._init_features()
        # NOTE: May wish to have self.refresh_features() here...
        return self._features

    def _init_features(self) -> Mapping[T_FeatureIndexDtype, Feature]:
        features: List[Feature] = []
        for c in self._df_for_features().columns:
            if c in self.categorical_def:
                cat_value = self.categorical_def[c]
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

    def refresh_features(self, categorical_def: Optional[T_CategoricalDef_Arg] = None):
        if categorical_def is not None:
            self.set_categorical_def(categorical_def)
        self._features = self._init_features()

    def set_df_and_features(self, df: pd.DataFrame, categorical_def: T_CategoricalDef_Arg) -> None:
        self.df: pd.DataFrame
        self._set_categorical_def(categorical_def, check=False)
        self.df = df  # As expected, will fail on TimeSeriesSamples, where not allowed to set `df` directly.
        self._set_categorical_def(self.categorical_def, check=True)

    @property
    def feature_types(self) -> Mapping[T_FeatureIndexDtype, FeatureType]:
        return {k: v.feature_type for k, v in self._features.items()}

    @property
    def feature_names(self) -> Sequence[T_FeatureIndexDtype]:
        return [k for k in self._features.keys()]

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
    def all_numeric_compatible_features(self) -> bool:
        return all(f.numeric_compatible for f in self.features.values())

    @property
    def n_features(self) -> int:
        return len(self._features)
