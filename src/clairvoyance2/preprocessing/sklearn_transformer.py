from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from typing_extensions import Protocol, runtime_checkable

from ..data import Dataset, TimeSeriesSamples
from ..data.utils import cast_time_series_samples_feature_names_to_str
from ..interface.model import TParams, TransformerModel
from ..interface.requirements import DatasetRequirements, DataValueOpts, Requirements
from ..utils.common import split_multi_index_dataframe
from ..utils.dev import NEEDED, raise_not_implemented


@runtime_checkable
class SklearnTransformer(Protocol):
    # NOTE: Purely for typing.

    def fit(self, *args, **kwargs):
        ...

    def transform(self, *args, **kwargs):
        ...

    def fit_transform(self, *args, **kwargs):
        ...

    def inverse_transform(self, *args, **kwargs):
        ...


# TODO: Convenience classes that combine both static and temporal covariate transformations?


class SklearnTransformerForClairvoyance(TransformerModel):
    sklearn_model: SklearnTransformer
    check_unknown_params: bool = False
    non_sklearn_params: Dict[str, Any] = {"apply_to": NEEDED}

    def get_container(self, data: Dataset):
        container_map = {
            "temporal_covariates": data.temporal_covariates,
            "static_covariates": data.static_covariates,
            "temporal_targets": data.temporal_targets,
            "temporal_treatments": data.temporal_treatments,
        }
        return container_map[self.params.apply_to]

    def set_container(self, data: Dataset, value):
        if self.params.apply_to == "temporal_covariates":
            data.temporal_covariates = value
        elif self.params.apply_to == "static_covariates":
            data.static_covariates = value
        elif self.params.apply_to == "temporal_targets":
            data.temporal_targets = value
        elif self.params.apply_to == "temporal_treatments":
            data.temporal_treatments = value
        else:
            raise ValueError(f"Unknown data container {self.params.apply_to}")

    def get_sklearn_params(self):
        sklearn_params = dict()
        for k, v in self.params.items():
            if k not in self.non_sklearn_params:
                sklearn_params[k] = v
        return sklearn_params

    def __init__(self, sklearn_transformer: SklearnTransformer, params: Optional[TParams] = None) -> None:
        super().__init__(params)
        for k in self.non_sklearn_params:
            if params is not None and k in params:
                self.params[k] = params[k]
            else:
                self.params[k] = self.non_sklearn_params[k]
        self.sklearn_model = sklearn_transformer(**self.get_sklearn_params())  # type: ignore


# TODO: Unit test, additional testing.
class StaticDataSklearnTransformer(SklearnTransformerForClairvoyance):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_static_covariates_present=True,
            static_covariates_value_type=DataValueOpts.NUMERIC,
        ),
        prediction_requirements=None,
    )
    non_sklearn_params = {"apply_to": "static_covariates"}

    def _fit(self, data: Dataset, **kwargs) -> "StaticDataSklearnTransformer":
        assert self.get_container(data) is not None
        self.sklearn_model.fit(self.get_container(data).df)
        return self

    def _transformation_dispatch(self, data: Dataset, method: Callable) -> Dataset:
        data = data.copy()
        assert self.get_container(data) is not None
        self.get_container(data).df[:] = method(self.get_container(data).df)
        return data

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.transform)

    def _inverse_transform(self, data: Dataset, **kwargs) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.inverse_transform)


class TemporalDataSklearnTransformer(SklearnTransformerForClairvoyance):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_static_covariates_present=False,
            temporal_covariates_value_type=DataValueOpts.NUMERIC,
            temporal_targets_value_type=DataValueOpts.NUMERIC,
            temporal_treatments_value_type=DataValueOpts.NUMERIC,
            requires_all_temporal_containers_shares_index=False,
        ),
        prediction_requirements=None,
    )
    non_sklearn_params = {"apply_to": "temporal_covariates"}

    def _fit(self, data: Dataset, **kwargs) -> "TemporalDataSklearnTransformer":
        assert self.get_container(data) is not None
        self.sklearn_model.fit(self.get_container(data).to_multi_index_dataframe().values)
        return self

    def _transformation_dispatch(self, data: Dataset, method: Callable) -> Dataset:
        data = data.copy()
        assert self.get_container(data) is not None
        multi_index_df = self.get_container(data).to_multi_index_dataframe()
        transformed_ndarray = method(multi_index_df.values)
        transformed_multi_index_df = pd.DataFrame(
            data=transformed_ndarray, columns=multi_index_df.columns, index=multi_index_df.index
        )
        transformed_tuple_of_dfs = tuple(split_multi_index_dataframe(transformed_multi_index_df))
        new_value = TimeSeriesSamples.new_like(like=self.get_container(data), data=transformed_tuple_of_dfs)
        self.set_container(data, value=new_value)
        return data

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.transform)

    def _inverse_transform(self, data: Dataset, **kwargs) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.inverse_transform)


class StaticDataStandardScaler(StaticDataSklearnTransformer):
    DEFAULT_PARAMS = dict(apply_to="static_covariates", copy=True, with_mean=True, with_std=True)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(StandardScaler, params)


class StaticDataMinMaxScaler(StaticDataSklearnTransformer):
    DEFAULT_PARAMS = dict(apply_to="static_covariates", feature_range=(0, 1), copy=True, clip=False)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(MinMaxScaler, params)


class TemporalDataStandardScaler(TemporalDataSklearnTransformer):
    DEFAULT_PARAMS = dict(apply_to="temporal_covariates", copy=True, with_mean=True, with_std=True)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(StandardScaler, params)


class TemporalDataMinMaxScaler(TemporalDataSklearnTransformer):
    DEFAULT_PARAMS = dict(apply_to="temporal_covariates", feature_range=(0, 1), copy=True, clip=False)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(MinMaxScaler, params)


# TODO: Invert non-string column names back to the original dtype.
# TODO: Needs testing. (see 64_our-onehot.ipynb)
class TemporalDataOneHotEncoder(TemporalDataSklearnTransformer):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_static_covariates_present=False,
            temporal_covariates_value_type=DataValueOpts.NUMERIC,
            temporal_targets_value_type=DataValueOpts.NUMERIC,
            temporal_treatments_value_type=DataValueOpts.NUMERIC,
        ),
        prediction_requirements=None,
    )

    check_unknown_params: bool = False
    non_sklearn_params: Dict[str, Any] = {
        "apply_to": "temporal_covariates",
        "feature_name": NEEDED,
        "prefix": "OneHot",
        "prefix_sep": "_",
    }

    def __init__(self, params: Optional[TParams] = None) -> None:
        self.sklearn_model: OneHotEncoder
        super().__init__(OneHotEncoder, params)
        self.sklearn_model.set_params(sparse=False)  # NOTE: Override.
        if self.params.feature_name is None:
            raise ValueError("Must specify `feature_name` to apply one-hot encoder to")
        self.params.feature_name = str(self.params.feature_name)
        self.sklearn_categories: List = []
        self.sklearn_drop_idx: List = []
        self.original_column_names: List = []
        self.unchanged_column_names: List = []
        self.new_transformed_column_names: List[str] = []
        self.all_transformed_column_names: List[str] = []

    def _get_data_to_transform(self, data: Dataset):
        cast_time_series_samples_feature_names_to_str(self.get_container(data))
        original_data_multiindex_df = self.get_container(data).to_multi_index_dataframe()
        to_transform = original_data_multiindex_df.loc[:, [self.params.feature_name]]
        self.original_column_names = list(original_data_multiindex_df.columns)
        return to_transform, original_data_multiindex_df

    def _get_inverted_data_to_transform(self, data: Dataset):
        inverted_data_multiindex_df = self.get_container(data).to_multi_index_dataframe()
        to_transform = inverted_data_multiindex_df.loc[:, self.new_transformed_column_names]
        return to_transform, inverted_data_multiindex_df

    def _fit(self, data: Dataset, **kwargs) -> "TemporalDataSklearnTransformer":
        assert self.get_container(data) is not None
        to_transform, _ = self._get_data_to_transform(data)
        self.sklearn_model.fit(to_transform)
        self.sklearn_categories = self.sklearn_model.categories_[0]
        self.sklearn_drop_idx = self.sklearn_model.drop_idx_[0] if self.sklearn_model.drop_idx_ is not None else None
        return self

    def _transform(self, data: Dataset, **kwargs) -> Dataset:
        data = data.copy()
        assert self.get_container(data) is not None

        to_transform, original_data_multiindex_df = self._get_data_to_transform(data)
        transformed_ndarray = self.sklearn_model.transform(to_transform)

        if transformed_ndarray.shape[1] != len(self.sklearn_categories):
            raise_not_implemented(
                "Case where Sklearn OneHotEncoder produces a different number of columns than categories"
            )

        self.new_transformed_column_names = [
            f"{self.params.prefix}{self.params.prefix_sep}{self.params.feature_name}{self.params.prefix_sep}{cat}"
            for cat in self.sklearn_categories
        ]

        idx = list(self.original_column_names).index(self.params.feature_name)
        pre = self.original_column_names[:idx]
        post = self.original_column_names[idx + 1 :]
        self.unchanged_column_names = list(pre) + list(post)
        self.all_transformed_column_names = list(pre) + self.new_transformed_column_names + list(post)

        df = pd.DataFrame(
            data=original_data_multiindex_df.loc[:, self.unchanged_column_names],
            columns=self.all_transformed_column_names,
            index=original_data_multiindex_df.index,
        )
        df.loc[:, self.new_transformed_column_names] = transformed_ndarray

        transformed_tuple_of_dfs = tuple(split_multi_index_dataframe(df))
        new_ts = TimeSeriesSamples.new_like(
            like=self.get_container(data),
            data=transformed_tuple_of_dfs,
        )
        self.set_container(data, value=new_ts)
        return data

    def _inverse_transform(self, data: Dataset, **kwargs) -> Dataset:
        data = data.copy()
        assert self.get_container(data) is not None

        to_transform, inverted_data_multiindex_df = self._get_inverted_data_to_transform(data)
        inverse_transformed_ndarray = self.sklearn_model.inverse_transform(to_transform)

        df = pd.DataFrame(
            data=inverted_data_multiindex_df.loc[:, self.unchanged_column_names],
            columns=self.original_column_names,
            index=inverted_data_multiindex_df.index,
        )
        df.loc[:, [self.params.feature_name]] = inverse_transformed_ndarray

        inverse_transformed_tuple_of_dfs = tuple(split_multi_index_dataframe(df))
        new_ts = TimeSeriesSamples.new_like(
            like=self.get_container(data),
            data=inverse_transformed_tuple_of_dfs,
        )
        self.set_container(data, value=new_ts)
        return data
