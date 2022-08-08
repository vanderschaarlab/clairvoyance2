from typing import Callable, Optional, Protocol, runtime_checkable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..data import Dataset, TimeSeriesSamples
from ..data.utils import split_multi_index_dataframe
from ..interface.model import TParams, TransformerModel
from ..interface.requirements import DatasetRequirements, Requirements


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


# TODO: Convenience classes that combine both static and temporal covariate transformations.

# TODO: Unit test, additional testing.
class SklearnTransformerSC(TransformerModel):
    sklearn_model: SklearnTransformer

    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_static_samples_present=True, requires_all_numeric_features=True
        ),
        prediction_requirements=None,
    )

    check_unknown_params: bool = False

    def __init__(self, sklearn_transformer: SklearnTransformer, params: Optional[TParams] = None) -> None:
        super().__init__(params)
        self.sklearn_model = sklearn_transformer(**self.params)  # type: ignore  # (Ignore 'not callable' mypy error.)

    def _fit(self, data: Dataset) -> "SklearnTransformerSC":
        assert data.static_covariates is not None
        self.sklearn_model.fit(data.static_covariates.df)
        return self

    def _transformation_dispatch(self, data: Dataset, method: Callable) -> Dataset:
        data = data.copy()
        assert data.static_covariates is not None
        data.static_covariates.df[:] = method(data.static_covariates.df)
        return data

    def _transform(self, data: Dataset) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.transform)

    def _inverse_transform(self, data: Dataset) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.inverse_transform)


class SklearnTransformerTC(TransformerModel):
    sklearn_model: SklearnTransformer

    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_static_samples_present=False, requires_all_numeric_features=True
        ),
        prediction_requirements=None,
    )

    check_unknown_params: bool = False

    def __init__(self, sklearn_transformer: SklearnTransformer, params: Optional[TParams] = None) -> None:
        super().__init__(params)
        self.sklearn_model = sklearn_transformer(**self.params)  # type: ignore  # (Ignore 'not callable' mypy error.)

    def _fit(self, data: Dataset) -> "SklearnTransformerTC":
        assert data.temporal_covariates is not None
        self.sklearn_model.fit(data.temporal_covariates.to_multi_index_dataframe().values)
        return self

    def _transformation_dispatch(self, data: Dataset, method: Callable) -> Dataset:
        data = data.copy()
        assert data.temporal_covariates is not None
        multi_index_df = data.temporal_covariates.to_multi_index_dataframe()
        transformed_ndarray = method(multi_index_df.values)
        transformed_multi_index_df = pd.DataFrame(
            data=transformed_ndarray, columns=multi_index_df.columns, index=multi_index_df.index
        )
        transformed_tuple_of_dfs = tuple(split_multi_index_dataframe(transformed_multi_index_df))
        data.temporal_covariates = TimeSeriesSamples(
            data=transformed_tuple_of_dfs,
            categorical_features=data.temporal_covariates._categorical_def,  # pylint: disable=protected-access
            missing_indicator=data.temporal_covariates.missing_indicator,
        )
        return data

    def _transform(self, data: Dataset) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.transform)

    def _inverse_transform(self, data: Dataset) -> Dataset:
        return self._transformation_dispatch(data, method=self.sklearn_model.inverse_transform)


class StandardScalerSC(SklearnTransformerSC):
    DEFAULT_PARAMS = dict(copy=True, with_mean=True, with_std=True)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(StandardScaler, params)


class MinMaxScalerSC(SklearnTransformerSC):
    DEFAULT_PARAMS = dict(feature_range=(0, 1), copy=True, clip=False)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(MinMaxScaler, params)


class StandardScalerTC(SklearnTransformerTC):
    DEFAULT_PARAMS = dict(copy=True, with_mean=True, with_std=True)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(StandardScaler, params)


class MinMaxScalerTC(SklearnTransformerTC):
    DEFAULT_PARAMS = dict(feature_range=(0, 1), copy=True, clip=False)
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        super().__init__(MinMaxScaler, params)
