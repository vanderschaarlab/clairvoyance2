from typing import Collection, NamedTuple, Union

from ..data import Dataset, TimeSeriesSamples
from ..data.dataformat import TFeatureIndex
from ..interface.model import TransformerModel
from ..interface.requirements import DatasetRequirements, Requirements
from ..utils.dev import raise_not_implemented


class _DefaultParams(NamedTuple):
    targets: Union[TFeatureIndex, Collection[TFeatureIndex], slice]


class ExtractTargetsTS(TransformerModel):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(),
        prediction_requirements=None,
    )
    DEFAULT_PARAMS: _DefaultParams = _DefaultParams(targets=tuple())

    def _fit(self, data: Dataset) -> "ExtractTargetsTS":
        # Nothing happens in `fit` here.
        return self

    def _transform(self, data: Dataset) -> Dataset:
        data = data.copy()
        all_features = set(data.temporal_covariates.features.keys())

        temporal_targets: TimeSeriesSamples = data.temporal_covariates[:, self.params.targets]

        if len(temporal_targets.features) > 0:
            extracted_features = set(temporal_targets.features.keys())
            remaining_features = tuple(all_features - extracted_features)

            if len(remaining_features) == 0:
                raise_not_implemented("Selecting all temporal features as targets so that no covariates remains")

            temporal_covariates: TimeSeriesSamples = data.temporal_covariates[:, remaining_features]  # type: ignore
            # TODO: Need to make sure that __getitem__ supports collection of index items and update typehints.

            data.temporal_covariates = temporal_covariates
            data.temporal_targets = temporal_targets

        return data
