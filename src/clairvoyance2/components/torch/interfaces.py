"""
Useful reusable interfaces for PyTorch models.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...data import Dataset
from ...data.constants import T_SamplesIndexDtype
from ...interface import Horizon, TTreatmentScenarios

TPreparedData = Union[torch.Tensor, DataLoader]


class OrganizedModule(nn.Module, ABC):
    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        nn.Module.__init__(self)

    @abstractmethod
    def _init_submodules(self) -> None:
        ...

    @abstractmethod
    def _init_optimizers(self) -> None:
        ...

    @abstractmethod
    def _init_inferred_params(self, data: Dataset, **kwargs) -> None:
        ...

    @abstractmethod
    def _prep_data_for_fit(self, data: Dataset, **kwargs) -> Tuple[TPreparedData, ...]:
        ...

    @abstractmethod
    def _prep_submodules_for_fit(self) -> None:
        ...

    def prep_fit(self, data: Dataset, **kwargs) -> Tuple[TPreparedData, ...]:
        self._init_inferred_params(data, **kwargs)
        self._init_submodules()
        self._init_optimizers()
        self._prep_submodules_for_fit()
        return self._prep_data_for_fit(data=data, **kwargs)


class OrganizedPredictorModuleMixin(ABC):
    @abstractmethod
    def _prep_data_for_predict(self, data: Dataset, horizon: Horizon, **kwargs) -> Tuple[TPreparedData, ...]:
        ...

    @abstractmethod
    def _prep_submodules_for_predict(self) -> None:
        ...

    def prep_predict(self, data: Dataset, horizon: Horizon, **kwargs) -> Tuple[TPreparedData, ...]:
        self._prep_submodules_for_predict()
        return self._prep_data_for_predict(data=data, horizon=horizon, **kwargs)


class OrganizedTreatmentEffectsModuleMixin(ABC):
    @abstractmethod
    def _prep_data_for_predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Horizon,
        **kwargs,
    ) -> Tuple[TPreparedData, ...]:
        ...

    @abstractmethod
    def _prep_submodules_for_predict_counterfactuals(self) -> None:
        ...

    def prep_predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Horizon,
        **kwargs,
    ) -> Tuple[TPreparedData, ...]:
        self._prep_submodules_for_predict_counterfactuals()
        return self._prep_data_for_predict_counterfactuals(
            data=data, sample_index=sample_index, treatment_scenarios=treatment_scenarios, horizon=horizon, **kwargs
        )


class CustomizableLossMixin(ABC):
    def __init__(self, loss_fn) -> None:
        assert isinstance(loss_fn, nn.Module)
        self.loss_fn: nn.Module = loss_fn

    @abstractmethod
    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        ...
