from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

from ..data import StaticSamples, TimeSeriesSamples
from ..interface import DataStructureOpts

TMetricValue = Union[float]  # pyright: ignore

TMetricDataContainer = TypeVar("TMetricDataContainer", TimeSeriesSamples, StaticSamples)


# TODO: Metric Requirements and checks.
class BaseMetric(Generic[TMetricDataContainer], ABC):
    def __init__(self, metric_for: DataStructureOpts) -> None:
        self.metric_for = metric_for
        super().__init__()

    @abstractmethod
    def __call__(self, data_true: TMetricDataContainer, data_pred: TMetricDataContainer) -> TMetricValue:
        ...
