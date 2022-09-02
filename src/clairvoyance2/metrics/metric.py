from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Union

from ..data import Dataset

TMetricValue = Union[float]  # pyright: ignore


class MetricFor(Enum):
    STATIC_TARGETS = auto()
    TEMPORAL_TARGETS = auto()


# TODO: Metric Requirements and checks.
class BaseMetric(ABC):
    def __init__(self, metric_for: MetricFor) -> None:
        self.metric_for = metric_for
        super().__init__()

    @abstractmethod
    def __call__(self, data_true: Dataset, data_pred: Dataset) -> TMetricValue:
        ...
