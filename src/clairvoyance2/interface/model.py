import pprint
from abc import ABC, abstractmethod
from typing import Callable, Dict

from ..data import StaticSamples, TDataset, TimeSeriesSamples


class BaseModel(ABC):
    def __init__(self, params: Dict) -> None:
        self.params = params
        super().__init__()

    @staticmethod
    def parse_data_argument(data: TDataset):
        # NOTE: Very provisional.
        if not isinstance(data, tuple):
            assert isinstance(data, TimeSeriesSamples)
            return data
        else:
            assert len(data) == 2
            time_series_samples, static_samples = data
            assert isinstance(time_series_samples, TimeSeriesSamples)
            assert isinstance(static_samples, StaticSamples)
            return time_series_samples, static_samples

    def fit(self, data: TDataset) -> "BaseModel":
        # NOTE: Simply an alias for train. Do not override in derived classes!
        return self.train(data)

    @abstractmethod
    def train(self, data: TDataset) -> "BaseModel":
        ...

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(\n"
        tab = "    "
        pp = pprint.PrettyPrinter(indent=4)

        pretty_params = pp.pformat(self.params).replace("\t", tab)
        params_prefix = "params="
        params = f"{params_prefix}{pretty_params}"
        params = tab + f"\n{tab}{' ' * len(params_prefix)}".join(params.split("\n"))

        repr_str += f"{params}\n)"

        return repr_str


class TransformerMixin(ABC):
    fit: Callable

    @abstractmethod
    def transform(self, data: TDataset) -> TDataset:
        ...

    def fit_transform(self, data: TDataset) -> TDataset:
        try:
            self.fit(data)
        except AttributeError as ex:
            if self.fit.__name__ in str(ex):
                pass
            else:
                raise
        return self.transform(data)
