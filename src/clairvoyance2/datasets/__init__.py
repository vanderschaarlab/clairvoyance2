from typing import Tuple, Union

from ..data import StaticSamples, TimeSeriesSamples

TDataset = Union[TimeSeriesSamples, Tuple[TimeSeriesSamples, StaticSamples]]  # NOTE: This will evolve.

from .uci import uci_diabetes  # noqa: E402

__all__ = ["uci_diabetes"]
