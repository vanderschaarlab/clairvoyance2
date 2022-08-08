from dataclasses import astuple, dataclass
from typing import Optional

from .dataformat import StaticSamples, TimeSeriesSamples
from .dataformat_base import Copyable

TTemporalCovariates = TimeSeriesSamples
TTemporalTargets = TimeSeriesSamples
TStaticCovariates = StaticSamples


# TODO: Easy initialisation from pd, np etc.
# TODO: Validate sample indexes match.
@dataclass(init=False, repr=False)
class Dataset(Copyable):
    temporal_covariates: TTemporalCovariates
    static_covariates: Optional[TStaticCovariates] = None
    temporal_targets: Optional[TTemporalTargets] = None

    def __init__(
        self,
        temporal_covariates: TTemporalCovariates,
        static_covariates: Optional[TStaticCovariates] = None,
        temporal_targets: Optional[TTemporalTargets] = None,
    ) -> None:
        self.temporal_covariates = temporal_covariates
        self.static_covariates = static_covariates
        self.temporal_targets = temporal_targets
        self.validate()

    @staticmethod
    def _time_series_samples_repr(time_series_samples: TimeSeriesSamples) -> str:
        name = time_series_samples.__class__.__name__
        shape = f"[{time_series_samples.n_samples},*,{len(time_series_samples.features)}]"
        return f"{name}({shape})"

    @staticmethod
    def _static_samples_repr(static_samples: StaticSamples) -> str:
        name = static_samples.__class__.__name__
        shape = f"[{static_samples.n_samples},{len(static_samples.features)}]"
        return f"{name}({shape})"

    def __repr__(self) -> str:
        tab = "    "
        sep = f"\n{tab}"

        attributes_repr = f"{sep}temporal_covariates={self._time_series_samples_repr(self.temporal_covariates)},"
        if self.static_covariates is not None:
            attributes_repr += f"{sep}static_covariates={self._static_samples_repr(self.static_covariates)},"
        if self.temporal_targets is not None:
            attributes_repr += f"{sep}temporal_targets={self._time_series_samples_repr(self.temporal_targets)},"

        return f"{self.__class__.__name__}({attributes_repr}\n)"

    def validate(self) -> None:
        # Types:
        if not isinstance(self.temporal_covariates, TimeSeriesSamples):
            raise TypeError("`temporal_covariates` must be of type `TimeSeriesSamples`")
        if self.static_covariates is not None and not isinstance(self.static_covariates, StaticSamples):
            raise TypeError("`static_covariates` must be of type `StaticSamples`")
        if self.temporal_targets is not None and not isinstance(self.temporal_targets, TimeSeriesSamples):
            raise TypeError("`temporal_targets` must be of type `TimeSeriesSamples`")
        # Number of samples:
        n_samples_expected = self.temporal_covariates.n_samples
        if self.static_covariates is not None and self.static_covariates.n_samples != n_samples_expected:
            raise ValueError(
                f"Expected {n_samples_expected} samples "
                f"but found {self.static_covariates.n_samples} in `static_covariates`"
            )
        if self.temporal_targets is not None and self.temporal_targets.n_samples != n_samples_expected:
            raise ValueError(
                f"Expected {n_samples_expected} samples "
                f"but found {self.temporal_targets.n_samples} in `temporal_targets`"
            )

    @property
    def n_samples(self) -> int:
        return self.temporal_covariates.n_samples

    def __iter__(self):
        # In order to be able to unroll.
        return iter(astuple(self))
