import math

import numpy as np

from ..data import Dataset, StaticSamples, TimeSeriesSamples


class DummyDatasetGenerator:
    def __init__(
        self,
        n_samples: int,
        temporal_covariates_n_features: int,
        temporal_covariates_max_len: int,
        temporal_covariates_missing_prob: float,
        static_covariates_n_features: int,
        static_covariates_missing_prob: float,
    ) -> None:
        self.n_samples = n_samples
        self.temporal_covariates_n_features = temporal_covariates_n_features
        self.temporal_covariates_max_len = temporal_covariates_max_len
        self.temporal_covariates_missing_prob = temporal_covariates_missing_prob
        self.static_covariates_n_features = static_covariates_n_features
        self.static_covariates_missing_prob = static_covariates_missing_prob

    def _generate_temporal_covariates(self, random_seed: int) -> TimeSeriesSamples:
        rng = np.random.default_rng(seed=random_seed)
        lens = rng.integers(
            low=math.ceil(self.temporal_covariates_max_len / 2),
            high=self.temporal_covariates_max_len,
            size=(self.n_samples,),
        )
        noise_mus = rng.standard_normal(size=(self.temporal_covariates_n_features,))
        noise_sigmas = 1.0 + rng.standard_normal(size=(self.temporal_covariates_n_features,))
        list_ = []
        for sample_len in lens:
            trend = np.tile(np.arange(sample_len + 1), (self.temporal_covariates_n_features, 1)).T
            noise = noise_mus + noise_sigmas * rng.standard_normal(
                size=(sample_len + 1, self.temporal_covariates_n_features)
            )
            final = trend + noise
            missing = rng.uniform(low=0.0, high=1.0, size=final.shape) < self.temporal_covariates_missing_prob
            final[missing] = np.nan
            list_.append(final)
        return TimeSeriesSamples(data=list_)

    def _generate_static_covariates(self, random_seed: int) -> StaticSamples:
        rng = np.random.default_rng(seed=random_seed)
        mus = 2.0 + rng.standard_normal(size=(self.static_covariates_n_features,))
        sigmas = 0.5 * rng.standard_normal(size=(self.static_covariates_n_features,))
        array = mus + sigmas * rng.standard_normal(size=(self.n_samples, self.static_covariates_n_features))
        missing = rng.uniform(low=0.0, high=1.0, size=array.shape) < self.static_covariates_missing_prob
        array[missing] = np.nan
        return StaticSamples(data=array)

    def generate(self, random_seed: int) -> Dataset:
        temporal_covariates = self._generate_temporal_covariates(random_seed)
        static_covariates = self._generate_static_covariates(random_seed)
        return Dataset(temporal_covariates=temporal_covariates, static_covariates=static_covariates)


def dummy_dataset(
    n_samples: int = 100,
    temporal_covariates_n_features: int = 5,
    temporal_covariates_max_len: int = 20,
    temporal_covariates_missing_prob: float = 0.1,
    static_covariates_n_features: int = 4,
    static_covariates_missing_prob: float = 0.1,
    random_seed: int = 12345,
) -> Dataset:
    dummy_dataset_generator = DummyDatasetGenerator(
        n_samples=n_samples,
        temporal_covariates_n_features=temporal_covariates_n_features,
        temporal_covariates_max_len=temporal_covariates_max_len,
        temporal_covariates_missing_prob=temporal_covariates_missing_prob,
        static_covariates_n_features=static_covariates_n_features,
        static_covariates_missing_prob=static_covariates_missing_prob,
    )
    return dummy_dataset_generator.generate(random_seed=random_seed)
