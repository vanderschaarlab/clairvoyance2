import numpy as np

from clairvoyance2.data import Dataset
from clairvoyance2.datasets import dummy_dataset


class TestIntegration:
    def test_dummy_dataset_custom_args(self):
        dataset = dummy_dataset(
            n_samples=100,
            temporal_covariates_n_features=3,
            temporal_covariates_max_len=30,
            temporal_covariates_missing_prob=0.5,
            static_covariates_n_features=5,
            static_covariates_missing_prob=0.3,
            random_seed=999,
        )

        assert isinstance(dataset, Dataset)
        assert dataset.temporal_covariates is not None
        assert dataset.static_covariates is not None
        assert dataset.temporal_targets is None
        assert dataset.n_samples == 100
        assert dataset.static_covariates.df.shape == (100, 5)
        assert dataset.temporal_covariates.df.shape == (100, 3)
        assert max(dataset.temporal_covariates.n_timesteps_per_sample) <= 30
        assert 15 <= min(dataset.temporal_covariates.n_timesteps_per_sample)

    def test_dummy_dataset_has_missing(self):
        dataset = dummy_dataset(static_covariates_missing_prob=1.0, temporal_covariates_missing_prob=1.0)
        assert dataset.temporal_covariates.has_missing
        assert dataset.static_covariates.has_missing

    def test_dummy_dataset_no_missing(self):
        dataset = dummy_dataset(static_covariates_missing_prob=0.0, temporal_covariates_missing_prob=0.0)
        assert not dataset.temporal_covariates.has_missing
        assert not dataset.static_covariates.has_missing

    def test_dummy_dataset_with_temporal_targets_continuous(self):
        dataset = dummy_dataset(
            temporal_targets_n_features=3,
            temporal_targets_n_categories=None,
            temporal_covariates_max_len=30,  # NOTE: t. targ. will use the same per sample lengths as t. cov.
        )
        assert dataset.temporal_targets is not None
        assert dataset.temporal_targets.df.shape == (100, 3)
        assert dataset.temporal_targets.n_timesteps_per_sample == dataset.temporal_covariates.n_timesteps_per_sample
        assert max(dataset.temporal_targets.n_timesteps_per_sample) <= 30
        assert 15 <= min(dataset.temporal_targets.n_timesteps_per_sample)

    def test_dummy_dataset_with_temporal_targets_categorical(self):
        dataset = dummy_dataset(temporal_targets_n_features=3, temporal_targets_n_categories=5)
        assert dataset.temporal_targets is not None
        assert dataset.temporal_targets.df.shape == (100, 3)
        assert dataset.temporal_targets.n_timesteps_per_sample == dataset.temporal_covariates.n_timesteps_per_sample
        all_vals_as_np_array = dataset.temporal_targets.to_multi_index_dataframe().values.copy()
        unqiues = sorted(list(np.unique(all_vals_as_np_array)))
        assert unqiues == [0, 1, 2, 3, 4]
