from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from clairvoyance2.data import Dataset, TimeSeriesSamples

from clairvoyance2.data import TimeSeries
from clairvoyance2.datasets.dummy import dummy_dataset
from clairvoyance2.preprocessing import TemporalTargetsExtractor
from clairvoyance2.treatment_effects.crn import (
    CRNClassifier,
    CRNRegressor,
    TimeIndexHorizon,
)

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


def get_dummy_data(with_static: bool):
    dataset = dummy_dataset(
        n_samples=3,
        temporal_covariates_n_features=5,
        temporal_covariates_max_len=6 * 2,
        temporal_covariates_missing_prob=0.0,
        static_covariates_n_features=0,
        temporal_treatments_n_features=2,
        temporal_treatments_n_categories=1,
        random_seed=12345,
    )
    if not with_static:
        dataset.static_covariates = None
    return dataset


@pytest.fixture
def dummy_data():
    return get_dummy_data(with_static=True)


@pytest.fixture
def dummy_data_no_static():
    return get_dummy_data(with_static=False)


def assert_all(
    data: "Dataset",
    pred: "TimeSeriesSamples",
    targets,
    horizon: TimeIndexHorizon,
    counterfactuals,
    n_counterfactuals_per_sample,
    padding_indicator,
):
    # Check the data containers are present.
    assert data.temporal_covariates is not None
    assert data.temporal_targets is not None
    assert pred is not None

    # Check padding values didn't end up in predictions.
    if not np.isnan(padding_indicator):
        assert (pred.to_multi_index_dataframe().values != padding_indicator).all().all()
    else:
        assert (~np.isnan(pred.to_multi_index_dataframe().values)).all().all()
    # Check no nans in general.
    if not np.isnan(padding_indicator):
        assert (~np.isnan(pred.to_multi_index_dataframe().values)).all().all()
    # Check predicted features.
    assert len(targets) == len(pred.features)
    assert [f in targets for f in pred.feature_names]

    # Check prediction index makes sense.
    ts: "TimeSeries"
    for ts, hor_ti in zip(pred, horizon.time_index_sequence):
        assert list(ts.time_index) == list(hor_ti)

    # Counterfactuals basic checks:
    assert len(counterfactuals) == data.n_samples
    for counterfactuals_sample in counterfactuals:
        assert len(counterfactuals_sample) == n_counterfactuals_per_sample
        assert all(isinstance(c, TimeSeries) for c in counterfactuals_sample)


def get_counterfactuals(data, model, horizon_counterfactuals, n_counterfactuals_per_sample, device=None):
    # TODO: This needs to be simplified.
    if device is not None:
        kwargs = {"device": device}
    else:
        kwargs = dict()
    counterfactuals = []
    for idx, sample_idx in enumerate(data.sample_indices):
        treat = data.temporal_treatments[sample_idx].df.values
        horizon_counterfactuals_sample = horizon_counterfactuals.time_index_sequence[idx]
        treat_scenarios = []
        for treat_sc_idx in range(n_counterfactuals_per_sample):
            np.random.seed(12345 + treat_sc_idx)
            treat_sc = np.random.randint(low=0, high=1 + 1, size=(len(horizon_counterfactuals_sample), treat.shape[1]))
            treat_scenarios.append(treat_sc)
        c = model.predict_counterfactuals(
            data,
            sample_index=sample_idx,
            treatment_scenarios=treat_scenarios,
            horizon=TimeIndexHorizon(time_index_sequence=[horizon_counterfactuals_sample]),
            **kwargs,
        )
        counterfactuals.append(c)
    return counterfactuals


class TestIntegration:
    class TestOnDummyDataset:
        @pytest.mark.slow
        @pytest.mark.vslow
        @pytest.mark.parametrize("param_encoder_rnn_type", ["GRU", "LSTM"])
        @pytest.mark.parametrize("param_decoder_rnn_type", ["LSTM", "RNN"])
        @pytest.mark.parametrize("param_encoder_num_layers", [1, 2])
        @pytest.mark.parametrize("param_decoder_hidden_size", [10, 20])
        @pytest.mark.parametrize("targets", ([4], [2, 4]))
        def test_fit_predict_vary_params_vary_target_n_feat(
            self,
            dummy_data,
            param_encoder_rnn_type,
            param_decoder_rnn_type,
            param_encoder_num_layers,
            param_decoder_hidden_size,
            targets,
        ):
            # Arrange.
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            treatment_effects_model = CRNRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=param_decoder_hidden_size,
                    encoder_num_layers=param_encoder_num_layers,
                    decoder_num_layers=1,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type=param_encoder_rnn_type,
                    decoder_rnn_type=param_decoder_rnn_type,
                    adapter_hidden_dims=[],
                )
            )
            horizon = TimeIndexHorizon(
                time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
            )
            treatment_effects_model: CRNRegressor = treatment_effects_model.fit(data, horizon=horizon)
            data_pred = treatment_effects_model.predict(data, horizon=horizon)

            n_counterfactuals_per_sample = 2
            horizon_counterfactuals = TimeIndexHorizon(
                time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
            )
            counterfactuals = get_counterfactuals(
                data=data,
                model=treatment_effects_model,
                horizon_counterfactuals=horizon_counterfactuals,
                n_counterfactuals_per_sample=n_counterfactuals_per_sample,
            )

            # Assert.
            assert_all(
                data,
                data_pred,
                targets,
                horizon,
                counterfactuals,
                n_counterfactuals_per_sample,
                treatment_effects_model.params.padding_indicator,
            )

        def test_fit_predict_no_static_cov(self, dummy_data_no_static):
            # Arrange.
            targets = [2, 4]
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data_no_static)

            # Act.
            treatment_effects_model = CRNRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=10,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="GRU",
                    decoder_rnn_type="LSTM",
                    adapter_hidden_dims=[2, 2],
                )
            )
            horizon = TimeIndexHorizon(
                time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
            )
            treatment_effects_model: CRNRegressor = treatment_effects_model.fit(data, horizon=horizon)
            data_pred = treatment_effects_model.predict(data, horizon=horizon)

            n_counterfactuals_per_sample = 2
            horizon_counterfactuals = TimeIndexHorizon(
                time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
            )
            counterfactuals = get_counterfactuals(
                data=data,
                model=treatment_effects_model,
                horizon_counterfactuals=horizon_counterfactuals,
                n_counterfactuals_per_sample=n_counterfactuals_per_sample,
            )

            # Assert.
            assert_all(
                data,
                data_pred,
                targets,
                horizon,
                counterfactuals,
                n_counterfactuals_per_sample,
                treatment_effects_model.params.padding_indicator,
            )

        def test_fit_predict_classifier(self):
            # Arrange.
            targets = [0, 1]
            data = dummy_dataset(
                n_samples=3,
                temporal_covariates_n_features=5,
                temporal_covariates_max_len=6 * 2,
                temporal_covariates_missing_prob=0.0,
                temporal_targets_n_categories=1,
                temporal_targets_n_features=2,
                static_covariates_n_features=0,
                temporal_treatments_n_features=2,
                temporal_treatments_n_categories=1,
                random_seed=12345,
            )

            # Act.
            treatment_effects_model = CRNClassifier(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=10,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="GRU",
                    decoder_rnn_type="LSTM",
                    adapter_hidden_dims=[3],
                )
            )
            horizon = TimeIndexHorizon(
                time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
            )
            treatment_effects_model: CRNClassifier = treatment_effects_model.fit(data, horizon=horizon)
            data_pred = treatment_effects_model.predict(data, horizon=horizon)

            n_counterfactuals_per_sample = 2
            horizon_counterfactuals = TimeIndexHorizon(
                time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
            )
            counterfactuals = get_counterfactuals(
                data=data,
                model=treatment_effects_model,
                horizon_counterfactuals=horizon_counterfactuals,
                n_counterfactuals_per_sample=n_counterfactuals_per_sample,
            )

            # Assert.
            assert_all(
                data,
                data_pred,
                targets,
                horizon,
                counterfactuals,
                n_counterfactuals_per_sample,
                treatment_effects_model.params.padding_indicator,
            )

    def test_device_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("Skipping CUDA device test, as no CUDA devices available")

        # Arrange.
        targets = [0, 1]
        data = dummy_dataset(
            n_samples=3,
            temporal_covariates_n_features=5,
            temporal_covariates_max_len=6 * 2,
            temporal_covariates_missing_prob=0.0,
            temporal_targets_n_categories=1,
            temporal_targets_n_features=2,
            static_covariates_n_features=0,
            temporal_treatments_n_features=2,
            temporal_treatments_n_categories=1,
            random_seed=12345,
        )

        # Act.
        treatment_effects_model = CRNClassifier(
            params=dict(
                encoder_hidden_size=10,
                decoder_hidden_size=10,
                epochs=2,
                batch_size=2,
                encoder_rnn_type="GRU",
                decoder_rnn_type="LSTM",
                adapter_hidden_dims=[3],
            )
        )
        horizon = TimeIndexHorizon(
            time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
        )
        treatment_effects_model: CRNClassifier = treatment_effects_model.fit(data, horizon=horizon, device="cuda")
        data_pred = treatment_effects_model.predict(data, horizon=horizon, device="cuda")

        n_counterfactuals_per_sample = 2
        horizon_counterfactuals = TimeIndexHorizon(
            time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
        )
        counterfactuals = get_counterfactuals(
            data=data,
            model=treatment_effects_model,
            horizon_counterfactuals=horizon_counterfactuals,
            n_counterfactuals_per_sample=n_counterfactuals_per_sample,
            device="cuda",
        )

        # Assert.
        assert_all(
            data,
            data_pred,
            targets,
            horizon,
            counterfactuals,
            n_counterfactuals_per_sample,
            treatment_effects_model.params.padding_indicator,
        )
