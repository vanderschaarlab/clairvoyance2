import os

import numpy as np
import pandas as pd
import pytest
import torch

from clairvoyance2.data import Dataset, TimeSeries, TimeSeriesSamples
from clairvoyance2.datasets.dummy import dummy_dataset
from clairvoyance2.interface.horizon import NStepAheadHorizon
from clairvoyance2.prediction import RNNClassifier, RNNRegressor
from clairvoyance2.preprocessing import TemporalTargetsExtractor

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def dummy_data():
    dataset = dummy_dataset(
        n_samples=3,
        temporal_covariates_n_features=5,
        temporal_covariates_max_len=6,
        temporal_covariates_missing_prob=0.0,
        static_covariates_n_features=0,
        random_seed=12345,
    )
    return dataset


def assert_all(data: Dataset, data_pred: TimeSeriesSamples, targets, horizon_n_step: int, padding_indicator: float):
    # Check the data containers are present.
    assert data.temporal_covariates is not None
    assert data.temporal_targets is not None
    assert data_pred is not None
    # Check temporal target lengths make sense.
    assert data_pred.n_timesteps_per_sample == data.temporal_targets.n_timesteps_per_sample
    # Check padding values didn't end up in predictions.
    if not np.isnan(padding_indicator):
        assert (data_pred.to_multi_index_dataframe().values != padding_indicator).all().all()
    else:
        assert (~np.isnan(data_pred.to_multi_index_dataframe().values)).all().all()
    # Check no nans in general.
    if not np.isnan(padding_indicator):
        assert (~np.isnan(data_pred.to_multi_index_dataframe().values)).all().all()
    # Check predicted features.
    assert len(targets) == len(data_pred.features)
    assert [f in targets for f in data_pred.feature_names]
    # Check prediction index makes sense.
    ts0 = data.temporal_targets[0]
    ts0_pred = data_pred[0]
    assert isinstance(ts0, TimeSeries)
    assert isinstance(ts0_pred, TimeSeries)
    assert len(ts0_pred.time_index) == len(ts0.time_index)
    assert list(ts0_pred.time_index[:-horizon_n_step]) == list(ts0.time_index[horizon_n_step:])
    # ^ E.g. Original time_index = [0, 1, 2, 3] ==> Pred. (n_step=1) time_index = [1, 2, 3, 4]


class TestIntegration:
    class TestOnDummyDataset:
        @pytest.mark.slow
        @pytest.mark.parametrize("param_rnn_type", ["RNN", "GRU", "LSTM"])
        @pytest.mark.parametrize(
            "targets, horizon_n_step",
            [
                ([4], 1),
                ([2, 4], 1),
                ([1], 3),
                ([1, 2, 3], 3),
            ],
        )
        def test_fit_predict_vary_targets_and_horizon(self, dummy_data, targets, horizon_n_step, param_rnn_type):
            # Arrange.
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(params=dict(hidden_size=10, epochs=3, batch_size=2, rnn_type=param_rnn_type))
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        def test_fit_predict_use_past_targets_false(self, dummy_data):
            # Arrange.
            targets = [4]
            horizon_n_step = 1
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(params=dict(hidden_size=10, epochs=3, batch_size=2, use_past_targets=False))
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_rnn_type", ["RNN", "GRU", "LSTM"])
        @pytest.mark.parametrize("param_num_layers", [1, 2])
        @pytest.mark.parametrize("param_hidden_size", [10, 20])
        def test_sweep_main_rnn_params(
            self,
            dummy_data,
            param_rnn_type,
            param_num_layers,
            param_hidden_size,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(
                params=dict(
                    hidden_size=param_hidden_size,
                    epochs=3,
                    batch_size=2,
                    rnn_type=param_rnn_type,
                    num_layers=param_num_layers,
                )
            )
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_rnn_type", ["RNN", "GRU", "LSTM"])
        @pytest.mark.parametrize("param_num_layers", [1, 3])
        def test_sweep_bidirectional(
            self,
            dummy_data,
            param_rnn_type,
            param_num_layers,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(
                params=dict(
                    hidden_size=10,
                    epochs=3,
                    batch_size=2,
                    rnn_type=param_rnn_type,
                    num_layers=param_num_layers,
                    bidirectional=True,
                )
            )
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_proj_size", [None, 0, 1, 10])
        def test_lstm_proj_size_param(self, dummy_data, param_proj_size):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(params=dict(epochs=3, batch_size=2, rnn_type="LSTM", proj_size=param_proj_size))
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        def test_batch_size_gt_n_samples(self, dummy_data):
            # Arrange.
            batch_size = 5000  # > 100 samples.
            targets = [2, 3]
            horizon_n_step = 2
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(params=dict(epochs=3, batch_size=batch_size, rnn_type="LSTM"))
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_ff_hidden_dims", [[], [2], [3, 2]])
        @pytest.mark.parametrize("param_ff_out_activation", [None, "Sigmoid", "Tanh"])
        def test_ff_params(self, dummy_data, param_ff_hidden_dims, param_ff_out_activation):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(
                params=dict(
                    epochs=3,
                    batch_size=2,
                    ff_hidden_dims=param_ff_hidden_dims,
                    ff_out_activation=param_ff_out_activation,
                )
            )
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        def test_fit_predict_classifier(self):
            # Arrange.
            horizon_n_step = 3
            targets = [0, 1]
            data = dummy_dataset(
                n_samples=3,
                temporal_covariates_n_features=5,
                temporal_covariates_max_len=6,
                temporal_covariates_missing_prob=0.0,
                temporal_targets_n_categories=1,
                temporal_targets_n_features=2,
                static_covariates_n_features=0,
                random_seed=12345,
            )

            # Act.
            predictor = RNNClassifier(params=dict(hidden_size=10, epochs=3, batch_size=2, rnn_type="LSTM"))
            predictor: RNNClassifier = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        def test_device_cuda(self):
            if not torch.cuda.is_available():
                pytest.skip("Skipping CUDA device test, as no CUDA devices available")

            # Arrange.
            horizon_n_step = 3
            targets = [0, 1]
            data = dummy_dataset(
                n_samples=3,
                temporal_covariates_n_features=5,
                temporal_covariates_max_len=6,
                temporal_covariates_missing_prob=0.0,
                temporal_targets_n_categories=1,
                temporal_targets_n_features=2,
                static_covariates_n_features=0,
                random_seed=12345,
            )

            # Act.
            predictor = RNNClassifier(params=dict(hidden_size=10, epochs=3, batch_size=2, rnn_type="LSTM"))
            predictor: RNNClassifier = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step), device="cuda")
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step), device="cuda")

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        def test_fit_save_load(self, dummy_data, tmpdir):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(
                params=dict(
                    epochs=3,
                    batch_size=2,
                    ff_hidden_dims=[2],
                    ff_out_activation="Sigmoid",
                )
            )
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            path = os.path.join(tmpdir, "predictor.p")
            predictor.save(path)
            loaded_model = RNNRegressor.load(path)

            # Assert:
            assert os.path.exists(os.path.join(tmpdir, "predictor.p.params"))
            assert os.path.exists(os.path.join(tmpdir, "predictor.p"))
            assert loaded_model.params == predictor.params
            assert loaded_model.inferred_params == predictor.inferred_params
            assert loaded_model.inferred_params != dict()
            assert len(list(loaded_model.parameters())) > 0
            assert loaded_model.inferred_params is not None
            for p_original, p_loaded in zip(predictor.parameters(), loaded_model.parameters()):
                assert (p_original == p_loaded).all()

    class TestOnManualData:
        def test_fit_predict_non_numeric_irregular_unaligned_time_index(self, dummy_data):
            # Arrange.
            targets = [4]
            horizon_n_step = 1

            dfs = [
                pd.DataFrame(
                    {"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "d": [0, 1, 1]},
                    index=pd.to_datetime(["2000-01-01", "2000-01-05", "2000-01-06"]),
                ),
                pd.DataFrame(
                    {"a": [7, 8, 9, 10], "b": [7.0, 8.0, 9.0, 10.0], "d": [0, 1, 0, 0]},
                    index=pd.to_datetime(["2000-02-01", "2000-02-03", "2000-02-11", "2000-02-20"]),
                ),
                pd.DataFrame(
                    {"a": [3, 4, 5], "b": [3.0, 4.0, 5.0], "d": [1, 1, 1]},
                    index=pd.to_datetime(["2000-03-01", "2000-03-02", "2000-03-03"]),
                ),
            ]
            data = Dataset(temporal_covariates=TimeSeriesSamples(dfs))

            data = TemporalTargetsExtractor(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = RNNRegressor(params=dict(hidden_size=10, epochs=3, batch_size=1, use_past_targets=False))
            predictor: RNNRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)
