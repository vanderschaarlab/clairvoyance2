import os

import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data import Dataset, TimeSeriesSamples
from clairvoyance2.datasets.dummy import DummyDatasetGenerator
from clairvoyance2.prediction import NStepAheadHorizon, RecurrentNetNStepAheadRegressor
from clairvoyance2.preprocessing import ExtractTargetsTC

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def dummy_dataset():
    dg = DummyDatasetGenerator(
        n_samples=10,
        temporal_covariates_n_features=5,
        temporal_covariates_max_len=9,
        temporal_covariates_missing_prob=0.0,
        static_covariates_n_features=1,  # Not relevant
        static_covariates_missing_prob=0.0,  # Not relevant
    )
    return dg.generate(random_seed=12345)


def assert_all(data, data_pred, targets, horizon_n_step, padding_indicator):
    # Check the data containers are present.
    assert data.temporal_covariates is not None
    assert data.temporal_targets is not None
    assert data_pred.temporal_covariates is not None
    assert data_pred.temporal_targets is not None
    # Check temporal target lengths make sense.
    # assert max(data.temporal_targets.n_timesteps_per_sample) == 10
    assert (
        max(data_pred.temporal_targets.n_timesteps_per_sample)
        == max(data.temporal_targets.n_timesteps_per_sample) + horizon_n_step
    )
    # Check padding values didn't end up in predictions.
    if not np.isnan(padding_indicator):
        assert (data_pred.temporal_targets.to_multi_index_dataframe().values != padding_indicator).all().all()
    else:
        assert (~(data_pred.temporal_targets.to_multi_index_dataframe().values).isnull()).all().all()
    # Check predicted features.
    assert len(targets) == len(data_pred.temporal_targets.features)
    assert [f in targets for f in data_pred.temporal_targets.feature_names]
    # Check prediction index makes sense.
    ts0 = data.temporal_targets[0]
    ts0_pred = data_pred.temporal_targets[0]

    # TODO: This last check and the associated logic needs more thought:
    assert len(ts0_pred.time_index) == len(ts0.time_index) + horizon_n_step
    assert list(ts0_pred.time_index[:-horizon_n_step]) == list(ts0.time_index)
    # ^ E.g. Original time_index = [0, 1, 2, 3] ==> Pred. (n_step=1) time_index = [0, 1, 2, 3, 4]
    # NOTE: The value at time index 0 in the above case will be taken from original data.


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
        def test_fit_predict(self, dummy_dataset, targets, horizon_n_step, param_rnn_type):
            # Arrange.
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(hidden_size=10, epochs=3, batch_size=4, rnn_type=param_rnn_type)
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        def test_fit_predict_use_past_targets_false(self, dummy_dataset):
            # Arrange.
            targets = [4]
            horizon_n_step = 1
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(hidden_size=10, epochs=3, batch_size=4, use_past_targets=False)
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_rnn_type", ["RNN", "GRU", "LSTM"])
        @pytest.mark.parametrize("param_num_layers", [1, 2])
        @pytest.mark.parametrize("param_hidden_size", [10, 20])
        def test_sweep_main_rnn_params(
            self,
            dummy_dataset,
            param_rnn_type,
            param_num_layers,
            param_hidden_size,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(
                    hidden_size=param_hidden_size,
                    epochs=3,
                    batch_size=4,
                    rnn_type=param_rnn_type,
                    num_layers=param_num_layers,
                )
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_rnn_type", ["RNN", "GRU", "LSTM"])
        @pytest.mark.parametrize("param_num_layers", [1, 3])
        def test_sweep_bidirectional(
            self,
            dummy_dataset,
            param_rnn_type,
            param_num_layers,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(
                    hidden_size=10,
                    epochs=3,
                    batch_size=4,
                    rnn_type=param_rnn_type,
                    num_layers=param_num_layers,
                    bidirectional=True,
                )
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_proj_size", [None, 0, 1, 10])
        def test_lstm_proj_size_param(self, dummy_dataset, param_proj_size):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(epochs=3, batch_size=4, rnn_type="LSTM", proj_size=param_proj_size)
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        def test_batch_size_gt_n_samples(self, dummy_dataset):
            # Arrange.
            batch_size = 5000  # > 100 samples.
            targets = [2, 3]
            horizon_n_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(params=dict(epochs=3, batch_size=batch_size, rnn_type="LSTM"))
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_ff_hidden_dims", [[], [2], [3, 2]])
        @pytest.mark.parametrize("param_ff_out_activation", [None, "Sigmoid", "Tanh"])
        def test_ff_params(self, dummy_dataset, param_ff_hidden_dims, param_ff_out_activation):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(
                    epochs=3,
                    batch_size=4,
                    ff_hidden_dims=param_ff_hidden_dims,
                    ff_out_activation=param_ff_out_activation,
                )
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)

        def test_fit_save_load(self, dummy_dataset, tmpdir):
            # Arrange.
            targets = [2, 3]
            horizon_n_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(
                    epochs=3,
                    batch_size=4,
                    ff_hidden_dims=[2],
                    ff_out_activation="Sigmoid",
                )
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            path = os.path.join(tmpdir, "predictor.p")
            predictor.save(path)
            loaded_model = RecurrentNetNStepAheadRegressor.load(path)

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
        def test_fit_predict_non_numeric_irregular_unaligned_time_index(self, dummy_dataset):
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

            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_dataset)

            # Act.
            predictor = RecurrentNetNStepAheadRegressor(
                params=dict(hidden_size=10, epochs=3, batch_size=1, use_past_targets=False)
            )
            predictor: RecurrentNetNStepAheadRegressor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
            data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

            # Assert.
            assert_all(data, data_pred, targets, horizon_n_step, predictor.params.padding_indicator)
