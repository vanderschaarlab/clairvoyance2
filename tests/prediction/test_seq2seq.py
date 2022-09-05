import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from clairvoyance2.data import Dataset, TimeSeriesSamples, TimeSeries
from clairvoyance2.datasets.dummy import dummy_dataset
from clairvoyance2.prediction.seq2seq import (
    Seq2SeqCRNStyleClassifier,
    Seq2SeqCRNStyleRegressor,
    TimeIndexHorizon,
)
from clairvoyance2.preprocessing import ExtractTargetsTC

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


def get_dummy_data(with_static: bool):
    dataset = dummy_dataset(
        n_samples=3,
        temporal_covariates_n_features=5,
        temporal_covariates_max_len=6,
        temporal_covariates_missing_prob=0.0,
        static_covariates_n_features=2,
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
    data: "Dataset", pred: "TimeSeriesSamples", targets, horizon: TimeIndexHorizon, padding_indicator: float
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


class TestIntegration:
    class TestOnDummyDataset:
        @pytest.mark.slow
        @pytest.mark.vslow
        @pytest.mark.parametrize("param_rnn_type", ["RNN", "GRU", "LSTM"])
        @pytest.mark.parametrize(
            "targets, horizon_n_future_step",
            [
                ([4], 1),
                ([2, 4], 1),
                ([1], 3),
                ([1, 2, 3], 3),
            ],
        )
        def test_fit_predict_vary_targets_and_horizon(self, dummy_data, targets, horizon_n_future_step, param_rnn_type):
            # Arrange.
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = Seq2SeqCRNStyleRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=10,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type=param_rnn_type,
                    decoder_rnn_type=param_rnn_type,
                    adapter_hidden_dims=[],
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleRegressor = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        def test_fit_predict_non_future_horizon(self, dummy_data):
            # Arrange.
            targets = [0, 1]
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = Seq2SeqCRNStyleRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=10,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="GRU",
                    decoder_rnn_type="LSTM",
                    adapter_hidden_dims=[],
                )
            )
            horizon = TimeIndexHorizon(
                time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in data.temporal_covariates]
            )
            predictor: Seq2SeqCRNStyleRegressor = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.vslow
        @pytest.mark.parametrize("param_encoder_num_layers", [1, 2])
        @pytest.mark.parametrize("param_decoder_num_layers", [1, 2])
        @pytest.mark.parametrize("param_encoder_hidden_size", [10, 20])
        @pytest.mark.parametrize("param_decoder_hidden_size", [10, 20])
        def test_sweep_rnn_sizes(
            self,
            dummy_data,
            param_encoder_num_layers,
            param_decoder_num_layers,
            param_encoder_hidden_size,
            param_decoder_hidden_size,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_future_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = Seq2SeqCRNStyleRegressor(
                params=dict(
                    encoder_hidden_size=param_encoder_hidden_size,
                    decoder_hidden_size=param_decoder_hidden_size,
                    encoder_num_layers=param_encoder_num_layers,
                    decoder_num_layers=param_decoder_num_layers,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="GRU",
                    decoder_rnn_type="LSTM",
                    adapter_hidden_dims=[],
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleRegressor = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.vslow
        @pytest.mark.parametrize("param_encoder_rnn_type", ["RNN", "GRU", "LSTM"])
        @pytest.mark.parametrize("param_decoder_rnn_type", ["RNN", "GRU", "LSTM"])
        def test_sweep_rnn_types(
            self,
            dummy_data,
            param_encoder_rnn_type,
            param_decoder_rnn_type,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_future_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = Seq2SeqCRNStyleRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=20,
                    encoder_num_layers=2,
                    decoder_num_layers=1,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type=param_encoder_rnn_type,
                    decoder_rnn_type=param_decoder_rnn_type,
                    adapter_hidden_dims=[],
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleRegressor = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_adapter_hidden_dims", [[], [2], [3, 2]])
        def test_sweep_adapter_params(
            self,
            dummy_data,
            param_adapter_hidden_dims,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_future_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = Seq2SeqCRNStyleRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=20,
                    encoder_num_layers=2,
                    decoder_num_layers=1,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="LSTM",
                    decoder_rnn_type="GRU",
                    adapter_hidden_dims=param_adapter_hidden_dims,
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleRegressor = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        @pytest.mark.slow
        @pytest.mark.parametrize("param_predictor_hidden_dim", [[], [2], [3, 2]])
        def test_sweep_predictor_params(
            self,
            dummy_data,
            param_predictor_hidden_dim,
        ):
            # Arrange.
            targets = [2, 3]
            horizon_n_future_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = Seq2SeqCRNStyleRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=20,
                    encoder_num_layers=2,
                    decoder_num_layers=1,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="LSTM",
                    decoder_rnn_type="GRU",
                    predictor_hidden_dims=param_predictor_hidden_dim,
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleRegressor = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        def test_fit_predict_no_static_cov(self, dummy_data_no_static):
            # Arrange.
            horizon_n_future_step = 3
            targets = [0, 1]
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data_no_static)

            # Act.
            predictor = Seq2SeqCRNStyleClassifier(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=10,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="RNN",
                    decoder_rnn_type="LSTM",
                    adapter_hidden_dims=[3],
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleClassifier = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        def test_fit_predict_classifier(self):
            # Arrange.
            horizon_n_future_step = 3
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
            predictor = Seq2SeqCRNStyleClassifier(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=10,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="LSTM",
                    decoder_rnn_type="GRU",
                    adapter_hidden_dims=[],
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleClassifier = predictor.fit(data, horizon=horizon)
            data_pred = predictor.predict(data, horizon=horizon)

            # Assert.
            assert_all(data, data_pred, targets, horizon, predictor.params.padding_indicator)

        def test_fit_save_load(self, dummy_data, tmpdir):
            # Arrange.
            targets = [2, 3]
            horizon_n_future_step = 2
            data = ExtractTargetsTC(params=dict(targets=targets)).fit_transform(dummy_data)

            # Act.
            predictor = Seq2SeqCRNStyleRegressor(
                params=dict(
                    encoder_hidden_size=10,
                    decoder_hidden_size=20,
                    encoder_num_layers=2,
                    decoder_num_layers=1,
                    epochs=2,
                    batch_size=2,
                    encoder_rnn_type="LSTM",
                    decoder_rnn_type="GRU",
                    predictor_hidden_dims=[2],
                    adapter_hidden_dims=[3],
                )
            )
            horizon = TimeIndexHorizon.future_horizon_from_dataset(
                data, forecast_n_future_steps=horizon_n_future_step, time_delta=1
            )
            predictor: Seq2SeqCRNStyleRegressor = predictor.fit(data, horizon=horizon)
            path = os.path.join(tmpdir, "predictor.p")
            predictor.save(path)
            loaded_model = Seq2SeqCRNStyleRegressor.load(path)

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
