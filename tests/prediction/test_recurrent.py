import pytest

from clairvoyance2.datasets.dummy import DummyDatasetGenerator
from clairvoyance2.prediction import NStepAheadHorizon, RecurrentPredictor
from clairvoyance2.preprocessing import ExtractTargetsTS

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def dummy_dataset():
    dg = DummyDatasetGenerator(
        n_samples=100,
        temporal_covariates_n_features=5,
        temporal_covariates_max_len=30,
        temporal_covariates_missing_prob=0.0,
        static_covariates_n_features=1,  # Not relevant
        static_covariates_missing_prob=0.0,  # Not relevant
    )
    return dg.generate(random_seed=12345)


class TestIntegration:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "targets, horizon_n_step",
        [
            ([4], 1),
            ([2, 4], 1),
            ([1], 3),
            ([1, 2, 3], 3),
        ],
    )
    def test_successful_fit_predict_on_dummy_dataset(self, dummy_dataset, targets, horizon_n_step):
        # Arrange.
        horizon_n_step = 1
        data = ExtractTargetsTS(params=dict(targets=targets)).fit_transform(dummy_dataset)

        # Act.
        predictor = RecurrentPredictor(params=dict(epochs=5))
        predictor: RecurrentPredictor = predictor.fit(data, horizon=NStepAheadHorizon(horizon_n_step))
        data_pred = predictor.predict(data, horizon=NStepAheadHorizon(horizon_n_step))

        # Assert.
        # Check the data containers are present.
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None
        assert data_pred.temporal_covariates is not None
        assert data_pred.temporal_targets is not None
        # Check temporal target lengths make sense.
        assert max(data.temporal_targets.n_timesteps_per_sample) == 30
        assert max(data_pred.temporal_targets.n_timesteps_per_sample) == 30 - horizon_n_step
        # Check padding values didn't end up in predictions.
        assert (data_pred.temporal_targets.to_multi_index_dataframe().values != predictor.padding_value).all().all()
        # Check predicted features.
        assert len(targets) == len(data_pred.temporal_targets.features)
        assert [f in targets for f in data_pred.temporal_targets.features.keys()]
        # Check prediction index makes sense.
        ts0 = data.temporal_targets[0]
        ts0_pred = data_pred.temporal_targets[0]
        assert list(ts0_pred.df.index)[: len(ts0.df.index)] == [i + horizon_n_step for i in ts0.df.index]
        # ^ E.g. ts0.df.index = [0, 1, 2, 3] ==> ts0_pred.df.index = [1, 2, 3] for horizon = 1
