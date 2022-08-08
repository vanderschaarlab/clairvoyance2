from contextlib import nullcontext as does_not_raise
from typing import NamedTuple, Optional
from unittest.mock import Mock

import pandas as pd
import pytest

from clairvoyance2.data import StaticSamples, TimeSeriesSamples
from clairvoyance2.interface import DatasetRequirements, NStepAheadHorizon, Requirements
from clairvoyance2.interface.requirements import (
    PredictionHorizon,
    PredictionRequirements,
    PredictionTarget,
    RequirementsChecker,
)

# ^ Don't want RequirementsChecker to be "public" so it stays within requirements module.


class MockDataset(NamedTuple):
    temporal_covariates: TimeSeriesSamples
    static_covariates: Optional[StaticSamples]
    temporal_targets: Optional[TimeSeriesSamples]


class TestDataRequirements:
    class TestGeneralRequirements:
        @pytest.mark.parametrize(
            "data, expectation",
            [
                (
                    MockDataset(Mock(), Mock(), None),
                    does_not_raise(),
                ),
                (
                    MockDataset(Mock(), None, None),
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_static_samples_present(self, data, expectation):
            requirements = Requirements(dataset_requirements=DatasetRequirements(requires_static_samples_present=True))
            requirements_checker = RequirementsChecker()

            with expectation:
                try:
                    requirements_checker.check_data_requirements(requirements, data)
                except Exception as ex:
                    assert "requires static samples" in str(ex)  # Check message text is helpful.
                    raise

        @pytest.mark.parametrize(
            "is_regular_returns_t_cov, is_regular_returns_t_targ, expectation",
            [
                (
                    (True, None),
                    (True, None),
                    does_not_raise(),
                ),
                (
                    (False, None),
                    (True, None),
                    pytest.raises(RuntimeError),
                ),
                (
                    (True, None),
                    (False, None),
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_time_series_samples_regular(
            self, is_regular_returns_t_cov, is_regular_returns_t_targ, expectation
        ):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_time_series_samples_regular=True)
            )
            requirements_checker = RequirementsChecker()

            t_cov = Mock()
            t_cov.is_regular = Mock(return_value=is_regular_returns_t_cov)

            t_targ = Mock()
            t_targ.is_regular = Mock(return_value=is_regular_returns_t_targ)

            with expectation:
                try:
                    requirements_checker.check_data_requirements(requirements, MockDataset(t_cov, None, t_targ))
                except Exception as ex:
                    assert "requires regular timeseries" in str(ex)  # Check message text is helpful.
                    raise

        @pytest.mark.parametrize(
            "is_aligned_returns_t_cov, is_aligned_returns_t_targ, expectation",
            [
                (
                    True,
                    True,
                    does_not_raise(),
                ),
                (
                    False,
                    True,
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    False,
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_time_series_samples_aligned(
            self, is_aligned_returns_t_cov, is_aligned_returns_t_targ, expectation
        ):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_time_series_samples_aligned=True)
            )
            requirements_checker = RequirementsChecker()

            t_cov = Mock()
            t_cov.is_aligned = Mock(return_value=is_aligned_returns_t_cov)

            t_targ = Mock()
            t_targ.is_aligned = Mock(return_value=is_aligned_returns_t_targ)

            with expectation:
                try:
                    requirements_checker.check_data_requirements(requirements, MockDataset(t_cov, None, t_targ))
                except Exception as ex:
                    assert "requires aligned timeseries" in str(ex)  # Check message text is helpful.
                    raise

        @pytest.mark.parametrize(
            "t_cov_index, t_targ_index, expectation",
            [
                (
                    pd.Index([1, 2, 10]),
                    None,
                    does_not_raise(),
                ),
                (
                    pd.Index(["a", "b", "c"]),
                    None,
                    pytest.raises(RuntimeError),
                ),
                (
                    pd.Index([1, 2, 10]),
                    pd.Index([1, 2, 10]),
                    does_not_raise(),
                ),
                (
                    pd.Index([1, 2, 10]),
                    pd.Index(["a", "b", "c"]),
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_time_series_index_numeric(self, t_cov_index, t_targ_index, expectation):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_time_series_index_numeric=True)
            )
            requirements_checker = RequirementsChecker()

            mock_t_cov_0 = Mock()
            mock_t_cov_0.df = Mock()
            mock_t_cov_0.df.index = t_cov_index
            mock_t_cov = [mock_t_cov_0]

            if t_targ_index is not None:
                mock_t_targ_0 = Mock()
                mock_t_targ_0.df = Mock()
                mock_t_targ_0.df.index = t_targ_index
                mock_t_targ = [mock_t_targ_0]
            else:
                mock_t_targ = None

            with expectation:
                try:
                    requirements_checker.check_data_requirements(
                        requirements, MockDataset(mock_t_cov, None, mock_t_targ)
                    )
                except Exception as ex:
                    assert "requires numeric timeseries index" in str(ex)  # Check message text is helpful.
                    raise

        @pytest.mark.parametrize(
            "temporal_covariates_has_missing, static_covariates_has_missing, temporal_targets_has_missing, "
            "expectation, expectation_text",
            [
                (
                    False,
                    False,
                    False,
                    does_not_raise(),
                    None,
                ),
                (
                    True,
                    False,
                    False,
                    pytest.raises(RuntimeError),
                    "temporal covariates had missing data",
                ),
                (
                    False,
                    True,
                    False,
                    pytest.raises(RuntimeError),
                    "static samples had missing data",
                ),
                (
                    True,
                    True,
                    False,
                    pytest.raises(RuntimeError),
                    "temporal covariates had missing data",
                ),
                (
                    False,
                    False,
                    True,
                    pytest.raises(RuntimeError),
                    "temporal targets had missing data",
                ),
            ],
        )
        def test_requires_no_missing_data(
            self,
            temporal_covariates_has_missing,
            static_covariates_has_missing,
            temporal_targets_has_missing,
            expectation,
            expectation_text,
        ):
            requirements = Requirements(dataset_requirements=DatasetRequirements(requires_no_missing_data=True))
            requirements_checker = RequirementsChecker()

            data_temporal_covariates = Mock()
            data_temporal_covariates.has_missing = temporal_covariates_has_missing

            data_static_covariates = Mock()
            data_static_covariates.has_missing = static_covariates_has_missing

            data_temporal_targets = Mock()
            data_temporal_targets.has_missing = temporal_targets_has_missing

            with expectation:
                try:
                    requirements_checker.check_data_requirements(
                        requirements,
                        MockDataset(data_temporal_covariates, data_static_covariates, data_temporal_targets),
                    )
                except Exception as ex:
                    assert expectation_text in str(ex)  # Check message text is helpful.
                    raise

        @pytest.mark.parametrize(
            "timeseries_samples_has_missing, expectation",
            [
                (
                    False,
                    does_not_raise(),
                ),
                (
                    True,
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_no_missing_data_no_static(self, timeseries_samples_has_missing, expectation):
            requirements = Requirements(dataset_requirements=DatasetRequirements(requires_no_missing_data=True))
            requirements_checker = RequirementsChecker()

            data_time_series_samples = Mock()
            data_time_series_samples.has_missing = timeseries_samples_has_missing

            with expectation:
                try:
                    requirements_checker.check_data_requirements(
                        requirements, MockDataset(data_time_series_samples, None, None)
                    )
                except Exception as ex:
                    assert "temporal covariates had missing data" in str(ex)  # Check message text is helpful.
                    raise

        @pytest.mark.parametrize(
            "requirement_value, data_containers, expectation",
            [
                (
                    True,
                    [
                        Mock(all_numeric_features=False),
                        Mock(all_numeric_features=False),
                        Mock(all_numeric_features=False),
                    ],
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    [
                        Mock(all_numeric_features=False),
                        Mock(all_numeric_features=True),
                        Mock(all_numeric_features=False),
                    ],
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    [
                        Mock(all_numeric_features=True),
                        Mock(all_numeric_features=True),
                        Mock(all_numeric_features=True),
                    ],
                    does_not_raise(),
                ),
                (
                    False,
                    [
                        Mock(all_numeric_features=False),
                        Mock(all_numeric_features=False),
                        Mock(all_numeric_features=False),
                    ],
                    does_not_raise(),
                ),
                (
                    False,
                    [
                        Mock(all_numeric_features=False),
                        Mock(all_numeric_features=True),
                        Mock(all_numeric_features=False),
                    ],
                    does_not_raise(),
                ),
                (
                    False,
                    [
                        Mock(all_numeric_features=True),
                        Mock(all_numeric_features=True),
                        Mock(all_numeric_features=True),
                    ],
                    does_not_raise(),
                ),
            ],
        )
        def test_requires_all_numeric_features(self, requirement_value, data_containers, expectation):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_all_numeric_features=requirement_value)
            )
            requirements_checker = RequirementsChecker()

            with expectation:
                try:
                    requirements_checker.check_data_requirements(requirements, MockDataset(*data_containers))
                except Exception as ex:
                    assert "contained some non-numeric features" in str(ex)  # Check message text is helpful.
                    raise

    class TestPredictionSpecificRequirements:
        class TestTargetTimeSeries:
            def test_fails_no_temporal_targets(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target=PredictionTarget.TIME_SERIES),
                )
                requirements_checker = RequirementsChecker()

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(requirements, MockDataset(Mock(), None, None))
                assert "must contain temporal targets" in str(excinfo.value).lower()

            def test_fails_no_horizon_passed_to_check(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target=PredictionTarget.TIME_SERIES),
                )
                requirements_checker = RequirementsChecker()

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(requirements, MockDataset(Mock(), None, Mock()))
                assert "horizon must be passed" in str(excinfo.value).lower()

            @pytest.mark.parametrize(
                "horizon, expectation",
                [
                    (
                        NStepAheadHorizon(1),
                        does_not_raise(),
                    ),
                    (
                        None,
                        pytest.raises(RuntimeError),
                    ),
                ],
            )
            class TestNStepAhead:
                def test_wrong_horizon(self, horizon, expectation):
                    requirements = Requirements(
                        dataset_requirements=DatasetRequirements(),
                        prediction_requirements=PredictionRequirements(
                            target=PredictionTarget.TIME_SERIES, horizon=PredictionHorizon.N_STEP_AHEAD
                        ),
                    )
                    requirements_checker = RequirementsChecker()

                    with expectation:
                        try:
                            requirements_checker.check_data_requirements(
                                requirements,
                                MockDataset(
                                    Mock(n_timesteps_per_sample=[100]), None, Mock(n_timesteps_per_sample=[100])
                                ),
                                # ^ To avoid failing subsequent length checks.
                                horizon=horizon,
                            )
                        except Exception as ex:
                            assert "horizon of type" in str(ex)  # Check message text is helpful.
                            raise

            @pytest.mark.parametrize(
                "horizon, t_cov_max_len, t_targ_max_len, expectation",
                [
                    (
                        NStepAheadHorizon(2),
                        5,
                        5,
                        does_not_raise(),
                    ),
                    (
                        NStepAheadHorizon(2),
                        5,
                        2,
                        pytest.raises(RuntimeError),
                    ),
                    (
                        NStepAheadHorizon(2),
                        5,
                        1,
                        pytest.raises(RuntimeError),
                    ),
                    (
                        NStepAheadHorizon(2),
                        2,
                        5,
                        pytest.raises(RuntimeError),
                    ),
                    (
                        NStepAheadHorizon(2),
                        1,
                        5,
                        pytest.raises(RuntimeError),
                    ),
                ],
            )
            def test_horizon_length_check(self, horizon, t_cov_max_len, t_targ_max_len, expectation):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target=PredictionTarget.TIME_SERIES, horizon=PredictionHorizon.N_STEP_AHEAD
                    ),
                )
                requirements_checker = RequirementsChecker()
                data = MockDataset(
                    Mock(n_timesteps_per_sample=[t_cov_max_len]), None, Mock(n_timesteps_per_sample=[t_targ_max_len])
                )

                with expectation:
                    try:
                        requirements_checker.check_data_requirements(
                            requirements,
                            data,
                            horizon=horizon,
                        )
                    except Exception as ex:
                        assert "horizon must be < max timesteps" in str(ex)  # Check message text is helpful.
                        raise


class TestPredictionRequirements:
    def test_prediction_requirements_set(self):
        predictor = Mock()
        predictor.requirements = Requirements(prediction_requirements=None)

        requirements_checker = RequirementsChecker()

        with pytest.raises(RuntimeError) as excinfo:
            requirements_checker.check_prediction_requirements(predictor)
        assert "must have prediction requirements" in str(excinfo.value).lower()
