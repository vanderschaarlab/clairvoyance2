from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock

import pandas as pd
import pytest

from clairvoyance2.data import Dataset, StaticSamples, TimeSeriesSamples
from clairvoyance2.interface import DatasetRequirements, NStepAheadHorizon, Requirements
from clairvoyance2.interface.requirements import (
    HorizonType,
    PredictionRequirements,
    PredictionTargetType,
    RequirementsChecker,
    TreatmentEffectsRequirements,
    TreatmentType,
    TreatmentValueType,
)

# ^ Don't want RequirementsChecker to be "public" so it stays within requirements module.


@dataclass
class MockDataset:
    temporal_covariates: TimeSeriesSamples
    static_covariates: Optional[StaticSamples] = None
    temporal_targets: Optional[TimeSeriesSamples] = None
    temporal_treatments: Optional[TimeSeriesSamples] = None

    @property
    def static_data_containers(self):
        return {
            k: v
            for k, v in {
                "static_covariates": self.static_covariates,
            }.items()
            if v is not None
        }

    @property
    def temporal_data_containers(self):
        return {
            k: v
            for k, v in {
                "temporal_covariates": self.temporal_covariates,
                "temporal_targets": self.temporal_targets,
                "temporal_treatments": self.temporal_treatments,
            }.items()
            if v is not None
        }

    @property
    def all_data_containers(self):
        return {
            k: v
            for k, v in {
                "temporal_covariates": self.temporal_covariates,
                "static_covariates": self.static_covariates,
                "temporal_targets": self.temporal_targets,
                "temporal_treatments": self.temporal_treatments,
            }.items()
            if v is not None
        }


class TestDataRequirements:
    class TestGeneralRequirements:
        @pytest.mark.parametrize(
            "data, expectation",
            [
                (
                    MockDataset(temporal_covariates=Mock(), static_covariates=Mock()),
                    does_not_raise(),
                ),
                (
                    MockDataset(temporal_covariates=Mock()),
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_static_samples_present(self, data, expectation):
            requirements = Requirements(dataset_requirements=DatasetRequirements(requires_static_samples_present=True))
            requirements_checker = RequirementsChecker()

            with expectation as excinfo:
                requirements_checker.check_data_requirements(requirements, data)
            if excinfo is not None:
                assert "requires static samples" in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "is_regular_returns_t_cov, is_regular_returns_t_targ, is_regular_returns_t_treat, expectation",
            [
                (
                    (True, None),
                    (True, None),
                    (True, None),
                    does_not_raise(),
                ),
                (
                    (False, None),
                    (True, None),
                    (True, None),
                    pytest.raises(RuntimeError),
                ),
                (
                    (True, None),
                    (False, None),
                    (True, None),
                    pytest.raises(RuntimeError),
                ),
                (
                    (True, None),
                    (True, None),
                    (False, None),
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_time_series_samples_regular(
            self, is_regular_returns_t_cov, is_regular_returns_t_targ, is_regular_returns_t_treat, expectation
        ):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_time_series_samples_regular=True)
            )
            requirements_checker = RequirementsChecker()

            t_cov = Mock()
            t_cov.is_regular = Mock(return_value=is_regular_returns_t_cov)

            t_targ = Mock()
            t_targ.is_regular = Mock(return_value=is_regular_returns_t_targ)

            t_treat = Mock()
            t_treat.is_regular = Mock(return_value=is_regular_returns_t_treat)

            with expectation as excinfo:
                requirements_checker.check_data_requirements(
                    requirements,
                    MockDataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat),
                )
            if excinfo is not None:
                assert "requires regular timeseries" in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "is_aligned_returns_t_cov, is_aligned_returns_t_targ, is_aligned_returns_t_treat, expectation",
            [
                (
                    True,
                    True,
                    True,
                    does_not_raise(),
                ),
                (
                    False,
                    True,
                    True,
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    False,
                    True,
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    True,
                    False,
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_time_series_samples_aligned(
            self, is_aligned_returns_t_cov, is_aligned_returns_t_targ, is_aligned_returns_t_treat, expectation
        ):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_time_series_samples_aligned=True)
            )
            requirements_checker = RequirementsChecker()

            t_cov = Mock()
            t_cov.is_aligned = Mock(return_value=is_aligned_returns_t_cov)

            t_targ = Mock()
            t_targ.is_aligned = Mock(return_value=is_aligned_returns_t_targ)

            t_treat = Mock()
            t_treat.is_aligned = Mock(return_value=is_aligned_returns_t_treat)

            with expectation as excinfo:
                requirements_checker.check_data_requirements(
                    requirements,
                    MockDataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat),
                )
            if excinfo is not None:
                assert "requires aligned timeseries" in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "t_cov_index, t_targ_index, t_treat_index, expectation",
            [
                (
                    pd.Index([1, 2, 10]),
                    None,
                    None,
                    does_not_raise(),
                ),
                (
                    pd.Index(["a", "b", "c"]),
                    None,
                    None,
                    pytest.raises(RuntimeError),
                ),
                (
                    pd.Index([1, 2, 10]),
                    pd.Index([1, 2, 10]),
                    None,
                    does_not_raise(),
                ),
                (
                    pd.Index([1, 2, 10]),
                    pd.Index(["a", "b", "c"]),
                    None,
                    pytest.raises(RuntimeError),
                ),
                (
                    pd.Index([1, 2, 10]),
                    pd.Index([1, 2, 10]),
                    pd.Index([1, 2, 10]),
                    does_not_raise(),
                ),
                (
                    pd.Index([1, 2, 10]),
                    pd.Index([1, 2, 10]),
                    pd.Index(["a", "b", "c"]),
                    pytest.raises(RuntimeError),
                ),
            ],
        )
        def test_requires_time_series_index_numeric(self, t_cov_index, t_targ_index, t_treat_index, expectation):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_time_series_index_numeric=True)
            )
            requirements_checker = RequirementsChecker()

            mock_t_cov_0 = Mock()
            mock_t_cov_0.df = Mock()
            mock_t_cov_0.time_index = t_cov_index
            mock_t_cov = [mock_t_cov_0]

            if t_targ_index is not None:
                mock_t_targ_0 = Mock()
                mock_t_targ_0.df = Mock()
                mock_t_targ_0.time_index = t_targ_index
                mock_t_targ = [mock_t_targ_0]
            else:
                mock_t_targ = None

            if t_treat_index is not None:
                mock_t_treat_0 = Mock()
                mock_t_treat_0.df = Mock()
                mock_t_treat_0.time_index = t_treat_index
                mock_t_treat = [mock_t_treat_0]
            else:
                mock_t_treat = None

            with expectation as excinfo:
                requirements_checker.check_data_requirements(
                    requirements,
                    MockDataset(
                        temporal_covariates=mock_t_cov, temporal_targets=mock_t_targ, temporal_treatments=mock_t_treat
                    ),
                )
            if excinfo is not None:
                assert "requires numeric timeseries index" in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "temporal_covariates_has_missing, static_covariates_has_missing, temporal_targets_has_missing, "
            "temporal_treatments_has_missing, expectation, expectation_text",
            [
                (
                    False,
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
                    False,
                    pytest.raises(RuntimeError),
                    "temporal covariates had missing data",
                ),
                (
                    False,
                    True,
                    False,
                    False,
                    pytest.raises(RuntimeError),
                    "static covariates had missing data",
                ),
                (
                    True,
                    True,
                    False,
                    False,
                    pytest.raises(RuntimeError),
                    "temporal covariates had missing data",
                ),
                (
                    False,
                    False,
                    True,
                    False,
                    pytest.raises(RuntimeError),
                    "temporal targets had missing data",
                ),
                (
                    False,
                    False,
                    False,
                    True,
                    pytest.raises(RuntimeError),
                    "temporal treatments had missing data",
                ),
            ],
        )
        def test_requires_no_missing_data(
            self,
            temporal_covariates_has_missing,
            static_covariates_has_missing,
            temporal_targets_has_missing,
            temporal_treatments_has_missing,
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

            data_temporal_treatments = Mock()
            data_temporal_treatments.has_missing = temporal_treatments_has_missing

            with expectation as excinfo:
                requirements_checker.check_data_requirements(
                    requirements,
                    MockDataset(
                        temporal_covariates=data_temporal_covariates,
                        static_covariates=data_static_covariates,
                        temporal_targets=data_temporal_targets,
                        temporal_treatments=data_temporal_treatments,
                    ),
                )
            if excinfo is not None:
                assert expectation_text in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "temporal_covariates_has_missing, expectation",
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
        def test_requires_no_missing_data_no_static(self, temporal_covariates_has_missing, expectation):
            requirements = Requirements(dataset_requirements=DatasetRequirements(requires_no_missing_data=True))
            requirements_checker = RequirementsChecker()

            data_temporal_covariates_samples = Mock()
            data_temporal_covariates_samples.has_missing = temporal_covariates_has_missing

            with expectation as excinfo:
                requirements_checker.check_data_requirements(
                    requirements, MockDataset(temporal_covariates=data_temporal_covariates_samples)
                )
            if excinfo is not None:
                assert "temporal covariates had missing data" in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "requirement_value, data_containers, expectation",
            [
                (
                    True,
                    [
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=False),
                    ],
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    [
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                    ],
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    [
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                    ],
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    [
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=True),
                    ],
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    [
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=False),
                    ],
                    pytest.raises(RuntimeError),
                ),
                (
                    True,
                    [
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                    ],
                    does_not_raise(),
                ),
                (
                    False,
                    [
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=False),
                        Mock(all_numeric_compatible_features=False),
                    ],
                    does_not_raise(),
                ),
                (
                    False,
                    [
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                        Mock(all_numeric_compatible_features=True),
                    ],
                    does_not_raise(),
                ),
            ],
        )
        def test_requires_all_numeric_compatible_features(self, requirement_value, data_containers, expectation):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_all_numeric_features=requirement_value)
            )
            requirements_checker = RequirementsChecker()

            with expectation as excinfo:
                requirements_checker.check_data_requirements(requirements, MockDataset(*data_containers))
            if excinfo is not None:
                assert "non-numeric features" in str(excinfo.value).lower()

    class TestPredictionSpecificRequirements:
        class TestTargetTimeSeries:
            def test_fails_no_temporal_targets(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target=PredictionTargetType.TIME_SERIES),
                )
                requirements_checker = RequirementsChecker()

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(requirements, MockDataset(temporal_covariates=Mock()))
                assert "must contain temporal targets" in str(excinfo.value).lower()

            def test_fails_no_horizon_passed_to_check(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target=PredictionTargetType.TIME_SERIES),
                )
                requirements_checker = RequirementsChecker()

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(
                        requirements, MockDataset(temporal_covariates=Mock(), temporal_targets=Mock())
                    )
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
                            target=PredictionTargetType.TIME_SERIES, horizon=HorizonType.N_STEP_AHEAD
                        ),
                    )
                    requirements_checker = RequirementsChecker()

                    with expectation as excinfo:
                        requirements_checker.check_data_requirements(
                            requirements,
                            MockDataset(
                                temporal_covariates=Mock(n_timesteps_per_sample=[100]),
                                temporal_targets=Mock(n_timesteps_per_sample=[100]),
                            ),
                            # ^ To avoid failing subsequent length checks.
                            horizon=horizon,
                        )
                    if excinfo is not None:
                        assert "horizon of type" in str(excinfo.value).lower()

            @pytest.mark.parametrize(
                "horizon, t_cov_max_len, t_targ_max_len, t_treat_max_len, expectation",
                [
                    (
                        NStepAheadHorizon(2),
                        5,
                        5,
                        5,
                        does_not_raise(),
                    ),
                    (
                        NStepAheadHorizon(2),
                        5,
                        2,
                        5,
                        pytest.raises(RuntimeError),
                    ),
                    (
                        NStepAheadHorizon(2),
                        5,
                        5,
                        2,
                        pytest.raises(RuntimeError),
                    ),
                    (
                        NStepAheadHorizon(2),
                        5,
                        1,
                        5,
                        pytest.raises(RuntimeError),
                    ),
                    (
                        NStepAheadHorizon(2),
                        2,
                        5,
                        5,
                        pytest.raises(RuntimeError),
                    ),
                    (
                        NStepAheadHorizon(2),
                        1,
                        5,
                        5,
                        pytest.raises(RuntimeError),
                    ),
                ],
            )
            def test_horizon_length_check(self, horizon, t_cov_max_len, t_targ_max_len, t_treat_max_len, expectation):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target=PredictionTargetType.TIME_SERIES, horizon=HorizonType.N_STEP_AHEAD
                    ),
                )
                requirements_checker = RequirementsChecker()
                data = MockDataset(
                    temporal_covariates=Mock(n_timesteps_per_sample=[t_cov_max_len]),
                    temporal_targets=Mock(n_timesteps_per_sample=[t_targ_max_len]),
                    temporal_treatments=Mock(n_timesteps_per_sample=[t_treat_max_len]),
                )

                with expectation as excinfo:
                    requirements_checker.check_data_requirements(
                        requirements,
                        data,
                        horizon=horizon,
                    )
                if excinfo is not None:
                    assert "horizon must be < max timesteps" in str(excinfo.value).lower()

    class TestTreatmentEffectsSpecificRequirements:
        class TestTreatmentTypeTimeSeries:
            def test_fails_no_temporal_targets(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target=PredictionTargetType.TIME_SERIES, horizon=HorizonType.TIME_INDEX
                    ),
                    treatment_effects_requirements=TreatmentEffectsRequirements(
                        treatment_type=TreatmentType.TIME_SERIES
                    ),
                )
                requirements_checker = RequirementsChecker()

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(
                        requirements, MockDataset(temporal_covariates=Mock()), horizon=Mock()
                    )
                assert "must contain temporal targets" in str(excinfo.value).lower()

            def test_fails_no_temporal_treatments(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target=PredictionTargetType.TIME_SERIES, horizon=HorizonType.TIME_INDEX
                    ),
                    treatment_effects_requirements=TreatmentEffectsRequirements(
                        treatment_type=TreatmentType.TIME_SERIES
                    ),
                )
                requirements_checker = RequirementsChecker()

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(
                        requirements, MockDataset(temporal_covariates=Mock(), temporal_targets=Mock()), horizon=Mock()
                    )
                assert "must contain temporal treatments" in str(excinfo.value).lower()

            @pytest.mark.parametrize(
                "treatment_value_type",
                [
                    TreatmentValueType.BINARY,
                    TreatmentValueType.CATEGORICAL,
                ],
            )
            def test_treatment_value_categorical_feature_mismatch(self, treatment_value_type):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target=PredictionTargetType.TIME_SERIES, horizon=HorizonType.TIME_INDEX
                    ),
                    treatment_effects_requirements=TreatmentEffectsRequirements(
                        treatment_type=TreatmentType.TIME_SERIES, treatment_value_type=treatment_value_type
                    ),
                )
                requirements_checker = RequirementsChecker()
                mock_temporal_treatments = Mock(all_categorical_features=False)
                mock_temporal_treatments.features = {
                    "dummy1": Mock(feature_type="dummy"),
                    "dummy2": Mock(feature_type="dummy"),
                }
                data = MockDataset(
                    temporal_covariates=Mock(),
                    temporal_targets=Mock(),
                    temporal_treatments=mock_temporal_treatments,
                )

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(requirements, data, horizon=Mock())
                assert "treatments must all be categorical" in str(excinfo.value).lower()

            def test_treatment_value_binary_extra_check(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target=PredictionTargetType.TIME_SERIES, horizon=HorizonType.TIME_INDEX
                    ),
                    treatment_effects_requirements=TreatmentEffectsRequirements(
                        treatment_type=TreatmentType.TIME_SERIES, treatment_value_type=TreatmentValueType.BINARY
                    ),
                )
                requirements_checker = RequirementsChecker()
                mock_temporal_treatments = Mock(all_categorical_features=True)

                def _f(_):
                    return (
                        x
                        for x in (
                            Mock(df=pd.DataFrame({"a": [0.0, 1.0, 0.0], "b": [0.0, 1.0, 1.0]})),
                            Mock(df=pd.DataFrame({"a": [0.0, 1.0, 0.0], "b": [0.0, 1.0, 999.0]})),
                        )
                    )

                mock_temporal_treatments.__iter__ = _f

                data = MockDataset(
                    temporal_covariates=Mock(),
                    temporal_targets=Mock(),
                    temporal_treatments=mock_temporal_treatments,
                )

                with pytest.raises(RuntimeError) as excinfo:
                    requirements_checker.check_data_requirements(requirements, data, horizon=Mock())
                assert "only contain values (0., 1.)" in str(excinfo.value).lower()


class TestPredictionRequirements:
    def test_prediction_requirements_set(self):
        predictor = Mock()
        predictor.requirements = Requirements(prediction_requirements=None)

        requirements_checker = RequirementsChecker()

        with pytest.raises(RuntimeError) as excinfo:
            requirements_checker.check_prediction_requirements(predictor)
        assert "must have prediction requirements" in str(excinfo.value).lower()


class TestIntegration:
    class TestDataRequirements:
        @pytest.mark.parametrize(
            "data, expect_str_1, expect_str_2",
            [
                (
                    Dataset(
                        temporal_covariates=TimeSeriesSamples(
                            [
                                pd.DataFrame({"a": [1, 2, 3]}, index=[1, 7, 8]),
                                pd.DataFrame({"a": [1, 2]}, index=[1, 10]),
                            ]
                        ),
                        temporal_targets=TimeSeriesSamples(
                            [
                                pd.DataFrame({"a": [10, 20, 30]}, index=[1, 7, 8]),
                                pd.DataFrame({"a": [10, 20]}, index=[1, 999]),  # <-- This one.
                            ]
                        ),
                    ),
                    "temporal_covariates",
                    "temporal_targets",
                ),
                (
                    Dataset(
                        temporal_covariates=TimeSeriesSamples(
                            [
                                pd.DataFrame({"a": [1, 2, 3]}, index=[1, 7, 8]),
                                pd.DataFrame({"a": [1, 2]}, index=[1, 10]),
                            ]
                        ),
                        temporal_treatments=TimeSeriesSamples(
                            [
                                pd.DataFrame({"a": [10, 20, 30]}, index=[1, 7, 8]),
                                pd.DataFrame({"a": [10, 20]}, index=[1, 999]),  # <-- This one.
                            ]
                        ),
                    ),
                    "temporal_covariates",
                    "temporal_treatments",
                ),
                (
                    Dataset(
                        temporal_covariates=TimeSeriesSamples(
                            [
                                pd.DataFrame({"a": [1, 2, 3]}, index=[1, 7, 8]),
                                pd.DataFrame({"a": [1, 2]}, index=[1, 10]),
                            ]
                        ),
                        temporal_targets=TimeSeriesSamples(
                            [
                                pd.DataFrame({"a": [1, 2, 3]}, index=[1, 7, 8]),
                                pd.DataFrame({"a": [1, 2]}, index=[1, 10]),
                            ]
                        ),
                        temporal_treatments=TimeSeriesSamples(
                            [
                                pd.DataFrame({"a": [10, 20, 30]}, index=[1, 7, 8]),
                                pd.DataFrame({"a": [10, 20]}, index=[1, 999]),  # <-- This one.
                            ]
                        ),
                    ),
                    "temporal_targets",
                    "temporal_treatments",
                ),
            ],
        )
        def test_requires_temporal_containers_have_same_time_index(self, data, expect_str_1, expect_str_2):
            req = Requirements(
                dataset_requirements=DatasetRequirements(requires_temporal_containers_have_same_time_index=True)
            )

            requirements_checker = RequirementsChecker()
            with pytest.raises(RuntimeError) as excinfo:
                requirements_checker.check_data_requirements(req, data)
            assert (
                "same time index" in str(excinfo.value)
                and expect_str_1 in str(excinfo.value)
                and expect_str_2 in str(excinfo.value)
            )
