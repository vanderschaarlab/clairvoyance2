from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock

import pandas as pd
import pytest

from clairvoyance2.data import Dataset, EventSamples, StaticSamples, TimeSeriesSamples
from clairvoyance2.interface import DatasetRequirements, NStepAheadHorizon, Requirements
from clairvoyance2.interface.requirements import (
    DataStructureOpts,
    DataValueOpts,
    HorizonOpts,
    PredictionRequirements,
    RequirementsChecker,
    TreatmentEffectsRequirements,
)

# ^ Don't want RequirementsChecker to be "public" so it stays within requirements module.


@dataclass
class MockDataset:
    temporal_covariates: TimeSeriesSamples
    static_covariates: Optional[EventSamples] = None
    event_covariates: Optional[StaticSamples] = None
    temporal_targets: Optional[TimeSeriesSamples] = None
    temporal_treatments: Optional[TimeSeriesSamples] = None
    event_targets: Optional[EventSamples] = None
    event_treatments: Optional[EventSamples] = None

    def check_temporal_containers_have_same_time_index(self):
        return True, None

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
    class TestDataRequirementsGeneral:
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
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_static_covariates_present=True)
            )

            with expectation as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data
                )
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
                dataset_requirements=DatasetRequirements(requires_all_temporal_data_regular=True)
            )

            t_cov = Mock()
            t_cov.is_regular = Mock(return_value=is_regular_returns_t_cov)

            t_targ = Mock()
            t_targ.is_regular = Mock(return_value=is_regular_returns_t_targ)

            t_treat = Mock()
            t_treat.is_regular = Mock(return_value=is_regular_returns_t_treat)

            with expectation as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True,
                    requirements=requirements,
                    data=MockDataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat),
                )
            if excinfo is not None:
                assert "requires regular timeseries" in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "samples_aligned_returns_t_cov, samples_aligned_returns_t_targ, samples_aligned_returns_t_treat, expectation",
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
            self,
            samples_aligned_returns_t_cov,
            samples_aligned_returns_t_targ,
            samples_aligned_returns_t_treat,
            expectation,
        ):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_all_temporal_data_samples_aligned=True)
            )

            t_cov = Mock()
            t_cov.all_samples_aligned = samples_aligned_returns_t_cov

            t_targ = Mock()
            t_targ.all_samples_aligned = samples_aligned_returns_t_targ

            t_treat = Mock()
            t_treat.all_samples_aligned = samples_aligned_returns_t_treat

            with expectation as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True,
                    requirements=requirements,
                    data=MockDataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat),
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
        def test_requires_time_series_index_numeric(
            self, t_cov_index, t_targ_index, t_treat_index, expectation, monkeypatch
        ):
            monkeypatch.setattr(
                "clairvoyance2.interface.requirements.RequirementsChecker._check_data_value_type",
                Mock(),
                raising=True,
            )

            requirements = Requirements(
                dataset_requirements=DatasetRequirements(requires_all_temporal_data_index_numeric=True)
            )

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
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True,
                    requirements=requirements,
                    data=MockDataset(
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

            data_temporal_covariates = Mock()
            data_temporal_covariates.has_missing = temporal_covariates_has_missing

            data_static_covariates = Mock()
            data_static_covariates.has_missing = static_covariates_has_missing

            data_temporal_targets = Mock()
            data_temporal_targets.has_missing = temporal_targets_has_missing

            data_temporal_treatments = Mock()
            data_temporal_treatments.has_missing = temporal_treatments_has_missing

            with expectation as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True,
                    requirements=requirements,
                    data=MockDataset(
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

            data_temporal_covariates_samples = Mock()
            data_temporal_covariates_samples.has_missing = temporal_covariates_has_missing

            with expectation as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True,
                    requirements=requirements,
                    data=MockDataset(temporal_covariates=data_temporal_covariates_samples),
                )
            if excinfo is not None:
                assert "temporal covariates had missing data" in str(excinfo.value).lower()

        @pytest.mark.parametrize(
            "value_type",
            [
                DataValueOpts.NUMERIC_BINARY,
                DataValueOpts.NUMERIC_CATEGORICAL,
            ],
        )
        def test_static_covariates_value_type(self, value_type):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(static_covariates_value_type=value_type),
            )
            mock_static_covariates = Mock(all_features_categorical=False, all_features_binary=False)
            data = MockDataset(
                temporal_covariates=Mock(),
                static_covariates=mock_static_covariates,
            )

            with pytest.raises(RuntimeError) as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data
                )
            assert value_type.name in str(excinfo.value)

        @pytest.mark.parametrize(
            "value_type",
            [
                DataValueOpts.NUMERIC_BINARY,
                DataValueOpts.NUMERIC_CATEGORICAL,
            ],
        )
        def test_temporal_covariates_value_type(self, value_type):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(temporal_covariates_value_type=value_type),
            )
            mock_temporal_covariates = Mock(all_features_categorical=False, all_features_binary=False)
            data = MockDataset(
                temporal_covariates=mock_temporal_covariates,
            )

            with pytest.raises(RuntimeError) as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data
                )
            assert value_type.name in str(excinfo.value)

        @pytest.mark.parametrize(
            "value_type",
            [
                DataValueOpts.NUMERIC_BINARY,
                DataValueOpts.NUMERIC_CATEGORICAL,
            ],
        )
        def test_temporal_targets_value_type(self, value_type):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(temporal_targets_value_type=value_type),
            )
            mock_temporal_targets = Mock(all_features_categorical=False, all_features_binary=False)
            data = MockDataset(
                temporal_covariates=Mock(),
                temporal_targets=mock_temporal_targets,
            )

            with pytest.raises(RuntimeError) as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data
                )
            assert value_type.name in str(excinfo.value)

        @pytest.mark.parametrize(
            "value_type",
            [
                DataValueOpts.NUMERIC_BINARY,
                DataValueOpts.NUMERIC_CATEGORICAL,
            ],
        )
        def test_temporal_treatments_value_type(self, value_type):
            requirements = Requirements(
                dataset_requirements=DatasetRequirements(temporal_treatments_value_type=value_type),
            )
            mock_temporal_treatments = Mock(all_features_categorical=False, all_features_binary=False)
            data = MockDataset(
                temporal_covariates=Mock(),
                temporal_treatments=mock_temporal_treatments,
            )

            with pytest.raises(RuntimeError) as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data
                )
            assert value_type.name in str(excinfo.value)

        class TestWhenPredictionRequirementsProvided:
            def test_called_no_horizon(self, monkeypatch):
                mock_call = Mock()
                monkeypatch.setattr(
                    "clairvoyance2.interface.requirements.RequirementsChecker._check_data_requirements_predict",
                    mock_call,
                    raising=True,
                )
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target_data_structure=DataStructureOpts.TIME_SERIES),
                )

                data = MockDataset(temporal_covariates=Mock())
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data
                )

                mock_call.assert_called_once_with(
                    called_at_fit_time=True, requirements=requirements, data=data, horizon=None
                )

            def test_called_horizon(self, monkeypatch):
                mock_call = Mock()
                monkeypatch.setattr(
                    "clairvoyance2.interface.requirements.RequirementsChecker._check_data_requirements_predict",
                    mock_call,
                    raising=True,
                )
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target_data_structure=DataStructureOpts.TIME_SERIES),
                )
                horizon = Mock()

                data = MockDataset(temporal_covariates=Mock())
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data, horizon=horizon
                )

                mock_call.assert_called_once_with(
                    called_at_fit_time=True, requirements=requirements, data=data, horizon=horizon
                )

        class TestWhenTreatmentEffectsRequirementsProvided:
            def test_called_no_extra_kwargs(self, monkeypatch):
                monkeypatch.setattr(  # Skip prediction requirements checks.
                    "clairvoyance2.interface.requirements.RequirementsChecker._check_data_requirements_predict",
                    Mock(),
                    raising=True,
                )
                mock_call = Mock()
                monkeypatch.setattr(
                    "clairvoyance2.interface.requirements.RequirementsChecker."
                    "_check_data_requirements_predict_counterfactuals",
                    mock_call,
                    raising=True,
                )
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(),
                    treatment_effects_requirements=TreatmentEffectsRequirements(),
                )

                data = MockDataset(temporal_covariates=Mock())
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data
                )

                mock_call.assert_called_once_with(
                    called_at_fit_time=True,
                    requirements=requirements,
                    data=data,
                    sample_index=None,
                    treatment_scenarios=None,
                    horizon=None,
                )

            def test_called_extra_kwargs(self, monkeypatch):
                monkeypatch.setattr(  # Skip prediction requirements checks.
                    "clairvoyance2.interface.requirements.RequirementsChecker._check_data_requirements_predict",
                    Mock(),
                    raising=True,
                )
                mock_call = Mock()
                monkeypatch.setattr(
                    "clairvoyance2.interface.requirements.RequirementsChecker."
                    "_check_data_requirements_predict_counterfactuals",
                    mock_call,
                    raising=True,
                )
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(),
                    treatment_effects_requirements=TreatmentEffectsRequirements(),
                )
                horizon = Mock()

                data = MockDataset(temporal_covariates=Mock())
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=requirements, data=data, horizon=horizon
                )

                mock_call.assert_called_once_with(
                    called_at_fit_time=True,
                    requirements=requirements,
                    data=data,
                    sample_index=None,
                    treatment_scenarios=None,
                    horizon=horizon,
                )

    class TestDataRequirementsPredict:
        class TestTargetTimeSeries:
            def test_fails_no_temporal_targets(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target_data_structure=DataStructureOpts.TIME_SERIES),
                )

                with pytest.raises(RuntimeError) as excinfo:
                    RequirementsChecker.check_data_requirements_predict(
                        requirements, MockDataset(temporal_covariates=Mock()), horizon=Mock()
                    )
                assert "must contain temporal targets" in str(excinfo.value).lower()

            def test_fails_no_horizon_passed_to_check(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(target_data_structure=DataStructureOpts.TIME_SERIES),
                )

                with pytest.raises(RuntimeError) as excinfo:
                    RequirementsChecker.check_data_requirements_predict(
                        requirements, MockDataset(temporal_covariates=Mock(), temporal_targets=Mock()), horizon=None
                    )
                assert "must receive a horizon" in str(excinfo.value).lower()

            @pytest.mark.parametrize(
                "horizon, expectation",
                [
                    (
                        NStepAheadHorizon(1),
                        does_not_raise(),
                    ),
                    (
                        Mock(),
                        pytest.raises(RuntimeError),
                    ),
                ],
            )
            class TestNStepAhead:
                def test_wrong_horizon(self, horizon, expectation):
                    requirements = Requirements(
                        dataset_requirements=DatasetRequirements(),
                        prediction_requirements=PredictionRequirements(
                            target_data_structure=DataStructureOpts.TIME_SERIES, horizon_type=HorizonOpts.N_STEP_AHEAD
                        ),
                    )

                    with expectation as excinfo:
                        RequirementsChecker.check_data_requirements_predict(
                            requirements,
                            MockDataset(
                                temporal_covariates=Mock(n_timesteps_per_sample=[100]),
                                temporal_targets=Mock(n_timesteps_per_sample=[100], all_features_numeric=True),
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
                        target_data_structure=DataStructureOpts.TIME_SERIES, horizon_type=HorizonOpts.N_STEP_AHEAD
                    ),
                )
                data = MockDataset(
                    temporal_covariates=Mock(n_timesteps_per_sample=[t_cov_max_len]),
                    temporal_targets=Mock(n_timesteps_per_sample=[t_targ_max_len]),
                    temporal_treatments=Mock(n_timesteps_per_sample=[t_treat_max_len]),
                )

                with expectation as excinfo:
                    RequirementsChecker.check_data_requirements_general(
                        called_at_fit_time=False,
                        requirements=requirements,
                        data=data,
                        horizon=horizon,
                    )
                if excinfo is not None:
                    assert "horizon must be < max timesteps" in str(excinfo.value).lower()

    class TestDataRequirementsPredictCounterfactuals:
        class TestDataStructureTimeSeries:
            def test_fails_no_temporal_targets(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target_data_structure=DataStructureOpts.TIME_SERIES, horizon_type=HorizonOpts.TIME_INDEX
                    ),
                    treatment_effects_requirements=TreatmentEffectsRequirements(
                        treatment_data_structure=DataStructureOpts.TIME_SERIES
                    ),
                )

                with pytest.raises(RuntimeError) as excinfo:
                    RequirementsChecker.check_data_requirements_predict_counterfactuals(
                        requirements=requirements,
                        data=MockDataset(temporal_covariates=Mock()),
                        horizon=Mock(),
                        sample_index=Mock(),
                        treatment_scenarios=Mock(),
                    )
                assert "must contain temporal targets" in str(excinfo.value).lower()

            def test_fails_no_temporal_treatments(self):
                requirements = Requirements(
                    dataset_requirements=DatasetRequirements(),
                    prediction_requirements=PredictionRequirements(
                        target_data_structure=DataStructureOpts.TIME_SERIES, horizon_type=HorizonOpts.TIME_INDEX
                    ),
                    treatment_effects_requirements=TreatmentEffectsRequirements(
                        treatment_data_structure=DataStructureOpts.TIME_SERIES
                    ),
                )

                with pytest.raises(RuntimeError) as excinfo:
                    RequirementsChecker.check_data_requirements_predict_counterfactuals(
                        requirements=requirements,
                        data=MockDataset(temporal_covariates=Mock(), temporal_targets=Mock()),
                        horizon=Mock(),
                        sample_index=Mock(),
                        treatment_scenarios=Mock(),
                    )
                assert "must contain temporal treatments" in str(excinfo.value).lower()


class TestPredictorModelRequirements:
    def test_prediction_requirements_set(self):
        predictor = Mock()
        predictor.requirements = Requirements(prediction_requirements=None)

        with pytest.raises(RuntimeError) as excinfo:
            RequirementsChecker.check_predictor_model_requirements(predictor)
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
                dataset_requirements=DatasetRequirements(requires_all_temporal_containers_shares_index=True)
            )

            with pytest.raises(RuntimeError) as excinfo:
                RequirementsChecker.check_data_requirements_general(
                    called_at_fit_time=True, requirements=req, data=data
                )
            assert (
                "same time index" in str(excinfo.value)
                and expect_str_1 in str(excinfo.value)
                and expect_str_2 in str(excinfo.value)
            )
