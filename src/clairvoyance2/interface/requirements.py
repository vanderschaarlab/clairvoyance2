from dataclasses import dataclass
from enum import Enum, auto
from typing import NoReturn, Optional

from ..data import Dataset, StaticSamples, TimeSeriesSamples
from ..data.utils import python_type_from_np_pd_dtype
from ..utils.dev import raise_not_implemented
from .prediction_horizon import NStepAheadHorizon, THorizon


class PredictionTask(Enum):
    REGRESSION = auto()
    CLASSIFICATION = auto()


class PredictionTarget(Enum):
    TIME_SERIES = auto()
    STATIC = auto()


class PredictionHorizon(Enum):
    N_STEP_AHEAD = auto()
    ARBITRARY_TIME_INDEX = auto()
    # Other ideas: N_STEP_FORECAST = auto()


@dataclass(frozen=True)
class DatasetRequirements:
    requires_static_samples_present: bool = False
    requires_time_series_samples_aligned: bool = False
    requires_time_series_samples_regular: bool = False
    requires_time_series_index_numeric: bool = False
    requires_no_missing_data: bool = False
    requires_all_numeric_features: bool = False


@dataclass(frozen=True)
class PredictionRequirements:
    task: PredictionTask = PredictionTask.REGRESSION
    target: PredictionTarget = PredictionTarget.TIME_SERIES
    horizon: PredictionHorizon = PredictionHorizon.N_STEP_AHEAD


@dataclass(frozen=True)
class Requirements:
    dataset_requirements: DatasetRequirements = DatasetRequirements()
    prediction_requirements: Optional[PredictionRequirements] = None


def raise_requirements_mismatch_error(requirement_name: str, explanation_text: str) -> NoReturn:
    raise RuntimeError(f"Requirements mismatch occurred. Requirement: '{requirement_name}'. {explanation_text}.")


# NOTE: Needs testing.
class RequirementsChecker:
    @staticmethod
    def check_data_requirements(requirements: Requirements, data: Dataset, **kwargs):
        temporal_covariates: TimeSeriesSamples
        static_covariates: Optional[StaticSamples]
        temporal_targets: Optional[TimeSeriesSamples]
        temporal_covariates, static_covariates, temporal_targets = data

        # General data requirements:
        if requirements.dataset_requirements.requires_static_samples_present:
            if static_covariates is None:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires static samples", "Dataset did not have static samples"
                )
        if requirements.dataset_requirements.requires_no_missing_data:
            time_series_samples_has_missing = temporal_covariates.has_missing
            static_samples_has_missing = static_covariates.has_missing if static_covariates is not None else None
            temporal_targets_has_missing = temporal_targets.has_missing if temporal_targets is not None else None
            if time_series_samples_has_missing is True:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires no missing data",
                    "Dataset temporal covariates had missing data",
                )
            if static_samples_has_missing is True:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires no missing data",
                    "Dataset static samples had missing data",
                )
            if temporal_targets_has_missing is True:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires no missing data",
                    "Dataset temporal targets had missing data",
                )
        if requirements.dataset_requirements.requires_time_series_samples_regular:
            temporal_covariates_is_regular, _ = temporal_covariates.is_regular()
            if not temporal_covariates_is_regular:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires regular timeseries",
                    "Dataset temporal covariates did not have a regular time index",
                )
            if temporal_targets is not None:
                temporal_targets_is_regular, _ = temporal_targets.is_regular()
                if not temporal_targets_is_regular:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires regular timeseries",
                        "Dataset temporal targets did not have a regular time index",
                    )
        if requirements.dataset_requirements.requires_time_series_samples_aligned:
            # TODO: Compare the diff. and ensure they are the same?
            if not temporal_covariates.is_aligned():
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires aligned timeseries",
                    "Dataset temporal covariates were not all aligned by their index",
                )
            if temporal_targets is not None and not temporal_targets.is_aligned():
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires aligned timeseries",
                    "Dataset temporal targets were not all aligned by their index",
                )
        if requirements.dataset_requirements.requires_time_series_index_numeric:
            acceptable_types = (int, float)
            if len(temporal_covariates) > 0:
                dtype = python_type_from_np_pd_dtype(temporal_covariates[0].df.index.dtype)
                if dtype not in acceptable_types:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires numeric timeseries index",
                        f"Dataset temporal covariates had index of dtype {dtype}",
                    )
            if temporal_targets is not None and len(temporal_targets) > 0:
                dtype = python_type_from_np_pd_dtype(temporal_targets[0].df.index.dtype)
                if dtype not in acceptable_types:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires numeric timeseries index",
                        f"Dataset temporal targets had index of dtype {dtype}",
                    )
        if requirements.dataset_requirements.requires_all_numeric_features:
            all_numeric = all(
                d.all_numeric_features
                for d in (temporal_covariates, static_covariates, temporal_targets)
                if d is not None
            )
            if not all_numeric:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires all numeric features",
                    "Dataset contained some non-numeric features",
                )

        # Prediction-specific data requirements:
        if requirements.prediction_requirements is not None:
            # PredictionTarget.TIME_SERIES:
            if requirements.prediction_requirements.target == PredictionTarget.TIME_SERIES:
                if data.temporal_targets is None:
                    raise_requirements_mismatch_error(
                        f"Prediction requirement: prediction target `{PredictionTarget.TIME_SERIES}`",
                        "Dataset must contain temporal targets in this case but did not",
                    )
                if "horizon" not in kwargs:
                    raise_requirements_mismatch_error(
                        f"Prediction requirement: prediction target `{PredictionTarget.TIME_SERIES}`",
                        "A prediction horizon must be passed to prediction methods, but found None",
                    )
                horizon: THorizon = kwargs["horizon"]
                # PredictionTarget.TIME_SERIES > PredictionTarget.N_STEP_AHEAD:
                if requirements.prediction_requirements.horizon == PredictionHorizon.N_STEP_AHEAD:
                    if not isinstance(horizon, NStepAheadHorizon):
                        raise_requirements_mismatch_error(
                            f"Prediction requirement: prediction horizon `{PredictionHorizon.N_STEP_AHEAD}`",
                            f"A prediction horizon of type {NStepAheadHorizon} is expected, but found {type(horizon)}",
                        )
                    len_ = max(data.temporal_covariates.n_timesteps_per_sample)
                    if horizon.n_step >= len_:
                        raise_requirements_mismatch_error(
                            f"Prediction requirement: prediction horizon `{PredictionHorizon.N_STEP_AHEAD}`",
                            "N step ahead horizon must be < max timesteps in covariates, but was "
                            f"{horizon.n_step} >= {len_}",
                        )
                    len_ = max(data.temporal_targets.n_timesteps_per_sample)
                    if horizon.n_step >= len_:
                        raise_requirements_mismatch_error(
                            f"Prediction requirement: prediction horizon `{PredictionHorizon.N_STEP_AHEAD}`",
                            "N step ahead horizon must be < max timesteps in targets, but was "
                            f"{horizon.n_step} >= {len_}",
                        )

    @staticmethod
    def check_prediction_requirements(predictor):
        requirements: Requirements = predictor.requirements
        if requirements.prediction_requirements is None:
            raise_requirements_mismatch_error(
                "Prediction requirements",
                f"Prediction model {predictor.__class__.__name__} must have prediction requirements defined, "
                "but found None",
            )
        if requirements.prediction_requirements.task != PredictionTask.REGRESSION:  # pragma: no cover
            raise_not_implemented(f"Prediction task `{requirements.prediction_requirements.task}`")
        if requirements.prediction_requirements.target != PredictionTarget.TIME_SERIES:  # pragma: no cover
            raise_not_implemented(f"Prediction target `{requirements.prediction_requirements.target}`")
        if requirements.prediction_requirements.horizon != PredictionHorizon.N_STEP_AHEAD:  # pragma: no cover
            raise_not_implemented(f"Prediction horizon `{requirements.prediction_requirements.horizon}`")
