from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NoReturn, Optional, Sequence

from ..data import Dataset, FeatureType, TimeSeries
from ..data.constants import T_NumericDtype_AsTuple
from ..utils.common import python_type_from_np_pd_dtype
from ..utils.dev import raise_not_implemented
from .horizon import Horizon, HorizonType, NStepAheadHorizon


class PredictionTaskType(Enum):
    REGRESSION = auto()
    CLASSIFICATION = auto()


class PredictionTargetType(Enum):
    TIME_SERIES = auto()
    STATIC = auto()


class TreatmentType(Enum):
    TIME_SERIES = auto()
    STATIC = auto()


class TreatmentValueType(Enum):
    CONTINUOUS = auto()
    CATEGORICAL = auto()
    BINARY = auto()
    # NOTE: ^ In theory, binary is just a subset of categorical, but for the sake of simplicity, make it separate.


@dataclass(frozen=True)
class DatasetRequirements:
    requires_static_samples_present: bool = False
    requires_time_series_samples_aligned: bool = False
    requires_time_series_samples_regular: bool = False
    requires_time_series_index_numeric: bool = False
    requires_temporal_containers_have_same_time_index: bool = False
    requires_no_missing_data: bool = False
    requires_all_numeric_features: bool = False


@dataclass(frozen=True)
class PredictionRequirements:
    task: PredictionTaskType = PredictionTaskType.REGRESSION
    target: PredictionTargetType = PredictionTargetType.TIME_SERIES
    horizon: HorizonType = HorizonType.N_STEP_AHEAD


@dataclass(frozen=True)
class TreatmentEffectsRequirements:
    # NOTE: task, target, horizon are expected to be the same as for PredictionRequirements.
    treatment_type: TreatmentType = TreatmentType.TIME_SERIES
    treatment_value_type: TreatmentValueType = TreatmentValueType.BINARY


@dataclass(frozen=True)
class Requirements:
    dataset_requirements: DatasetRequirements = DatasetRequirements()
    prediction_requirements: Optional[PredictionRequirements] = None
    treatment_effects_requirements: Optional[TreatmentEffectsRequirements] = None


def raise_requirements_mismatch_error(requirement_name: str, explanation_text: str) -> NoReturn:
    raise RuntimeError(f"Requirements mismatch occurred. Requirement: '{requirement_name}'. {explanation_text}.")


def get_container_friendly_name(container_name: str) -> str:
    return container_name.replace("_", " ")


# NOTE: Needs more testing.
class RequirementsChecker:
    @staticmethod
    def check_data_requirements(requirements: Requirements, data: Dataset, **kwargs):
        # General data requirements:
        if requirements.dataset_requirements.requires_static_samples_present:
            if data.static_covariates is None:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires static samples", "Dataset did not have static samples"
                )
        if requirements.dataset_requirements.requires_no_missing_data:
            for container_name, container in data.all_data_containers.items():
                if container.has_missing:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires no missing data",
                        f"Dataset {get_container_friendly_name(container_name)} had missing data",
                    )
        if requirements.dataset_requirements.requires_time_series_samples_regular:
            for container_name, container in data.temporal_data_containers.items():
                is_regular, _ = container.is_regular()
                # TODO: Compare the diff. and ensure they are the same?
                if not is_regular:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires regular timeseries",
                        f"Dataset {get_container_friendly_name(container_name)} did not have a regular time index",
                    )
        if requirements.dataset_requirements.requires_time_series_samples_aligned:
            for container_name, container in data.temporal_data_containers.items():
                if not container.is_aligned():
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires aligned timeseries",
                        f"Dataset {get_container_friendly_name(container_name)} were not all aligned by their index",
                    )
        if requirements.dataset_requirements.requires_time_series_index_numeric:
            acceptable_types = T_NumericDtype_AsTuple
            for container_name, container in data.temporal_data_containers.items():
                if len(container) > 0:
                    ts = container[0]
                    if TYPE_CHECKING:
                        assert isinstance(ts, TimeSeries)
                    dtype = python_type_from_np_pd_dtype(ts.time_index.dtype)
                    if dtype not in acceptable_types:
                        raise_requirements_mismatch_error(
                            "Dataset requirement: requires numeric timeseries index",
                            f"Dataset {get_container_friendly_name(container_name)} had index of dtype {dtype}",
                        )
        if requirements.dataset_requirements.requires_all_numeric_features:
            # If there is a treatment effects requirements for treatment value type categorical/binary,
            # do not enforce that treatment covariates should be numeric features, even if
            # `requires_all_numeric_features` is True.
            # TODO: This needs to be revised, as it is convoluted and non-obvious.
            do_not_check_treatments = (
                requirements.treatment_effects_requirements is not None
                and requirements.treatment_effects_requirements.treatment_value_type
                in (TreatmentValueType.BINARY, TreatmentValueType.CATEGORICAL)
            )
            for container_name, container in data.all_data_containers.items():
                if not container.all_numeric_compatible_features:
                    if not (do_not_check_treatments and "container_name" in ("temporal_treatments",)):
                        raise_requirements_mismatch_error(
                            "Dataset requirement: requires all numeric features",
                            f"Non-numeric features found in {get_container_friendly_name(container_name)}",
                        )
        if requirements.dataset_requirements.requires_temporal_containers_have_same_time_index:
            check_outcome, names = data.check_temporal_containers_have_same_time_index()
            if check_outcome is False:
                assert names is not None
                a_name, b_name = names
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires all temporal containers have same time index (for each sample)",
                    f"The containers {a_name} and {b_name} did not have the same time index for all samples",
                )

        # Prediction-specific data requirements:
        if requirements.prediction_requirements is not None:

            # PredictionTargetType.TIME_SERIES:
            if requirements.prediction_requirements.target == PredictionTargetType.TIME_SERIES:
                if data.temporal_targets is None:
                    raise_requirements_mismatch_error(
                        f"Prediction requirement: prediction target `{PredictionTargetType.TIME_SERIES}`",
                        "Dataset must contain temporal targets in this case but did not",
                    )
                if "horizon" not in kwargs:
                    raise_requirements_mismatch_error(
                        f"Prediction requirement: prediction target `{PredictionTargetType.TIME_SERIES}`",
                        "A prediction horizon must be passed to prediction methods, but found None",
                    )
                horizon: Horizon = kwargs["horizon"]

                # PredictionTargetType.TIME_SERIES > PredictionHorizonType.N_STEP_AHEAD:
                if requirements.prediction_requirements.horizon == HorizonType.N_STEP_AHEAD:
                    if not isinstance(horizon, NStepAheadHorizon):
                        raise_requirements_mismatch_error(
                            f"Prediction requirement: prediction horizon `{HorizonType.N_STEP_AHEAD}`",
                            f"A prediction horizon of type {NStepAheadHorizon} is expected, but found {type(horizon)}",
                        )
                    for container_name, container in data.temporal_data_containers.items():
                        len_ = max(container.n_timesteps_per_sample)
                        if horizon.n_step >= len_:
                            raise_requirements_mismatch_error(
                                f"Prediction requirement: prediction horizon `{HorizonType.N_STEP_AHEAD}`",
                                "N step ahead horizon must be < max timesteps in "
                                f"{get_container_friendly_name(container_name)}, but was "
                                f"{horizon.n_step} >= {len_}",
                            )

                # PredictionTargetType.TIME_SERIES > PredictionHorizonType.TIME_INDEX:
                if requirements.prediction_requirements.horizon == HorizonType.TIME_INDEX:
                    # TODO: Implement data requirements checks for this case!
                    pass

        # Treatment effects -specific data requirements:
        if requirements.treatment_effects_requirements is not None:
            # TreatmentType.TIME_SERIES:
            if requirements.treatment_effects_requirements.treatment_type == TreatmentType.TIME_SERIES:
                if data.temporal_targets is None:
                    raise_requirements_mismatch_error(
                        f"Treatment effects requirements: treatment type `{TreatmentType.TIME_SERIES}`",
                        "Dataset must contain temporal targets in this case but did not",
                    )
                if data.temporal_treatments is None:
                    raise_requirements_mismatch_error(
                        f"Treatment effects requirements: treatment type `{TreatmentType.TIME_SERIES}`",
                        "Dataset must contain temporal treatments in this case but did not",
                    )
                if requirements.treatment_effects_requirements.treatment_value_type in (
                    TreatmentValueType.BINARY,
                    TreatmentValueType.CATEGORICAL,
                ):
                    if not data.temporal_treatments.all_categorical_features:
                        non_categorical_features = [
                            k
                            for k, f in data.temporal_treatments.features.items()
                            if f.feature_type != FeatureType.CATEGORICAL
                        ]
                        raise_requirements_mismatch_error(
                            "Treatment effects requirements: treatment value type "
                            f"`{requirements.treatment_effects_requirements.treatment_value_type}`",
                            "Temporal treatments must all be categorical features, but some were not:\n"
                            f"{non_categorical_features}",
                        )
                if requirements.treatment_effects_requirements.treatment_value_type == TreatmentValueType.BINARY:
                    for ts in data.temporal_treatments:
                        for col in ts.df:
                            if tuple(sorted(ts.df[col].unique())) != (0.0, 1.0):
                                raise_requirements_mismatch_error(
                                    "Treatment effects requirements: treatment value type "
                                    f"`{TreatmentValueType.BINARY}`",
                                    "Temporal treatments must only contain values (0., 1.), "
                                    "but other values were found",
                                )
            if "treatment_scenarios" in kwargs:
                # This code path will occur when .predict_counterfactuals() is called.
                # NOTE: This will likely be revised.
                treatment_scenarios = kwargs["treatment_scenarios"]
                if requirements.treatment_effects_requirements.treatment_type == TreatmentType.TIME_SERIES:
                    assert isinstance(treatment_scenarios, Sequence)
                    for ts in treatment_scenarios:
                        assert isinstance(ts, TimeSeries)
                if requirements.treatment_effects_requirements.treatment_value_type == TreatmentValueType.BINARY:
                    for ts in treatment_scenarios:
                        for col in ts.df:
                            assert tuple(sorted(col.unique())) == (0.0, 1.0)

    @staticmethod
    def check_prediction_requirements(predictor):
        requirements: Requirements = predictor.requirements
        if requirements.prediction_requirements is None:
            raise_requirements_mismatch_error(
                "Prediction requirements",
                f"Prediction model {predictor.__class__.__name__} must have prediction requirements defined, "
                "but found None",
            )
        # NOTE: The below NotImplemented errors are to be removed as appropriate models are added.
        if requirements.prediction_requirements.target != PredictionTargetType.TIME_SERIES:  # pragma: no cover
            raise_not_implemented(f"Prediction target `{requirements.prediction_requirements.target}`")
        if requirements.prediction_requirements.horizon not in (
            HorizonType.N_STEP_AHEAD,
            HorizonType.TIME_INDEX,
        ):  # pragma: no cover
            raise_not_implemented(f"Prediction horizon `{requirements.prediction_requirements.horizon}`")

    @staticmethod
    def check_treatment_effects_requirements(treatment_effects_model):
        requirements: Requirements = treatment_effects_model.requirements
        if requirements.treatment_effects_requirements is None:
            raise_requirements_mismatch_error(
                "Treatment effects requirements",
                f"Treatment effects model {treatment_effects_model.__class__.__name__} must have treatment effects "
                "requirements defined, but found None",
            )
        # NOTE: The below NotImplemented errors are to be removed as appropriate models are added.
        if requirements.treatment_effects_requirements.treatment_type != TreatmentType.TIME_SERIES:  # pragma: no cover
            raise_not_implemented(f"Treatment type `{requirements.treatment_effects_requirements.treatment_type}`")
        if (
            requirements.treatment_effects_requirements.treatment_value_type != TreatmentValueType.CATEGORICAL
        ):  # pragma: no cover
            raise_not_implemented(
                f"Treatment value type `{requirements.treatment_effects_requirements.treatment_value_type}`"
            )
