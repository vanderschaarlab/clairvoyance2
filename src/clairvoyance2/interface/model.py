import pprint
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from dotmap import DotMap

from ..data import Dataset, StaticSamples, TimeSeries, TimeSeriesSamples
from ..data.constants import (
    T_ContainerInitializable,
    T_SamplesIndexDtype,
    T_SamplesIndexDtype_AsTuple,
)
from ..utils.common import safe_init_dotmap
from ..utils.dev import function_is_notimplemented_stub, raise_not_implemented
from .horizon import Horizon, TimeIndexHorizon
from .requirements import DataStructureOpts, Requirements, RequirementsChecker

TParams = Dict[str, Any]  # TODO: May constrain this.
TDefaultParams = Union[TParams, NamedTuple]
TPredictOutput = Union[TimeSeriesSamples, StaticSamples]


class BaseModel(ABC):
    requirements: Requirements = Requirements()
    DEFAULT_PARAMS: TDefaultParams = dict()
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        self.params: DotMap = self._process_params(params)
        self.inferred_params: DotMap = safe_init_dotmap(dict(), _dynamic=False)
        self.check_model_requirements()
        self._fit_called = False

    def _process_params(self, params: Optional[TParams]) -> DotMap:
        if self.check_unknown_params is False and len(self.DEFAULT_PARAMS) > 0:
            warnings.warn(
                "`check_unknown_params` was set to False even though `DEFAULT_PARAMS` were explicitly set "
                f"in {self.__class__.__name__}. This could lead to user confusion over parameters, "
                "consider setting `check_unknown_params` to True.",
                category=UserWarning,
            )
        try:
            default_params = self.DEFAULT_PARAMS._asdict()  # type: ignore
        except AttributeError as ex:
            if "_asdict" in str(ex):
                default_params = self.DEFAULT_PARAMS
            else:
                raise
        if params is not None:
            unknown_params = [p for p in params.keys() if p not in default_params]
            if len(unknown_params) > 0 and self.check_unknown_params is True:
                raise ValueError(f"Unknown parameter(s) passed: {unknown_params}")
        copied_params = default_params.copy()
        processed_params: Dict[str, Any] = safe_init_dotmap(copied_params, _dynamic=False)
        if params is not None:
            processed_params.update({k: v for k, v in params.items() if k in default_params})
        return processed_params

    @abstractmethod
    def check_model_requirements(self) -> None:  # pragma: no cover
        ...

    def check_data_requirements_general(self, data: Dataset, **kwargs):
        RequirementsChecker.check_data_requirements_general(self.requirements, data, **kwargs)

    def fit(self, data: Dataset) -> "BaseModel":
        self.check_data_requirements_general(data)
        result = self._fit(data)
        self._fit_called = True
        return result

    @abstractmethod
    def _fit(self, data: Dataset) -> "BaseModel":  # pragma: no cover
        ...

    def _repr_dict(self, dict_: Dict, name: str) -> str:
        tab = "    "
        pp = pprint.PrettyPrinter(indent=4)
        pretty_dict = pp.pformat(dict_).replace("\t", tab)
        dict_prefix = f"{name}="
        dict_repr = f"{dict_prefix}{pretty_dict}"
        dict_repr = tab + f"\n{tab}{' ' * len(dict_prefix)}".join(dict_repr.split("\n"))
        return dict_repr

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(\n"
        params = self._repr_dict(self.params.toDict(), name="params")
        if len(self.inferred_params) > 0:
            inferred_params = "\n" + self._repr_dict(self.inferred_params.toDict(), name="inferred_params")
        else:
            inferred_params = ""
        repr_str += f"{params}{inferred_params}\n)"
        return repr_str


class TransformerModel(BaseModel, ABC):
    def check_data_requirements_transform(self, data: Dataset, **kwargs):
        RequirementsChecker.check_data_requirements_transform(self.requirements, data, **kwargs)

    def transform(self, data: Dataset) -> Dataset:
        if not self._fit_called:
            raise RuntimeError("Must call `fit` before calling `transform`")
        self.check_data_requirements_general(data)
        self.check_data_requirements_transform(data)
        return self._transform(data)

    def inverse_transform(self, data: Dataset) -> Dataset:
        if function_is_notimplemented_stub(self._inverse_transform):
            raise NotImplementedError(f"`_inverse_transform` method was not implemented for {self.__class__.__name__}")
        if not self._fit_called:
            raise RuntimeError("Must call `fit` before calling `transform`")
        self.check_data_requirements_general(data)
        self.check_data_requirements_transform(data)
        return self._inverse_transform(data)

    @abstractmethod
    def _transform(self, data: Dataset) -> Dataset:  # pragma: no cover
        ...

    def _inverse_transform(self, data: Dataset) -> Dataset:
        # Not a mandatory method.
        raise NotImplementedError

    def fit_transform(self, data: Dataset) -> Dataset:
        self.fit(data)
        return self.transform(data)

    def check_model_requirements(self) -> None:
        super().check_model_requirements()
        # Additional requirements for any Transformer:
        if self.requirements.prediction_requirements is not None:
            raise ValueError("Transformer model have PredictionRequirements be None")


# TODO: Unit test once the interface is solidified.
class PredictorModel(BaseModel, ABC):
    def check_data_requirements_predict(self, data: Dataset, horizon: Horizon, **kwargs):
        RequirementsChecker.check_data_requirements_predict(self.requirements, data, horizon, **kwargs)

    def predict(self, data: Dataset, horizon: Horizon) -> TPredictOutput:
        if not self._fit_called:
            raise RuntimeError("Must call `fit` before calling `predict`")
        self.check_data_requirements_general(data, horizon=horizon)
        self.check_data_requirements_predict(data, horizon=horizon)
        return self._predict(data, horizon)

    def fit(self, data: Dataset, horizon: Optional[Horizon] = None) -> "PredictorModel":
        self.check_data_requirements_general(data, horizon=horizon)
        result = self._fit(data, horizon=horizon)
        self._fit_called = True
        return result

    @abstractmethod
    def _fit(self, data: Dataset, horizon: Optional[Horizon] = None) -> "PredictorModel":  # pragma: no cover
        ...

    @abstractmethod
    def _predict(self, data: Dataset, horizon: Horizon) -> TPredictOutput:  # pragma: no cover
        ...

    def fit_predict(self, data: Dataset, horizon: Horizon) -> TPredictOutput:
        self.fit(data)
        return self.predict(data, horizon)

    def check_model_requirements(self) -> None:
        super().check_model_requirements()

        # Additional requirements for any PredictorModel:
        RequirementsChecker.check_predictor_model_requirements(self)


# TODO: Static counterfactual predictions / treatments case TBD.
# TODO: This will likely need an overhaul - to handle more than just "one sample at a time" case.
# NOTE: The Sequence below iterates over the different treatment "cases".
TTreatmentScenariosInitializable = Sequence[Union[T_ContainerInitializable, TimeSeries]]
TTreatmentScenarios = Sequence[TimeSeries]
TCounterfactualPredictions = Sequence[TimeSeries]


class TreatmentEffectsModel(PredictorModel, ABC):
    def check_data_requirements_predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Horizon,
        **kwargs,
    ):
        RequirementsChecker.check_data_requirements_predict_counterfactuals(
            self.requirements,
            data=data,
            sample_index=sample_index,
            treatment_scenarios=treatment_scenarios,
            horizon=horizon,
            **kwargs,
        )

    def predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenariosInitializable,
        horizon: TimeIndexHorizon,
    ) -> TCounterfactualPredictions:
        if not self._fit_called:
            raise RuntimeError("Must call `fit` before calling `predict_counterfactuals`")
        data_processed, treatment_scenarios_processed = self._process_predict_counterfactuals_input(
            data, sample_index=sample_index, treatment_scenarios=treatment_scenarios, horizon=horizon
        )
        self.check_data_requirements_general(
            data_processed,
            horizon=horizon,
            treatment_scenarios=treatment_scenarios_processed,
        )
        self.check_data_requirements_predict_counterfactuals(
            data_processed,
            horizon=horizon,
            sample_index=sample_index,
            treatment_scenarios=treatment_scenarios_processed,
        )
        return self._predict_counterfactuals(data_processed, sample_index, treatment_scenarios_processed, horizon)

    def fit(self, data: Dataset, horizon: Optional[Horizon] = None) -> "TreatmentEffectsModel":
        self.check_data_requirements_general(data, horizon=horizon)
        result = self._fit(data, horizon=horizon)
        self._fit_called = True
        return result

    @abstractmethod
    def _fit(self, data: Dataset, horizon: Optional[Horizon] = None) -> "TreatmentEffectsModel":  # pragma: no cover
        ...

    # TODO: Simplify this or move elsewhere. Test.
    def _process_predict_counterfactuals_input(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenariosInitializable,
        horizon: TimeIndexHorizon,
    ) -> Tuple[Dataset, Sequence[TimeSeries]]:
        if not isinstance(sample_index, T_SamplesIndexDtype_AsTuple):
            raise ValueError(
                f"Expected sample index to be one of {T_SamplesIndexDtype_AsTuple} " f"but was {type(sample_index)}"
            )
        if sample_index not in data.sample_indices:
            raise ValueError(f"Sample with index {sample_index} not found in data")
        if len(horizon.time_index_sequence) != 1:
            raise ValueError(
                "Time index sequence specified in the time index horizon must contain exactly one "
                "time index when predicting counterfactuals for a specific sample, "
                f"but {len(horizon.time_index_sequence)}-many time indexes were found"
            )
        if len(treatment_scenarios) == 0:
            raise ValueError("Must provide at least one treatment scenario")
        assert self.requirements.treatment_effects_requirements is not None
        if self.requirements.treatment_effects_requirements.treatment_data_structure == DataStructureOpts.TIME_SERIES:
            horizon_time_index = horizon.time_index_sequence[0]
            expect_timesteps = len(treatment_scenarios[0])
            if not all(len(t) == expect_timesteps for t in treatment_scenarios):
                raise ValueError("All treatment scenarios must have the same number of timesteps but did not")
            if data.temporal_treatments is None:
                raise ValueError("`temporal_treatments` must be specified in data but was None")
            template_ts = data.temporal_treatments[sample_index]
            assert isinstance(template_ts, TimeSeries)
            list_ts: List[TimeSeries] = []
            for treatment_scenario in treatment_scenarios:
                if isinstance(treatment_scenario, TimeSeries):
                    treatment_scenario_df = treatment_scenario.df
                if isinstance(treatment_scenario, np.ndarray):
                    treatment_scenario_df = pd.DataFrame(
                        data=treatment_scenario, columns=template_ts.df.columns, index=horizon_time_index
                    )
                else:
                    treatment_scenario_df = treatment_scenario
                assert isinstance(treatment_scenario_df, pd.DataFrame)
                if list(treatment_scenario_df.index) != list(horizon_time_index):
                    raise ValueError(
                        f"Unexpected time index in treatment scenarios, expected {horizon_time_index} "
                        f"found {treatment_scenario_df.index}."
                    )
                treatment_scenario_ts = TimeSeries.new_like(like=template_ts, data=treatment_scenario_df)
                list_ts.append(treatment_scenario_ts)
            return data[sample_index], list_ts
        else:
            raise_not_implemented(f"predict_counterfactuals() for non-{str(DataStructureOpts.TIME_SERIES)}")

    @abstractmethod
    def _predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: TimeIndexHorizon,
    ) -> TCounterfactualPredictions:
        ...

    def check_model_requirements(self) -> None:
        super().check_model_requirements()

        # Additional requirements for any TreatmentEffectsModel:
        RequirementsChecker.check_treatment_effects_model_requirements(self)
