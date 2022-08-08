import pprint
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Union

from dotmap import DotMap

from ..data import Dataset
from ..utils.dev import function_is_notimplemented_stub
from .prediction_horizon import THorizon
from .requirements import Requirements, RequirementsChecker

TParams = Dict[str, Any]  # TODO: May constrain this.
TDefaultParams = Union[TParams, NamedTuple]


class BaseModel(ABC):
    requirements: Requirements = Requirements()
    DEFAULT_PARAMS: TDefaultParams = dict()
    check_unknown_params: bool = True

    def __init__(self, params: Optional[TParams] = None) -> None:
        self.params: DotMap = self._process_params(params)
        self.check_model_requirements()
        self._fit_called = False
        super().__init__()

    def _process_params(self, params: Optional[TParams]) -> DotMap:
        # TODO: Unit test this function.
        if self.check_unknown_params is False and len(self.DEFAULT_PARAMS) > 0:
            warnings.warn(
                "`check_unknown_params` was set to False even though `DEFAULT_PARAMS` were explicitly set "
                f"in {self.__class__.__name__}. This could lead to user confusion over parameters, "
                "consider setting `check_unknown_params` to True.",
                category=UserWarning,
            )
        try:
            default_params = self.DEFAULT_PARAMS._asdict()  # type: ignore
        except AttributeError:
            default_params = self.DEFAULT_PARAMS
        if params is None:
            copied_params = default_params.copy()
            processed_params: Dict[str, Any] = (
                DotMap(copied_params) if not isinstance(copied_params, DotMap) else copied_params
            )
        else:
            unknown_params = [p for p in params.keys() if p not in default_params]
            if len(unknown_params) > 0 and self.check_unknown_params is True:
                raise ValueError(f"Unknown parameter(s) passed: {unknown_params}")
            copied_params = default_params.copy()
            processed_params = DotMap(copied_params) if not isinstance(copied_params, DotMap) else copied_params
            processed_params.update({k: v for k, v in params.items() if k in default_params})
        return processed_params

    @abstractmethod
    def check_model_requirements(self) -> None:  # pragma: no cover
        ...

    def check_data_requirements(self, data: Dataset, **kwargs):
        RequirementsChecker.check_data_requirements(self.requirements, data, **kwargs)

    def fit(self, data: Dataset) -> "BaseModel":
        self.check_data_requirements(data)
        result = self._fit(data)
        self._fit_called = True
        return result

    @abstractmethod
    def _fit(self, data: Dataset) -> "BaseModel":  # pragma: no cover
        ...

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(\n"
        tab = "    "
        pp = pprint.PrettyPrinter(indent=4)

        pretty_params = pp.pformat(self.params.toDict()).replace("\t", tab)
        params_prefix = "params="
        params = f"{params_prefix}{pretty_params}"
        params = tab + f"\n{tab}{' ' * len(params_prefix)}".join(params.split("\n"))

        repr_str += f"{params}\n)"

        return repr_str


class TransformerModel(BaseModel, ABC):
    def transform(self, data: Dataset) -> Dataset:
        if not self._fit_called:
            raise RuntimeError("Must call `fit` before calling `transform`")
        self.check_data_requirements(data)
        return self._transform(data)

    def inverse_transform(self, data: Dataset) -> Dataset:
        if function_is_notimplemented_stub(self._inverse_transform):
            raise NotImplementedError(f"`_inverse_transform` method was not implemented for {self.__class__.__name__}")
        if not self._fit_called:
            raise RuntimeError("Must call `fit` before calling `transform`")
        self.check_data_requirements(data)
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
    def predict(self, data: Dataset, horizon: THorizon = None) -> Dataset:
        if not self._fit_called:
            raise RuntimeError("Must call `fit` before calling `predict`")
        self.check_data_requirements(data, horizon=horizon)
        return self._predict(data, horizon)

    def fit(self, data: Dataset, horizon: THorizon = None) -> "PredictorModel":
        self.check_data_requirements(data, horizon=horizon)
        result = self._fit(data, horizon=horizon)
        self._fit_called = True
        return result

    @abstractmethod
    def _fit(self, data: Dataset, horizon: THorizon = None) -> "PredictorModel":  # pragma: no cover
        ...

    @abstractmethod
    def _predict(self, data: Dataset, horizon: THorizon) -> Dataset:  # pragma: no cover
        ...

    def fit_predict(self, data: Dataset, horizon: THorizon = None) -> Dataset:
        self.fit(data)
        return self.predict(data, horizon)

    def check_model_requirements(self) -> None:
        super().check_model_requirements()

        # Additional requirements for any PredictorModel:
        RequirementsChecker.check_prediction_requirements(self)
