import os
import pickle
from typing import Callable

import torch
from dotmap import DotMap

from ..utils.common import safe_init_dotmap

# TODO: Alternatively save parameters as JSON?


class SavableModelMixin:
    params: DotMap
    inferred_params: DotMap

    @staticmethod
    def _validate_path(path: str) -> None:
        basename = os.path.basename(path)
        if len(basename) == 0:
            raise ValueError(f"`path` must be a path with a basename but was {path}")

    @staticmethod
    def _get_params_file_path(path: str):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        return os.path.join(dirname, f"{basename}.params")

    def save(self, path: str) -> None:
        self._validate_path(path)
        params_file_path = self._get_params_file_path(path)
        to_save_in_params_file = {
            "params": self.params.toDict(),
            "inferred_params": self.inferred_params.toDict(),
        }
        with open(params_file_path, "wb") as f:
            pickle.dump(to_save_in_params_file, f)

    @classmethod
    def load(cls, path: str):
        SavableModelMixin._validate_path(path)
        params_file_path = SavableModelMixin._get_params_file_path(path)
        with open(params_file_path, "rb") as f:
            loaded_from_params_file = pickle.load(f)
        params = safe_init_dotmap(loaded_from_params_file["params"], _dynamic=False)
        inferred_params = safe_init_dotmap(loaded_from_params_file["inferred_params"], _dynamic=False)
        new = cls(params=params)  # type: ignore
        # NOTE: ^ This Mixin assumes that the class is BaseModel-like, so it's expected to have an initialization like:
        # __init___(self, params).
        new.inferred_params = inferred_params
        return new


class SavableTorchModelMixin(SavableModelMixin):
    state_dict: Callable
    load_state_dict: Callable
    _init_submodules: Callable[[], None]

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str):
        # Load `params` and `inferred params`:
        loaded = super().load(path)

        # Run _init_submodules() if our model provides this method.
        has_init_submodules_method = False
        try:
            _ = loaded._init_submodules  # pylint: disable=protected-access
            has_init_submodules_method = True
        except AttributeError:
            pass
        if has_init_submodules_method:
            loaded._init_submodules()  # pylint: disable=protected-access

        # Finally, load the state dict.
        loaded.load_state_dict(torch.load(path))
        return loaded
