import os

import torch
import torch.nn as nn

from clairvoyance2.components.torch.interfaces import SavableTorchModelMixin
from clairvoyance2.interface import BaseModel


class TestIntegration:
    class TestSavableTorchModel:
        def test_save_load_no_init_submodules(self, tmpdir):
            # Arrange:
            class MyModel(SavableTorchModelMixin, BaseModel, nn.Module):
                DEFAULT_PARAMS = {"param_1": 0}

                def __init__(self, params) -> None:
                    BaseModel.__init__(self, params)
                    nn.Module.__init__(self)
                    self.inner = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

                def _fit(self, data, **kwargs) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    pass

            my_model = MyModel(params={"param_1": 20})
            my_model.inferred_params.ip_1 = 10
            torch.nn.init.xavier_uniform_(my_model.inner[0].weight)
            torch.nn.init.zeros_(my_model.inner[2].weight)

            # Act:
            path = os.path.join(tmpdir, "my_model.p")
            my_model.save(path)
            loaded_model = MyModel.load(path)

            # Assert:
            assert os.path.exists(os.path.join(tmpdir, "my_model.p.params"))
            assert os.path.exists(os.path.join(tmpdir, "my_model.p"))
            assert loaded_model.params == my_model.params
            assert loaded_model.inferred_params == my_model.inferred_params
            assert loaded_model.inferred_params != dict()
            assert loaded_model.inferred_params.ip_1 == 10
            assert len(list(loaded_model.parameters())) > 0
            for p_original, p_loaded in zip(my_model.parameters(), loaded_model.parameters()):
                assert (p_original == p_loaded).all()

        def test_save_load_has_init_submodules(self, tmpdir):
            # Arrange:
            class MyModel(SavableTorchModelMixin, BaseModel, nn.Module):
                DEFAULT_PARAMS = {"param_1": 0}

                def __init__(self, params) -> None:
                    BaseModel.__init__(self, params)
                    nn.Module.__init__(self)
                    self.inner = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
                    self.inner2 = None

                def _init_submodules(self):
                    self.inner2 = nn.Linear(self.inferred_params.inner2_in, 1)

                def _fit(self, data, **kwargs) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    pass

            my_model = MyModel(params={"param_1": 20})
            my_model.inferred_params.inner2_in = 7
            my_model._init_submodules()  # pylint: disable=protected-access
            torch.nn.init.xavier_uniform_(my_model.inner[0].weight)
            torch.nn.init.zeros_(my_model.inner[2].weight)
            torch.nn.init.ones_(my_model.inner2.weight)

            # Act:
            path = os.path.join(tmpdir, "my_model.p")
            my_model.save(path)
            loaded_model = MyModel.load(path)

            # Assert:
            assert os.path.exists(os.path.join(tmpdir, "my_model.p.params"))
            assert os.path.exists(os.path.join(tmpdir, "my_model.p"))
            assert loaded_model.params == my_model.params
            assert loaded_model.inferred_params == my_model.inferred_params
            assert loaded_model.inferred_params != dict()
            assert loaded_model.inferred_params.inner2_in == 7
            assert len(list(loaded_model.parameters())) > 0
            assert loaded_model.inferred_params is not None
            for p_original, p_loaded in zip(my_model.parameters(), loaded_model.parameters()):
                assert (p_original == p_loaded).all()
