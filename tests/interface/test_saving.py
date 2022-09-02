import os

import pytest
import torch
import torch.nn as nn
from dotmap import DotMap

from clairvoyance2.interface import BaseModel, SavableModelMixin, SavableTorchModelMixin


class TestIntegration:
    class TestSavableModel:
        @pytest.mark.parametrize(
            "init_params",
            [
                dict(),
                {"param_1": 10},
                {"param_1": 20, "pram_2": [1, 2, 3]},
            ],
        )
        def test_save_load(self, init_params, tmpdir):
            class MyModel(SavableModelMixin, BaseModel):
                DEFAULT_PARAMS = init_params

                def _fit(self, data) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    pass

            my_model = MyModel(params=init_params)

            path = os.path.join(tmpdir, "my_model.p")
            my_model.save(path)

            loaded_model = MyModel.load(path)

            assert os.path.exists(os.path.join(tmpdir, "my_model.p.params"))
            assert loaded_model.params == my_model.params
            assert loaded_model.inferred_params == my_model.inferred_params
            assert loaded_model.inferred_params == dict()

        @pytest.mark.parametrize(
            "inferred_params",
            [
                dict(),
                {"param_1": 10},
                {"param_1": 20, "pram_2": [1, 2, 3]},
            ],
        )
        def test_save_load_modify_inferred_params(self, inferred_params, tmpdir):
            class MyModel(SavableModelMixin, BaseModel):
                DEFAULT_PARAMS = {"a": 1}

                def _fit(self, data) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    pass

            my_model = MyModel(params={"a": 1})
            my_model.inferred_params = DotMap(inferred_params, _dynamic=False)

            path = os.path.join(tmpdir, "my_model.p")
            my_model.save(path)

            loaded_model = MyModel.load(path)

            assert os.path.exists(os.path.join(tmpdir, "my_model.p.params"))
            assert loaded_model.params == {"a": 1}
            assert loaded_model.inferred_params == my_model.inferred_params

        @pytest.mark.parametrize(
            "path",
            ["./dir/", "/home/dir/"],
        )
        def test_wrong_kind_of_path(self, path):
            class MyModel(SavableModelMixin, BaseModel):
                DEFAULT_PARAMS = {"a": 1}

                def _fit(self, data) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    pass

            my_model = MyModel(params={"a": 1})

            with pytest.raises(ValueError) as excinfo:
                my_model.save(path)
            assert "path" in str(excinfo.value)

    class TestSavableTorchModel:
        def test_save_load_no_init_submodules(self, tmpdir):
            # Arrange:
            class MyModel(SavableTorchModelMixin, BaseModel, nn.Module):
                DEFAULT_PARAMS = {"param_1": 0}

                def __init__(self, params) -> None:
                    BaseModel.__init__(self, params)
                    nn.Module.__init__(self)
                    self.inner = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

                def _fit(self, data) -> "MyModel":
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

                def _fit(self, data) -> "MyModel":
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
