import os

import pytest
from dotmap import DotMap

from clairvoyance2.interface import BaseModel, SavableModelMixin


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

                def _fit(self, data, **kwargs) -> "MyModel":
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

                def _fit(self, data, **kwargs) -> "MyModel":
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

                def _fit(self, data, **kwargs) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    pass

            my_model = MyModel(params={"a": 1})

            with pytest.raises(ValueError) as excinfo:
                my_model.save(path)
            assert "path" in str(excinfo.value)
