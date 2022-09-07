from typing import Dict, Mapping, NamedTuple
from unittest.mock import Mock

import pytest
from dotmap import DotMap

from clairvoyance2.data import Dataset
from clairvoyance2.interface import BaseModel, TransformerModel

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def mock_model_for_process_params():
    class MyModel(BaseModel):
        mock_check_model_requirements = Mock()
        DEFAULT_PARAMS = dict()

        def __init__(self) -> None:  # pylint: disable=super-init-not-called
            # Do nothing here on purpose.
            pass

        def _fit(self, data: Dataset, **kwargs) -> "MyModel":
            # Do something...
            return self

        def check_model_requirements(self) -> None:
            # Do some checks...
            self.mock_check_model_requirements()

    return MyModel()


class TestBaseModel:
    def test_init(self):
        class MyModel(BaseModel):
            mock_check_model_requirements = Mock()
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyModel":
                # Do something...
                return self

            def check_model_requirements(self) -> None:
                # Do some checks...
                self.mock_check_model_requirements()

        my_model = MyModel(params={"param_1": 10})

        assert my_model.params == {"param_1": 10}
        my_model.mock_check_model_requirements.assert_called_once()
        assert my_model._fit_called is False  # pylint: disable=protected-access

    def test_fit_call(self, monkeypatch):
        # Arrange:
        mock_check_data_requirements = Mock()
        monkeypatch.setattr(
            "clairvoyance2.interface.model.RequirementsChecker.check_data_requirements_general",
            mock_check_data_requirements,
            raising=True,
        )

        class MyModel(BaseModel):
            mock_fit = Mock()
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyModel":
                # Do something...
                self.mock_fit(data)
                return self

            def check_model_requirements(self) -> None:
                # Do some checks...
                pass

        mock_data = Mock()

        # Act:
        my_model = MyModel(params={"param_1": 10})
        my_model.fit(mock_data)

        # Assert:
        my_model.mock_fit.assert_called_once_with(mock_data)
        mock_check_data_requirements.assert_called_once()
        assert my_model._fit_called is True  # pylint: disable=protected-access

    class TestProcessParamsMethod:
        @pytest.mark.parametrize(
            "default_params, params, expected_processed_params",
            [
                ({"param_1": 0, "param_2": False}, {"param_1": 0, "param_2": False}, {"param_1": 0, "param_2": False}),
                (
                    {"param_1": 0, "param_2": False},
                    DotMap({"param_1": 0, "param_2": False}, _dynamic=False),
                    {"param_1": 0, "param_2": False},
                ),
                ({"param_1": 0, "param_2": False}, {"param_1": 1, "param_2": True}, {"param_1": 1, "param_2": True}),
                ({"param_1": 0, "param_2": False}, {"param_1": 6}, {"param_1": 6, "param_2": False}),
                ({"param_1": 0, "param_2": False}, dict(), {"param_1": 0, "param_2": False}),
                ({"param_1": 0, "param_2": False}, None, {"param_1": 0, "param_2": False}),
                (dict(), dict(), dict()),
                (dict(), None, dict()),
            ],
        )
        def test_params_provided(
            self, mock_model_for_process_params, default_params, params, expected_processed_params
        ):
            mock_model_for_process_params.DEFAULT_PARAMS = default_params
            params_processed = mock_model_for_process_params._process_params(params)  # pylint: disable=protected-access

            assert isinstance(params_processed, Mapping)
            assert isinstance(params_processed, Dict)
            assert isinstance(params_processed, DotMap)

            assert params_processed == expected_processed_params

        @pytest.mark.parametrize(
            "default_params, params",
            [
                ({"param_1": 0, "param_2": False}, {"param_1": 0, "param_3": False}),
                ({"param_1": 0, "param_2": False}, {"param_3": False}),
                ({"param_1": 0}, {"param_3": False}),
                (dict(), {"param_1": 0, "param_3": False}),
            ],
        )
        def test_raises_unknown_param_error(self, mock_model_for_process_params, default_params, params):
            mock_model_for_process_params.DEFAULT_PARAMS = default_params

            with pytest.raises(ValueError) as excinfo:
                _ = mock_model_for_process_params._process_params(params)  # pylint: disable=protected-access
            assert "unknown parameter" in str(excinfo.value).lower()

        def test_unknown_param_error_suppressed(self, mock_model_for_process_params):
            default_params = dict()
            params = {"param_1": 0, "param_3": False}

            mock_model_for_process_params.DEFAULT_PARAMS = default_params
            mock_model_for_process_params.check_unknown_params = False

            params_processed = mock_model_for_process_params._process_params(params)  # pylint: disable=protected-access

            assert params_processed == dict()

        def test_unknown_param_error_suppressed_with_warning(self, mock_model_for_process_params):
            default_params = {"param_1": 0, "param_2": False}
            params = {"param_1": 0, "param_3": False}

            mock_model_for_process_params.DEFAULT_PARAMS = default_params
            mock_model_for_process_params.check_unknown_params = False

            with pytest.warns(UserWarning):
                params_processed = mock_model_for_process_params._process_params(  # pylint: disable=protected-access
                    params
                )

            assert params_processed == {"param_1": 0, "param_2": False}

    class TestRepr:
        @pytest.mark.parametrize(
            "default_params",
            [
                {"param_1": 0, "param_2": [1, 22]},
                NamedTuple("DP", param_1=int, param_2=list)(param_1=0, param_2=[1, 22]),  # type: ignore
            ],
        )
        def test_typical_case(self, default_params):
            class MyModel(BaseModel):
                mock_check_model_requirements = Mock()
                DEFAULT_PARAMS = default_params

                def _fit(self, data: Dataset, **kwargs) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    self.mock_check_model_requirements()

            my_model = MyModel()
            repr_ = str(my_model)

            assert "MyModel(" == repr_[0:8]
            assert repr_[-1] == ")"
            assert "\n" in repr_
            assert "params:" in repr_
            assert "inferred_params:" not in repr_
            assert '"param_1": 0' in repr_
            assert '"param_2":[1,22]' in repr_.replace("\n", "").replace(" ", "")

        def test_empty_params(self):
            class MyModel(BaseModel):
                mock_check_model_requirements = Mock()
                DEFAULT_PARAMS = dict()

                def _fit(self, data: Dataset, **kwargs) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    self.mock_check_model_requirements()

            my_model = MyModel()
            repr_ = str(my_model)

            assert "MyModel(" == repr_[0:8]
            assert repr_[-1] == ")"
            assert "\n" in repr_
            assert "params:" in repr_
            assert "inferred_params:" not in repr_
            assert "{}" in repr_

        def test_has_inferred_params(self):
            class MyModel(BaseModel):
                mock_check_model_requirements = Mock()
                DEFAULT_PARAMS = {"param_1": 0, "param_2": [1, 22]}

                def _fit(self, data: Dataset, **kwargs) -> "MyModel":
                    # Do something...
                    return self

                def check_model_requirements(self) -> None:
                    # Do some checks...
                    self.mock_check_model_requirements()

            my_model = MyModel()
            my_model.inferred_params.ip_1 = 123
            my_model.inferred_params.ip_2 = (2, 8)
            repr_ = str(my_model)

            assert "MyModel(" == repr_[0:8]
            assert repr_[-1] == ")"
            assert "\n" in repr_
            assert "params:" in repr_
            assert "inferred_params:" in repr_
            assert '"param_1": 0' in repr_
            assert '"param_2":[1,22]' in repr_.replace("\n", "").replace(" ", "")
            assert '"ip_1": 123' in repr_
            assert '"ip_2":[2,8]' in repr_.replace("\n", "").replace(" ", "")


class TestTransformerModel:
    def test_init(self):
        class MyTransformerModel(TransformerModel):
            check_model_requirements = Mock()
            requirements = Mock(prediction_requirements=None)
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                # Do something...
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                return Mock()

        my_transformer_model = MyTransformerModel(params={"param_1": 20})

        assert my_transformer_model.params == {"param_1": 20}
        my_transformer_model.check_model_requirements.assert_called_once()
        assert my_transformer_model._fit_called is False  # pylint: disable=protected-access

    def test_init_fails_incompatible_requirements(self):
        class MyTransformerModel(TransformerModel):
            requirements = Mock(prediction_requirements="something")
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                # Do something...
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                return Mock()

        with pytest.raises(ValueError) as excinfo:
            MyTransformerModel(params={"param_1": 20})
        assert "PredictionRequirements" in str(excinfo.value)

    def test_transform(self, monkeypatch):
        # Arrange:
        mock_check_data_requirements = Mock()
        monkeypatch.setattr(
            "clairvoyance2.interface.model.RequirementsChecker.check_data_requirements_general",
            mock_check_data_requirements,
            raising=True,
        )

        class MyTransformerModel(TransformerModel):
            mock_transform = Mock()
            requirements = Mock(prediction_requirements=None)
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                # Do something...
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                self.mock_transform(data)
                return Mock()

        my_transformer_model = MyTransformerModel(params={"param_1": 20})
        my_transformer_model._fit_called = True  # pylint: disable=protected-access

        mock_data = Mock()

        # Act:
        my_transformer_model.transform(data=mock_data)

        # Assert:
        mock_check_data_requirements.assert_called_once()
        my_transformer_model.mock_transform.assert_called_once_with(mock_data)

    def test_transform_fails_fit_not_called(
        self,
    ):
        class MyTransformerModel(TransformerModel):
            mock_transform = Mock()
            requirements = Mock(prediction_requirements=None)
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                # Do something...
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                self.mock_transform(data)
                return Mock()

        my_transformer_model = MyTransformerModel(params={"param_1": 20})

        mock_data = Mock()

        with pytest.raises(RuntimeError) as excinfo:
            my_transformer_model.transform(data=mock_data)
        assert "must call `fit`" in str(excinfo.value).lower()

    def test_fit_transform(self, monkeypatch):
        # Arrange:
        mock_check_data_requirements = Mock()
        monkeypatch.setattr(
            "clairvoyance2.interface.model.RequirementsChecker.check_data_requirements_general",
            mock_check_data_requirements,
            raising=True,
        )

        class MyTransformerModel(TransformerModel):
            mock_fit = Mock()
            mock_transform = Mock()
            requirements = Mock(prediction_requirements=None)
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                self.mock_fit(data)
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                self.mock_transform(data)
                return Mock()

        my_transformer_model = MyTransformerModel(params={"param_1": 20})
        mock_data = Mock()

        # Act:
        my_transformer_model.fit_transform(data=mock_data)

        # Assert:
        mock_check_data_requirements.assert_called()
        my_transformer_model.mock_fit.assert_called_once_with(mock_data)
        my_transformer_model.mock_transform.assert_called_once_with(mock_data)

    def test_inverse_transform(self, monkeypatch):
        # Arrange:
        mock_check_data_requirements = Mock()
        monkeypatch.setattr(
            "clairvoyance2.interface.model.RequirementsChecker.check_data_requirements_general",
            mock_check_data_requirements,
            raising=True,
        )

        class MyTransformerModel(TransformerModel):
            mock_inverse_transform = Mock()
            requirements = Mock(prediction_requirements=None)
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                # Do something...
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                # Do something...
                return Mock()

            def _inverse_transform(self, data: Dataset, **kwargs) -> Dataset:
                self.mock_inverse_transform(data)
                return Mock()

        my_transformer_model = MyTransformerModel(params={"param_1": 20})
        my_transformer_model._fit_called = True  # pylint: disable=protected-access

        mock_data = Mock()

        # Act:
        my_transformer_model.inverse_transform(data=mock_data)

        # Assert:
        mock_check_data_requirements.assert_called_once()
        my_transformer_model.mock_inverse_transform.assert_called_once_with(mock_data)

    def test_inverse_transform_fails_fit_not_called(
        self,
    ):
        class MyTransformerModel(TransformerModel):
            mock_inverse_transform = Mock()
            requirements = Mock(prediction_requirements=None)
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                # Do something...
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                # Do something...
                return Mock()

            def _inverse_transform(self, data: Dataset, **kwargs) -> Dataset:
                self.mock_inverse_transform(data)
                return Mock()

        my_transformer_model = MyTransformerModel(params={"param_1": 20})

        mock_data = Mock()

        with pytest.raises(RuntimeError) as excinfo:
            my_transformer_model.inverse_transform(data=mock_data)
        assert "must call `fit`" in str(excinfo.value).lower()

    def test_inverse_transform_fails_not_implemented(
        self,
    ):
        class MyTransformerModel(TransformerModel):
            mock_inverse_transform = Mock()
            requirements = Mock(prediction_requirements=None)
            DEFAULT_PARAMS = {"param_1": 0}

            def _fit(self, data: Dataset, **kwargs) -> "MyTransformerModel":
                # Do something...
                return self

            def _transform(self, data: Dataset, **kwargs) -> Dataset:
                # Do something...
                return Mock()

            # Did not implement _inverse_transform(self, data: Dataset).

        my_transformer_model = MyTransformerModel(params={"param_1": 20})

        mock_data = Mock()

        with pytest.raises(NotImplementedError) as excinfo:
            my_transformer_model.inverse_transform(data=mock_data)
        assert "not implemented" in str(excinfo.value).lower()
