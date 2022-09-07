from unittest.mock import Mock

import pytest

from clairvoyance2.data.constants import DEFAULT_PADDING_INDICATOR
from clairvoyance2.data.to_tensor_like_mixin import ToTensorLikeMixin

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.

# pylint: disable=unused-argument
# ^ Some fixtures intentionally included without being explicitly used in test functions.

# pylint: disable=protected-access
# ^ Expected to access to some protected attributes here.


@pytest.fixture
def mock_to_numpy_methods(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(
        "clairvoyance2.data.to_tensor_like_mixin.ToTensorLikeMixin._to_numpy_static",
        mock._to_numpy_static,
        raising=True,
    )
    monkeypatch.setattr(
        "clairvoyance2.data.to_tensor_like_mixin.ToTensorLikeMixin._to_numpy_time_series",
        mock._to_numpy_time_series,
        raising=True,
    )
    return mock


class TestToNumpy:
    class TestSuccess:
        def test_static_no_kwargs(self, mock_to_numpy_methods):
            mixin = ToTensorLikeMixin()
            mixin.to_numpy()
            mock_to_numpy_methods._to_numpy_static.assert_called_once_with()

        def test_time_series_no_kwargs(self, monkeypatch):
            mock = Mock()
            # ._to_numpy_static() will raise NotImplementedError and will fall back on ._to_numpy_time_series().
            monkeypatch.setattr(
                "clairvoyance2.data.to_tensor_like_mixin.ToTensorLikeMixin._to_numpy_time_series",
                mock._to_numpy_time_series,
                raising=True,
            )
            mixin = ToTensorLikeMixin()
            mixin.to_numpy()
            mock._to_numpy_time_series.assert_called_once_with(
                padding_indicator=DEFAULT_PADDING_INDICATOR, max_len=None
            )

        def test_time_series_padding_indicator_only(self, mock_to_numpy_methods):
            mixin = ToTensorLikeMixin()
            mixin.to_numpy(padding_indicator=11.0)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(padding_indicator=11.0, max_len=None)

        def test_time_series_max_len_only(self, mock_to_numpy_methods):
            mixin = ToTensorLikeMixin()
            mixin.to_numpy(max_len=5)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(
                padding_indicator=DEFAULT_PADDING_INDICATOR, max_len=5
            )

        def test_time_series_both_kwargs(self, mock_to_numpy_methods):
            mixin = ToTensorLikeMixin()
            mixin.to_numpy(padding_indicator=11.0, max_len=12)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(padding_indicator=11.0, max_len=12)

    class TestValidationFails:
        def test_args_fails(self):
            mixin = ToTensorLikeMixin()
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0)  # pylint: disable=too-many-function-args
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0, 5)  # pylint: disable=too-many-function-args
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0, 12, 13.0)  # pylint: disable=too-many-function-args
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0, max_len=5)  # pylint: disable=E
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0, padding_indicator=11.0, max_len=5)  # pylint: disable=E
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0, padding_indicator=11.0, max_len=5, some_kwarg=13.0)  # pylint: disable=E
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0, 12, some_kwarg=13.0)  # pylint: disable=E

        def test_kwargs_fails(self):
            mixin = ToTensorLikeMixin()
            with pytest.raises(TypeError):
                mixin.to_numpy(some_kwarg=13.0)  # pylint: disable=E
            with pytest.raises(TypeError):
                mixin.to_numpy(11.0, 12, some_kwarg=13.0)  # pylint: disable=E
            with pytest.raises(TypeError):
                mixin.to_numpy(padding_indicator=11.0, max_len=12, some_kwarg=13.0)  # pylint: disable=E

        def test_wrong_type_padding_indicator(self, mock_to_numpy_methods):
            mixin = ToTensorLikeMixin()
            with pytest.raises(TypeError) as excinfo:
                mixin.to_numpy(padding_indicator="11.")
            assert "padding_indicator" in str(excinfo.value).lower()

        def test_wrong_type_max_len(self, mock_to_numpy_methods):
            mixin = ToTensorLikeMixin()
            with pytest.raises(TypeError) as excinfo:
                mixin.to_numpy(padding_indicator=11.0, max_len="12")
            assert "max_len" in str(excinfo.value).lower()
