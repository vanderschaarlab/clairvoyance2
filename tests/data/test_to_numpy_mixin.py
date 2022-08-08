from unittest.mock import Mock

import pytest

from clairvoyance2.data.constants import DEFAULT_PADDING_INDICATOR
from clairvoyance2.data.to_numpy_mixin import ToNumpyMixin

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
        "clairvoyance2.data.to_numpy_mixin.ToNumpyMixin._to_numpy_static",
        mock._to_numpy_static,
        raising=True,
    )
    monkeypatch.setattr(
        "clairvoyance2.data.to_numpy_mixin.ToNumpyMixin._to_numpy_time_series",
        mock._to_numpy_time_series,
        raising=True,
    )
    return mock


class TestToNumpyArgsKwargsValidation:
    class TestPassingCase:
        def test_static_no_args(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            mixin.to_numpy()
            mock_to_numpy_methods._to_numpy_static.assert_called_once_with()

        def test_time_series_no_args(self, monkeypatch):
            mock = Mock()
            # ._to_numpy_static() will raise NotImplementedError and will fall back on ._to_numpy_time_series().
            monkeypatch.setattr(
                "clairvoyance2.data.to_numpy_mixin.ToNumpyMixin._to_numpy_time_series",
                mock._to_numpy_time_series,
                raising=True,
            )
            mixin = ToNumpyMixin()
            mixin.to_numpy()
            mock._to_numpy_time_series.assert_called_once_with(
                padding_indicator=DEFAULT_PADDING_INDICATOR, max_len=None
            )

        def test_time_series_padding_indicator_only_as_kwarg(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            mixin.to_numpy(padding_indicator=11.0)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(padding_indicator=11.0, max_len=None)

        def test_time_series_padding_indicator_only_as_arg(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            mixin.to_numpy(11.0)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(padding_indicator=11.0, max_len=None)

        def test_time_series_both_as_kwargs(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            mixin.to_numpy(padding_indicator=11.0, max_len=12)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(padding_indicator=11.0, max_len=12)

        def test_time_series_both_as_arg_kwarg(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            mixin.to_numpy(11.0, max_len=12)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(padding_indicator=11.0, max_len=12)

        def test_time_series_both_as_args(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            mixin.to_numpy(11.0, 12)
            mock_to_numpy_methods._to_numpy_time_series.assert_called_once_with(padding_indicator=11.0, max_len=12)

    class TestFailingCase:
        def test_padding_indicator_both_args_kwargs(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(11.0, padding_indicator=11.0)
            assert "both arguments and keyword arguments" in str(excinfo.value) and "padding_indicator" in str(
                excinfo.value
            )
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(11.0, padding_indicator=11.0, max_len=12)
            assert "both arguments and keyword arguments" in str(excinfo.value) and "padding_indicator" in str(
                excinfo.value
            )
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(11.0, 12, padding_indicator=11.0, max_len=12)
            assert "both arguments and keyword arguments" in str(excinfo.value) and "padding_indicator" in str(
                excinfo.value
            )

        def test_max_len_both_args_kwargs(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(11.0, 12, max_len=12)
            assert "both arguments and keyword arguments" in str(excinfo.value) and "max_len" in str(excinfo.value)

        def test_too_many_args(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(11.0, 12, 13.0)
            assert "no more than" in str(excinfo.value).lower()

        def test_unknown_kwarg(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(some_kwarg=13.0)
            assert "unknown" in str(excinfo.value).lower()

        def test_too_many_kwargs(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(11.0, 12, some_kwarg=13.0)
            assert "too many" in str(excinfo.value).lower()
            with pytest.raises(ValueError) as excinfo:
                mixin.to_numpy(padding_indicator=11.0, max_len=12, some_kwarg=13.0)
            assert "too many" in str(excinfo.value).lower()

        def test_wrong_type_padding_indicator(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            with pytest.raises(TypeError) as excinfo:
                mixin.to_numpy(padding_indicator="11.")
            assert "padding_indicator" in str(excinfo.value).lower()

        def test_wrong_type_max_len(self, mock_to_numpy_methods):
            mixin = ToNumpyMixin()
            with pytest.raises(TypeError) as excinfo:
                mixin.to_numpy(padding_indicator=11.0, max_len="12")
            assert "max_len" in str(excinfo.value).lower()
