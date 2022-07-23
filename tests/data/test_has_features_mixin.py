from typing import NamedTuple

import pandas as pd
import pytest

from clairvoyance2.data.has_features_mixin import FeatureType, HasFeaturesMixin

# pylint: disable=protected-access


def test_df_for_features():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 1], "c": [11.0, 12.0, 111.0]})

    mixin = HasFeaturesMixin()
    mixin._data = df

    assert id(df) == id(mixin._df_for_features())


class TestCategoricalDefProperty:
    def test_get_first_access(self):
        mixin = HasFeaturesMixin()
        assert mixin._categorical_def == dict()
        assert mixin._categorical_def_ == dict()

    def test_get_when_value_available(self):
        mixin = HasFeaturesMixin()
        categorical_def = {"a": (1, 2), "b": (8, 9)}
        mixin._categorical_def_ = categorical_def  # Set value directly.

        assert mixin._categorical_def == categorical_def
        assert mixin._categorical_def_ == categorical_def

    def test_setter(self, monkeypatch):
        mixin = HasFeaturesMixin()
        categorical_def = {"a": (1, 2), "b": (8, 9)}

        def mock_set_categorical_def(self, value):
            self._categorical_def_ = value

        monkeypatch.setattr(
            "clairvoyance2.data.has_features_mixin.HasFeaturesMixin.set_categorical_def",
            mock_set_categorical_def,
            raising=True,
        )

        mixin._categorical_def = categorical_def  # Use setter.

        assert mixin._categorical_def_ == categorical_def

    @pytest.mark.parametrize(
        "categorical_def, expected_categorical_def",
        [
            (
                {"a": (1, 2), "b": (1,)},
                {"a": (1, 2), "b": (1,)},
            ),
            (
                {"a": [1, 2], "b": [1]},
                {"a": (1, 2), "b": (1,)},
            ),
            (
                ["a", "b"],
                {"a": tuple(), "b": tuple()},
            ),
        ],
    )
    def test_set_categorical_def_method_success(self, categorical_def, expected_categorical_def, monkeypatch):
        mixin = HasFeaturesMixin()
        df = pd.DataFrame({"a": [1, 2, 2], "b": [1, 1, 1]})

        def mock_init_features(self):  # pylint: disable=unused-argument
            return None

        monkeypatch.setattr(
            "clairvoyance2.data.has_features_mixin.HasFeaturesMixin._init_features",
            mock_init_features,
            raising=True,
        )

        mixin._data = df
        mixin.set_categorical_def(categorical_def)

        assert mixin._categorical_def_ == expected_categorical_def

    @pytest.mark.parametrize(
        "df, categorical_def",
        [
            (
                pd.DataFrame({"a": [1, 2, 2], "b": [1, 1, 1]}),
                {"a": (1, 2), "c": (1,)},
            ),
            (
                pd.DataFrame({"a": [1, 2, 2], "b": [1, 1, 1]}),
                ["a", "c"],
            ),
        ],
    )
    def test_set_categorical_def_method_exception_feature_not_found_in_data(self, df, categorical_def, monkeypatch):
        mixin = HasFeaturesMixin()

        def mock_init_features(self):  # pylint: disable=unused-argument
            return None

        monkeypatch.setattr(
            "clairvoyance2.data.has_features_mixin.HasFeaturesMixin._init_features",
            mock_init_features,
            raising=True,
        )

        mixin._data = df

        with pytest.raises(ValueError) as excinfo:
            mixin.set_categorical_def(categorical_def)
        assert "not found" in str(excinfo.value).lower()


class TestFeaturesProperty:
    def test_init_features_method(self, monkeypatch):
        # Arrange.
        df = pd.DataFrame({"a": [1, 2, 3], "b": [11.0, 12.0, 111.0]})
        mixin = HasFeaturesMixin()
        mixin._categorical_def_ = {"a": [1, 2, 3]}
        mixin._data = df

        def mock_categorical_feature(*args, **kwargs):  # pylint: disable=unused-argument
            return "mock_categorical_feature"

        def mock_numeric_feature(*args, **kwargs):  # pylint: disable=unused-argument
            return "mock_numeric_feature"

        monkeypatch.setattr(
            "clairvoyance2.data.has_features_mixin.CategoricalFeatureCreator.create_feature",
            mock_categorical_feature,
            raising=True,
        )

        monkeypatch.setattr(
            "clairvoyance2.data.has_features_mixin.FeatureCreator.create_feature",
            mock_numeric_feature,
            raising=True,
        )

        # Act.
        features = mixin._init_features()

        # Assert.
        assert len(features) == 2
        assert "a" in features and "b" in features
        assert features["a"] == "mock_categorical_feature"
        assert features["b"] == "mock_numeric_feature"

    def test_features_getter_first_access(self, monkeypatch):
        mixin = HasFeaturesMixin()

        def mock_init_features(self):  # pylint: disable=unused-argument
            return ("mock", "features")

        monkeypatch.setattr(
            "clairvoyance2.data.has_features_mixin.HasFeaturesMixin._init_features",
            mock_init_features,
            raising=True,
        )

        features = mixin.features

        assert features == ("mock", "features")

    def test_features_getter_when_value_available(self):
        mixin = HasFeaturesMixin()
        mixin._features = ("some", "existing", "features")

        features = mixin.features

        assert features == ("some", "existing", "features")


def test_feature_types():
    mixin = HasFeaturesMixin()

    class MockFeature(NamedTuple):
        feature_type: str

    features = {"a": MockFeature("numeric_feature"), "b": MockFeature("categorical_feature")}

    mixin._features = features

    feature_types = mixin.feature_types

    assert feature_types == {"a": "numeric_feature", "b": "categorical_feature"}


class TestIntegration:
    def test_features_success(self):
        mixin = HasFeaturesMixin()
        mixin._data = pd.DataFrame({"a": [1, 2, 2], "b": [1, 9, 3], "c": [11.0, 12.0, 111.0], "d": [1, 1, 1]})
        mixin._categorical_def = {"a": [1, 2], "d": tuple()}

        f = mixin.features

        assert len(f) == 4
        assert f["a"].feature_type == f["d"].feature_type == FeatureType.CATEGORICAL
        assert f["b"].feature_type == f["c"].feature_type == FeatureType.NUMERIC

    def test_features_fails_wrong_def(self):
        mixin = HasFeaturesMixin()
        mixin._data = pd.DataFrame({"a": [1, 2, 2]})

        with pytest.raises(TypeError) as excinfo:
            mixin._categorical_def = {"a": [1]}
        assert "expected categories" in str(excinfo.value).lower()
