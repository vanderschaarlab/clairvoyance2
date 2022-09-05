import pandas as pd

from clairvoyance2.data.has_features_mixin import Feature, HasFeaturesMixin

# pylint: disable=protected-access


def test_df_for_features():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 1], "c": [11.0, 12.0, 111.0]})

    mixin = HasFeaturesMixin()
    mixin._data = df

    assert id(df) == id(mixin._df_for_features)


class TestFeaturesProperty:
    def test_init_features_method(self):
        # Arrange.
        df = pd.DataFrame({"a": [1, 2, 3], "b": [11.0, 12.0, 111.0]})
        mixin = HasFeaturesMixin()
        mixin._data = df

        # Act.
        features = mixin.features

        # Assert.
        assert len(features) == 2
        assert "a" in features and "b" in features
        assert isinstance(features["a"], Feature)
        assert isinstance(features["b"], Feature)
        assert features["a"].name == "a"
        assert features["b"].name == "b"
        assert list(features["a"].series.values) == [1, 2, 3]
        assert list(features["b"].series.values) == [11.0, 12.0, 111.0]


class TestIntegration:
    def test_all_categorical_features(self):
        mixin = HasFeaturesMixin()
        mixin._data = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "a", "b"]})

        assert mixin.all_features_categorical is True

    def test_all_numeric_features_true(self):
        mixin = HasFeaturesMixin()
        mixin._data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        assert mixin.all_features_numeric is True

    def test_all_numeric_features_false(self):
        mixin = HasFeaturesMixin()
        mixin._data = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        assert mixin.all_features_numeric is False

    def test_n_features(self):
        mixin = HasFeaturesMixin()
        mixin._data = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        assert mixin.n_features == 2
