import warnings

import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data.feature import Feature

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def pd_series_ints():
    return pd.Series([1, 2, 6, 5, 3, 4])


@pytest.fixture
def pd_series_ints_1_2_only():
    return pd.Series([1, 2, 1, 2, 1, 2])


class TestFeature_SeriesAttribute:
    def test_get(self, pd_series_ints):
        f = Feature(name="dummy", series=pd_series_ints)
        assert id(f.series) == id(pd_series_ints)

    @pytest.mark.parametrize(
        "data, data_new",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), pd.Series([88, 99, 77, 33])),
            (pd.Series([1.0, 2.0, 6.0, 5.0, 3.0, 4.0]), pd.Series([-88.0, -99.0, -11.0, -22.0])),
            (pd.Series(["aa", "bb", "cc"]), pd.Series(["xx", "yy", "zz"])),
        ],
    )
    def test_set_success(self, data, data_new):
        f = Feature(name="dummy", series=data)
        f.series = data_new

        assert id(f.series) == id(data_new)


class TestFeature_NumericCompatibleProperty:
    @pytest.mark.parametrize(
        "data, expect",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), True),
            (pd.Series([1.5, 2.5, 6.5, 5.5, 3.5, 4.5]), True),
            (pd.Series([1, 2, 2, 1]), True),
            (pd.Series([1.5, 2.5, 2.5, 1.5]), True),
            (pd.Series(["a", "b", "c", "c"]), False),
        ],
    )
    def test_get_success(self, data, expect):
        f = Feature(name="dummy", series=data)
        assert f.numeric_compatible == expect


class TestFeature_CategoricalCompatibleProperty:
    @pytest.mark.parametrize(
        "data, expect",
        [
            (pd.Series([1, 2, 2, 1]), True),
            (pd.Series([1.5, 2.5, 2.5, 1.5]), True),
            (pd.Series(["a", "b", "c", "c"]), True),
        ],
    )
    def test_get_success(self, data, expect):
        f = Feature(name="dummy", series=data)
        assert f.categorical_compatible == expect


class TestFeature_CategoriesProperty:
    def test_get_success(self, pd_series_ints_1_2_only):
        f = Feature(
            name="dummy",
            series=pd_series_ints_1_2_only,
        )
        assert f.categories == (1, 2)

    @pytest.mark.parametrize(
        "data, categories_expected",
        [
            (pd.Series([1, 1]), (1,)),
            (pd.Series([1, 1, 2, 1]), (1, 2)),
            (pd.Series(["a", "b", "c", "c"]), ("a", "b", "c")),
        ],
    )
    def test_more_get_success(self, data, categories_expected):
        f = Feature(name="dummy", series=data)
        assert f.categories == categories_expected

    def test_warns_too_many(self):
        num = 100
        f = Feature(name="dummy", series=pd.Series(np.random.rand(num)))
        with pytest.warns(UserWarning) as record:
            _ = f.categories
        assert len(record) == 1
        assert "number of categories" in str(record[0].message)

    def test_does_not_warn_too_many(self):
        num = 99
        f = Feature(name="dummy", series=pd.Series(np.random.rand(num)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = f.categories


class TestFeature_Eq:
    @pytest.mark.parametrize(
        "data_1, name_1, data_2, name_2",
        [
            (pd.Series([1, 2, 3]), "a", pd.Series([1, 2, 3]), "a"),
            (pd.Series([1.0, 2.0, 3.0]), "a", pd.Series([1.0, 2.0, 3.0]), "a"),
        ],
    )
    def test_eq_numeric_feature(self, data_1, name_1, data_2, name_2):
        feature_1 = Feature(name=name_1, series=data_1)
        feature_2 = Feature(name=name_2, series=data_2)
        assert feature_1 == feature_2

    @pytest.mark.parametrize(
        "data_1, name_1, data_2, name_2",
        [
            (pd.Series([1, 2, 3]), "a", pd.Series([1, 2, 3]), "b"),
            (pd.Series([1.0, 2.0, 3.0]), "a", pd.Series([1.0, -2.0, 3.0]), "a"),
        ],
    )
    def test_neq_numeric_feature(self, data_1, name_1, data_2, name_2):
        feature_1 = Feature(name=name_1, series=data_1)
        feature_2 = Feature(name=name_2, series=data_2)
        assert feature_1 != feature_2

    def test_neq_numeric_feature_different_class(self, pd_series_ints):
        feature = Feature(name="dummy", series=pd_series_ints)
        assert feature != (1, 2, 3)

    @pytest.mark.parametrize(
        "data_1, name_1, data_2, name_2",
        [
            (pd.Series(["1", "2", "1"]), "a", pd.Series(["1", "2", "1"]), "a"),
        ],
    )
    def test_eq_categorical_feature(self, data_1, name_1, data_2, name_2):
        feature_1 = Feature(name=name_1, series=data_1)
        feature_2 = Feature(name=name_2, series=data_2)
        assert feature_1 == feature_2

    @pytest.mark.parametrize(
        "data_1, name_1, data_2, name_2",
        [
            (pd.Series(["1", "2", "1"]), "a", pd.Series(["1", "2", "1"]), "b"),
            (pd.Series(["1", "2", "1"]), "a", pd.Series(["1", "2", "2"]), "a"),
        ],
    )
    def test_neq_categorical_feature(self, data_1, name_1, data_2, name_2):
        feature_1 = Feature(name=name_1, series=data_1)
        feature_2 = Feature(name=name_2, series=data_2)
        assert feature_1 != feature_2
