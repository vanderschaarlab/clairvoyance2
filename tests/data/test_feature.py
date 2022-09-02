import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data.feature import (
    CategoricalFeature,
    CategoricalFeatureCreator,
    Feature,
    FeatureCreator,
    FeatureType,
    _rule_auto_determine_categorical_feature,
)

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def pd_series_ints():
    return pd.Series([1, 2, 6, 5, 3, 4])


@pytest.fixture
def pd_series_ints_1_2_only():
    return pd.Series([1, 2, 1, 2, 1, 2])


class TestFeature_Init:
    def test_infer_dtype_true_and_dtype_passed_fails(self, pd_series_ints):
        with pytest.raises(ValueError) as excinfo:
            Feature(data=pd_series_ints, feature_type=FeatureType.NUMERIC, infer_dtype=True, dtype=str)
        assert "`dtype` when `infer_dtype`" in str(excinfo.value).lower()

    def test_infer_dtype_false_and_dtype_not_passed_fails(self, pd_series_ints):
        with pytest.raises(ValueError) as excinfo:
            Feature(data=pd_series_ints, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=None)
        assert "must provide `dtype`" in str(excinfo.value).lower()

    def test_init_categorical_feature_directly_fails(self):
        with pytest.raises(ValueError) as excinfo:
            Feature(data=pd_series_ints, feature_type=FeatureType.CATEGORICAL, infer_dtype=False, dtype=str)
        assert "CategoricalFeature" in str(excinfo.value)


class TestFeature_DataProperty:
    def test_get(self, pd_series_ints):
        f = Feature(data=pd_series_ints, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=int)
        assert id(f.data) == id(pd_series_ints)

    @pytest.mark.parametrize(
        "data, data_new, dtype",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), pd.Series([88, 99, 77, 33]), int),
            (pd.Series([1.0, 2.0, 6.0, 5.0, 3.0, 4.0]), pd.Series([-88.0, -99.0, -11.0, -22.0]), float),
            (pd.Series(["aa", "bb", "cc"]), pd.Series(["xx", "yy", "zz"]), str),
        ],
    )
    def test_set_success(self, data, data_new, dtype, monkeypatch):

        # Patch so that dtype str case can still be tested with FeatureType.NUMERIC.
        monkeypatch.setattr(
            "clairvoyance2.data.feature.FEATURE_DTYPE_MAP",
            {
                FeatureType.NUMERIC: (float, int, str),
            },
            raising=True,
        )

        f = Feature(data=data, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype)
        f.data = data_new

        assert id(f.data) == id(data_new)

    @pytest.mark.parametrize(
        "data, data_new, dtype",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), pd.Series([88.0, 99.0, 77.0, 33.0]), int),
            (pd.Series([1.0, 2.0, 6.0, 5.0, 3.0, 4.0]), pd.Series([-1, -2, 3, 4, 6]), float),
            (pd.Series(["aa", "bb", "cc"]), pd.Series([1, 2, 3, 4, 5]), str),
        ],
    )
    def test_set_fails_validation(self, data, data_new, dtype, monkeypatch):

        # Patch so that dtype str case can still be tested with FeatureType.NUMERIC.
        monkeypatch.setattr(
            "clairvoyance2.data.feature.FEATURE_DTYPE_MAP",
            {
                FeatureType.NUMERIC: (float, int, str),
            },
            raising=True,
        )

        with pytest.raises(TypeError) as excinfo:
            f = Feature(data=data, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype)
            f.data = data_new
        assert "does not match" in str(excinfo.value).lower()


class TestNumericFeature_Init_ExplicitDtype:
    @pytest.mark.parametrize(
        "pd_series, dtype",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), int),
            (pd.Series([1.0, 2.0, 6.0, 5.0, 3.0, 4.0]), float),
        ],
    )
    def test_success(self, pd_series, dtype):
        f = Feature(data=pd_series, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype)
        assert f.feature_type == FeatureType.NUMERIC
        assert f.dtype == dtype

    @pytest.mark.parametrize(
        "pd_series, dtype",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), str),
            (pd.Series(["a", "b", "c"]), tuple),
        ],
    )
    def test_wrong_dtype(self, pd_series, dtype):
        with pytest.raises(TypeError) as excinfo:
            Feature(data=pd_series, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype)
        assert "incompatible" in str(excinfo.value).lower()

    @pytest.mark.parametrize(
        "pd_series, dtype",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), float),
            (pd.Series([1.0, 2.0, 6.0, 5.0, 3.0, 4.0]), int),
            (pd.Series([1, 2, (3, 4), "b"]), int),
        ],
    )
    def test_dtype_data_mismatch(self, pd_series, dtype):
        with pytest.raises(TypeError) as excinfo:
            Feature(data=pd_series, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype)
        assert "does not match" in str(excinfo.value).lower()


class TestNumericFeature_Init_InferDtype:
    @pytest.mark.parametrize(
        "pd_series, expected_dtype",
        [
            (pd.Series([1, 2, 6, 5, 3, 4]), int),
            (pd.Series([1.0, 2.0, 6.0, 5.0, 3.0, 4.0]), float),
        ],
    )
    def test_success(self, pd_series, expected_dtype):
        f = Feature(data=pd_series, feature_type=FeatureType.NUMERIC, infer_dtype=True)
        assert f.feature_type == FeatureType.NUMERIC
        assert f.dtype == expected_dtype


class TestNumericFeature_NumericCompatibleProperty:
    @pytest.mark.parametrize(
        "data",
        [
            pd.Series([1, 2, 6, 5, 3, 4]),
            pd.Series([1.5, 2.5, 6.5, 5.5, 3.5, 4.5]),
        ],
    )
    def test_get_success(self, data):
        f = Feature(data=data, feature_type=FeatureType.NUMERIC, infer_dtype=True)
        assert f.numeric_compatible is True


class TestCategoricalFeature_Init:
    def test_infer_categories_true_and_categories_passed_fails(self, pd_series_ints):
        with pytest.raises(ValueError) as excinfo:
            CategoricalFeature(
                data=pd_series_ints, infer_dtype=False, dtype=int, infer_categories=True, categories=[1, 2, 3]
            )
        assert "`categories` when `infer_categories`" in str(excinfo.value).lower()

    def test_infer_categories_false_and_categories_not_passed_fails(self, pd_series_ints):
        with pytest.raises(ValueError) as excinfo:
            CategoricalFeature(
                data=pd_series_ints, infer_dtype=False, dtype=int, infer_categories=False, categories=None
            )
        assert "must provide `categories`" in str(excinfo.value).lower()


class TestCategoricalFeature_Init_ExplicitDtypeAndCategories:
    @pytest.mark.parametrize(
        "data, dtype, categories",
        [
            (pd.Series([1, 2, 1, 2, 2]), int, [1, 2]),
            (pd.Series([1, 2, 1, 2, 2]), int, [1, 2, 2, 2]),  # Test accidentally provide duplicates in categories.
            (pd.Series([1, 2, 1, 2, 2]), int, [1, 2, 1978]),  # Test case with "extra" categories.
            (pd.Series(["cat", "dog", "dog", "ferret"]), str, ["cat", "dog", "ferret"]),
        ],
    )
    def test_success(self, data, dtype, categories):
        f = CategoricalFeature(data=data, infer_dtype=False, dtype=dtype, infer_categories=False, categories=categories)
        assert f.feature_type == FeatureType.CATEGORICAL
        assert f.dtype == dtype
        assert [x in f.categories for x in categories]
        assert len(f.categories) <= len(categories)

    @pytest.mark.parametrize(
        "data, dtype, categories",
        [
            (pd.Series([1, 2, 1, 2, 2]), int, [1, 4]),
            (pd.Series(["cat", "dog", "dog", "ferret"]), str, ["cat", "hedgehog"]),
        ],
    )
    def test_incompatible_categories(self, data, dtype, categories):
        with pytest.raises(TypeError) as excinfo:
            CategoricalFeature(data=data, infer_dtype=False, dtype=dtype, infer_categories=False, categories=categories)
        assert "expected categories" in str(excinfo.value).lower()


class TestCategoricalFeature_Init_ExplicitDtype_InferCategories:
    @pytest.mark.parametrize(
        "data, dtype, expected_categories",
        [
            (pd.Series([1, 2, 1, 2, 2]), int, (1, 2)),
            (
                pd.Series(["cat", "dog", "dog", "ferret"]),
                str,
                ("cat", "dog", "ferret"),
            ),
        ],
    )
    def test_success(self, data, dtype, expected_categories):
        f = CategoricalFeature(data=data, infer_dtype=False, dtype=dtype, infer_categories=True, categories=None)
        assert f.feature_type == FeatureType.CATEGORICAL
        assert f.dtype == dtype
        assert f.categories == expected_categories


class TestCategoricalFeature_Init_InferDtype_InferCategories:
    @pytest.mark.parametrize(
        "data, expected_dtype, expected_categories",
        [
            (pd.Series([1, 2, 1, 2, 2]), int, (1, 2)),
            (pd.Series(["cat", "dog", "dog", "ferret"]), str, ("cat", "dog", "ferret")),
        ],
    )
    def test_success(self, data, expected_dtype, expected_categories):
        f = CategoricalFeature(data=data, infer_dtype=True, dtype=None, infer_categories=True, categories=None)
        assert f.feature_type == FeatureType.CATEGORICAL
        assert f.dtype == expected_dtype
        assert f.categories == expected_categories


class TestCategoricalFeature_NumericCompatibleProperty:
    @pytest.mark.parametrize("infer_dtype", [True, False])
    @pytest.mark.parametrize(
        "data, dtype, expect",
        [
            (pd.Series([1, 2, 2, 1]), int, True),
            (pd.Series([1.5, 2.5, 2.5, 1.5]), float, True),
            (pd.Series(["a", "b", "c", "c"]), str, False),
        ],
    )
    def test_get_success(self, data, dtype, infer_dtype, expect):
        dtype = dtype if infer_dtype is False else None
        f = CategoricalFeature(data=data, infer_dtype=infer_dtype, dtype=dtype, infer_categories=True)
        assert f.numeric_compatible == expect


class TestCategoricalFeature_CategoriesProperty:
    def test_get_success(self, pd_series_ints_1_2_only):
        f = CategoricalFeature(
            data=pd_series_ints_1_2_only, infer_dtype=False, dtype=int, infer_categories=False, categories=[1, 2]
        )
        assert f.categories == (1, 2)

    @pytest.mark.parametrize(
        "data, dtype, categories, categories_new",
        [
            (pd.Series([1, 2, 2, 1]), int, (1, 2), (1, 2, 3)),
            (pd.Series(["a", "b", "c", "c"]), str, ("a", "b", "c", "d"), ("a", "b", "c")),
        ],
    )
    def test_set_success(self, data, dtype, categories, categories_new):
        f = CategoricalFeature(data=data, infer_dtype=False, dtype=dtype, infer_categories=False, categories=categories)
        f.categories = categories_new
        assert f.categories == categories_new

    @pytest.mark.parametrize(
        "data, dtype, categories, categories_new",
        [
            (pd.Series([1, 2, 2, 1]), int, (1, 2), (1, "2", 3)),
            (pd.Series(["a", "b", "c", "c"]), str, ("a", "b", "c", "d"), (1, 2, "a")),
        ],
    )
    def test_set_fails_new_categories_types(self, data, dtype, categories, categories_new):
        with pytest.raises(TypeError) as excinfo:
            f = CategoricalFeature(
                data=data, infer_dtype=False, dtype=dtype, infer_categories=False, categories=categories
            )
            f.categories = categories_new
        assert "elements of categories" in str(excinfo.value).lower()

    @pytest.mark.parametrize(
        "data, dtype, categories, categories_new",
        [
            (pd.Series([1, 2, 2, 1]), int, (1, 2), (1, 3)),
            (pd.Series(["a", "b", "c", "c"]), str, ("a", "b", "c"), ("a",)),
        ],
    )
    def test_set_fails_validation(self, data, dtype, categories, categories_new):
        with pytest.raises(TypeError) as excinfo:
            f = CategoricalFeature(
                data=data, infer_dtype=False, dtype=dtype, infer_categories=False, categories=categories
            )
            f.categories = categories_new
        assert "expected categories" in str(excinfo.value).lower()

    @pytest.mark.parametrize(
        "data, dtype, categories, data_new, categories_new",
        [
            (pd.Series([1, 2, 2, 1]), int, (1, 2), pd.Series([1, 3, 3, 1]), (1, 3)),
            (
                pd.Series(["a", "b", "c", "b"]),
                str,
                ("a", "b", "c"),
                pd.Series(["a", "b", "c", "d"]),
                ("a", "b", "c", "d"),
            ),
        ],
    )
    def test_update_data_and_categories_success(self, data, dtype, categories, data_new, categories_new):
        f = CategoricalFeature(data=data, infer_dtype=False, dtype=dtype, infer_categories=False, categories=categories)
        f.update_data_and_categories(data=data_new, categories=categories_new)
        assert f.categories == categories_new

    @pytest.mark.parametrize(
        "data, dtype, categories, data_new, categories_new",
        [
            (pd.Series([1, 2, 2, 1]), int, (1, 2), pd.Series([1, 3, 3, 1]), (1, 2)),
            (
                pd.Series(["a", "b", "c", "b"]),
                str,
                ("a", "b", "c"),
                pd.Series(["a", "b", "c", "d"]),
                ("a", "q"),
            ),
        ],
    )
    def test_update_data_and_categories_fails(self, data, dtype, categories, data_new, categories_new):
        with pytest.raises(TypeError) as excinfo:
            f = CategoricalFeature(
                data=data, infer_dtype=False, dtype=dtype, infer_categories=False, categories=categories
            )
            f.update_data_and_categories(data=data_new, categories=categories_new)
        assert "expected categories" in str(excinfo.value).lower()


class TestFeature_Eq:
    @pytest.mark.parametrize(
        "data_1, dtype_1, data_2, dtype_2",
        [
            (pd.Series([1, 2, 3]), int, pd.Series([1, 2, 3]), int),
            (pd.Series([1.0, 2.0, 3.0]), float, pd.Series([1.0, 2.0, 3.0]), float),
        ],
    )
    def test_eq_numeric_feature(self, data_1, dtype_1, data_2, dtype_2):
        feature_1 = Feature(data=data_1, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype_1)
        feature_2 = Feature(data=data_2, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype_2)
        assert feature_1 == feature_2

    @pytest.mark.parametrize(
        "data_1, dtype_1, data_2, dtype_2",
        [
            (pd.Series([1, 2, 3]), int, pd.Series([-1, -2, -3]), int),
            (pd.Series([1.0, 2.0, 3.0]), float, pd.Series([-1.0, -2.0, -3.0]), float),
        ],
    )
    def test_neq_numeric_feature(self, data_1, dtype_1, data_2, dtype_2):
        feature_1 = Feature(data=data_1, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype_1)
        feature_2 = Feature(data=data_2, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=dtype_2)
        assert feature_1 != feature_2

    def test_neq_numeric_feature_different_class(self, pd_series_ints):
        feature = Feature(data=pd_series_ints, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=int)
        assert feature != (1, 2, 3)

    def test_neq_numeric_feature_categorical_feature(self, pd_series_ints, pd_series_ints_1_2_only):
        feature_numeric = Feature(data=pd_series_ints, feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=int)
        feature_categorical = CategoricalFeature(
            data=pd_series_ints_1_2_only, infer_dtype=False, dtype=int, infer_categories=False, categories=[1, 2]
        )
        assert feature_numeric != feature_categorical

    @pytest.mark.parametrize(
        "data_1, dtype_1, categories_1, data_2, dtype_2, categories_2",
        [
            (pd.Series([1, 2, 2]), int, [1, 2], pd.Series([1, 2, 2]), int, [1, 2]),
            (pd.Series(["1", "2", "2"]), str, ["1", "2"], pd.Series(["1", "2", "2"]), str, ["1", "2"]),
        ],
    )
    def test_eq_categorical_feature(self, data_1, dtype_1, categories_1, data_2, dtype_2, categories_2):
        feature_1 = CategoricalFeature(
            data=data_1, infer_dtype=False, dtype=dtype_1, infer_categories=False, categories=categories_1
        )
        feature_2 = CategoricalFeature(
            data=data_2, infer_dtype=False, dtype=dtype_2, infer_categories=False, categories=categories_2
        )
        assert feature_1 == feature_2

    @pytest.mark.parametrize(
        "data_1, dtype_1, categories_1, data_2, dtype_2, categories_2",
        [
            (pd.Series([1, 2, 2]), int, [1, 2], pd.Series([1, 3, 3]), int, [1, 3]),
            (pd.Series([1, 2, 2]), int, [1, 2], pd.Series([1, 2, 2]), int, [1, 2, 3]),
            (pd.Series(["1", "2", "2"]), str, ["1", "2"], pd.Series(["a", "b", "b"]), str, ["a", "b"]),
        ],
    )
    def test_neq_categorical_feature(self, data_1, dtype_1, categories_1, data_2, dtype_2, categories_2):
        feature_1 = CategoricalFeature(
            data=data_1, infer_dtype=False, dtype=dtype_1, infer_categories=False, categories=categories_1
        )
        feature_2 = CategoricalFeature(
            data=data_2, infer_dtype=False, dtype=dtype_2, infer_categories=False, categories=categories_2
        )
        assert feature_1 != feature_2


class TestRuleAutoDetermineCategoricalFeature:
    @pytest.mark.parametrize(
        "max_categories_frac, max_categories_count, data, expected_result",
        [
            (0.2, 100, pd.Series([1, 2, 3, 4, 5]), False),  # False due to max_categories_frac
            (0.2, 100, pd.Series([1, 1, 1, 1, 1]), True),  # True as passes to max_categories_frac
            (0.0, 100, pd.Series(np.random.randint(0, 1, size=(99,))), False),  # False due to max_categories_frac = 0.
            (1.0, 100, pd.Series(np.random.randint(0, 1, size=(99,))), True),
            # ^ True due to max_categories_frac = 1. & <max_categories_count elements.
            (1.0, 100, pd.Series(np.asarray(range(0, 101))), False),
            # ^ False due to max_categories_frac = 1. & >max_categories_count elements.
            (0.0, 100, pd.Series(["a", "b", "c"]), True),  # Always True, because str.
            (0.0, 100, pd.Series([tuple("a"), tuple("b"), tuple("c")]), False),
            # ^ False because type tuple not registered as applicable to FeatureType.CATEGORICAL
        ],
    )
    def test_rule(self, max_categories_frac, max_categories_count, data, expected_result):
        result = _rule_auto_determine_categorical_feature(
            max_categories_frac=max_categories_frac, max_categories_count=max_categories_count, data=data
        )
        assert result == expected_result

    @pytest.mark.parametrize(
        "max_categories_frac",
        [1.1, -0.1],
    )
    def test_input_validation_fails(self, max_categories_frac, pd_series_ints):
        with pytest.raises(ValueError) as excinfo:
            _rule_auto_determine_categorical_feature(
                max_categories_frac=max_categories_frac, max_categories_count=100, data=pd_series_ints
            )
        assert "must be between 0" in str(excinfo.value).lower()


class TestFeatureCreator:
    def test_create_feature_method(self):
        f = FeatureCreator().create_feature(
            data=pd.Series([1, 2, 3, 4, 5]), feature_type=FeatureType.NUMERIC, infer_dtype=False, dtype=int
        )
        assert isinstance(f, Feature)

    @pytest.mark.parametrize(
        "data, max_categories_frac, max_categories_count, expect_categorical_feature",
        [
            (pd.Series([1, 2, 3, 4, 5, 6, 7, 8]), 0.2, 100, False),
            (pd.Series([1, 2, 2, 2, 2, 2, 1, 1]), 0.2, 100, True),
            (pd.Series(["1", "2", "2", "2", "2"]), 0.2, 100, True),
        ],
    )
    def test_auto_create_feature_from_data_method(
        self, data, max_categories_frac, max_categories_count, expect_categorical_feature
    ):
        f = FeatureCreator().auto_create_feature_from_data(
            data=data, max_categories_frac=max_categories_frac, max_categories_count=max_categories_count
        )
        assert isinstance(f, Feature)
        if expect_categorical_feature:
            assert isinstance(f, CategoricalFeature)
        else:
            assert not isinstance(f, CategoricalFeature)


class TestCategoricalFeatureCreator:
    def test_create_feature_method(self):
        cf = CategoricalFeatureCreator().create_feature(
            data=pd.Series([20, 20, 10, 10]),
            infer_dtype=False,
            dtype=int,
            infer_categories=False,
            categories=[10, 20],
        )
        assert isinstance(cf, CategoricalFeature)
