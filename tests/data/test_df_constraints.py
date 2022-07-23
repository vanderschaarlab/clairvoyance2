from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

import clairvoyance2.data.df_constraints as dfc


def test_empty_constraints():
    df = pd.DataFrame({"col_0": [1.0, 2.0, 3.0], ("col_1", "abc"): ["a", 111.0, "c"]}, index=pd.RangeIndex(0, 2 + 1, 1))
    constraints_checker = dfc.ConstraintsChecker(dfc.Constraints())
    constraints_checker.check(df)


class TestIndexConstraints:
    @pytest.mark.parametrize(
        "df, types, expectation",
        [
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.RangeIndex(0, 2 + 1, 1)),
                [pd.RangeIndex, pd.TimedeltaIndex],
                does_not_raise(),
            ),
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=["a", "b", "c"]),
                [pd.RangeIndex, pd.TimedeltaIndex],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_types(self, df, types, expectation):
        with expectation:
            constraints = dfc.Constraints(on_index=dfc.IndexConstraints(types=types))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(types) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, dtypes, expectation",
        [
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0]}, index=pd.Index([0, 1, 2], dtype=int)),
                [int, float],
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    {"col_0": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"])
                ),
                [np.datetime64],
                does_not_raise(),
            ),
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0]}, index=pd.Index(["a", "b", "c"], dtype=object)),
                [int, float],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_dtypes(self, df, dtypes, expectation):
        with expectation:
            constraints = dfc.Constraints(on_index=dfc.IndexConstraints(dtypes=dtypes))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(dtypes) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, dtype_object_constrain_types, expectation",
        [
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.Index(["i1", "i2", "i3"])),
                [str, tuple],
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    {"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]},
                    index=pd.Index([1, "a", (12.0,)], dtype=object),
                ),
                [str, tuple],
                pytest.raises(TypeError),
            ),
            (
                # Check also the case where it's not an object index, so the constraint is not relevant.
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.Index([1, 2, 3])),
                [str, tuple],
                does_not_raise(),
            ),
        ],
    )
    def test_dtype_object_constrain_types(self, df, dtype_object_constrain_types, expectation):
        with expectation:
            constraints = dfc.Constraints(
                on_index=dfc.IndexConstraints(dtype_object_constrain_types=dtype_object_constrain_types)
            )
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(dtype_object_constrain_types) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, expectation",
        [
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.Index([1.0, 123.0, 999.0])),
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    {"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.Index([1.0, -123.0, 999.0])
                ),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_enforce_monotonic_increasing(self, df, expectation):
        with expectation:
            constraints = dfc.Constraints(on_index=dfc.IndexConstraints(enforce_monotonic_increasing=True))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            constraints_checker.check(df)

    @pytest.mark.parametrize(
        "df, expectation",
        [
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.Index([0, 1, 2])),
                does_not_raise(),
            ),
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.Index([1, 1, 2])),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_enforce_unique(self, df, expectation):
        with expectation:
            constraints = dfc.Constraints(on_index=dfc.IndexConstraints(enforce_unique=True))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            constraints_checker.check(df)

    @pytest.mark.parametrize(
        "df, expectation",
        [
            (
                pd.DataFrame({"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]}, index=pd.Index([0, 1, 2])),
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    {"col_0": [1.0, 2.0, 3.0], "col_1": ["a", "b", "c"]},
                    index=pd.MultiIndex.from_tuples([(0, 1), (0, 2), (0, 3)]),
                ),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_enforce_not_multi_index(self, df, expectation):
        with expectation:
            constraints = dfc.Constraints(on_index=dfc.IndexConstraints(enforce_not_multi_index=True))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            constraints_checker.check(df)


class TestColumnConstraints:
    @pytest.mark.parametrize(
        "df, types, expectation",
        [
            (
                pd.DataFrame(
                    data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=pd.TimedeltaIndex(["4d", "1d"])
                ),
                [pd.RangeIndex, pd.TimedeltaIndex],
                does_not_raise(),
            ),
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=[1, 2]),
                [pd.RangeIndex, pd.TimedeltaIndex],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_types(self, df, types, expectation):
        with expectation:
            constraints = dfc.Constraints(on_columns=dfc.IndexConstraints(types=types))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(types) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, dtypes, expectation",
        [
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=["str", ("tu", "ple")]),
                [str, object],
                does_not_raise(),
            ),
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=[11.0, 12.0]),
                [str, object],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_dtypes(self, df, dtypes, expectation):
        with expectation:
            constraints = dfc.Constraints(on_columns=dfc.IndexConstraints(dtypes=dtypes))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(dtypes) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, dtype_object_constrain_types, expectation",
        [
            (
                pd.DataFrame(
                    data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=pd.Index(["goo", 11.0], dtype=object)
                ),
                [str, float],
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T,
                    columns=pd.Index([("tu", "ple"), 11.0], dtype=object),
                ),
                [str, float],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_dtype_object_constrain_types(self, df, dtype_object_constrain_types, expectation):
        with expectation:
            constraints = dfc.Constraints(
                on_columns=dfc.IndexConstraints(dtype_object_constrain_types=dtype_object_constrain_types)
            )
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(dtype_object_constrain_types) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, expectation",
        [
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=[12, 14]),
                does_not_raise(),
            ),
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=[12, 3]),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_enforce_monotonic_increasing(self, df, expectation):
        with expectation:
            constraints = dfc.Constraints(on_columns=dfc.IndexConstraints(enforce_monotonic_increasing=True))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            constraints_checker.check(df)

    @pytest.mark.parametrize(
        "df, expectation",
        [
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=["d", "e"]),
                does_not_raise(),
            ),
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=["d", "d"]),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_enforce_unique(self, df, expectation):
        with expectation:
            constraints = dfc.Constraints(on_columns=dfc.IndexConstraints(enforce_unique=True))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            constraints_checker.check(df)

    @pytest.mark.parametrize(
        "df, expectation",
        [
            (
                pd.DataFrame(data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T, columns=[0, 1]),
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    data=np.asarray([[1.0, 2.0, 3.0], ["a", "b", "c"]]).T,
                    columns=pd.MultiIndex.from_tuples([(0, 1), (0, 2)]),
                ),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_enforce_not_multi_index(self, df, expectation):
        with expectation:
            constraints = dfc.Constraints(on_columns=dfc.IndexConstraints(enforce_not_multi_index=True))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            constraints_checker.check(df)


class TestElementConstraints:
    @pytest.mark.parametrize(
        "df, dtypes, expectation",
        [
            (
                pd.DataFrame({"col_0": [1, 2, 3], "col_1": [11.0, 22.0, 33.0]}),
                [int, float],
                does_not_raise(),
            ),
            (
                pd.DataFrame({"col_0": ["a", "b", "c"], "col_1": [11.0, 22.0, 33.0]}),
                [int, float],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_dtypes(self, df, dtypes, expectation):
        with expectation:
            constraints = dfc.Constraints(on_elements=dfc.ElementConstraints(dtypes=dtypes))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(dtypes) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, dtype_object_constrain_types, expectation",
        [
            (
                pd.DataFrame(
                    {
                        "col_0": [["x", "y"], 2, ["aaa", "bbb", "ccc"]],  # dtype: object.
                        "col_1": [11.0, 22.0, 33.0],  # dtype: float, does not affect this test.
                        "col_2": [list(), -12, [11, 12]],  # dtype: object.
                    }
                ),
                [list, int],
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    {
                        "col_0": [["x", "y"], 2, ["aaa", "bbb", "ccc"]],  # dtype: object.
                        "col_1": [11.0, 22.0, 33.0],  # dtype: float, does not affect this test.
                        "col_2": [list(), -12, [11, 12]],  # dtype: object.
                    }
                ),
                [list, float],
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_dtype_object_constrain_types(self, df, dtype_object_constrain_types, expectation):
        with expectation:
            constraints = dfc.Constraints(
                on_elements=dfc.ElementConstraints(dtype_object_constrain_types=dtype_object_constrain_types)
            )
            constraints_checker = dfc.ConstraintsChecker(constraints)
            try:
                constraints_checker.check(df)
            except Exception as ex:
                assert str(dtype_object_constrain_types) in str(ex)  # Check message text is helpful.
                raise

    @pytest.mark.parametrize(
        "df, expectation",
        [
            (
                pd.DataFrame(
                    {
                        "col_0": ["a", "b", "c"],  # dtype: object.
                        "col_1": [11.0, 22.0, 33.0],  # dtype: float, does not affect this test.
                    }
                ),
                does_not_raise(),
            ),
            (
                pd.DataFrame(
                    {
                        "col_0": ["a", ("x", "y"), "c"],  # dtype: object.
                        "col_1": [11.0, 22.0, 33.0],  # dtype: float, does not affect this test.
                    }
                ),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_enforce_homogenous_type_per_column(self, df, expectation):
        with expectation:
            constraints = dfc.Constraints(on_elements=dfc.ElementConstraints(enforce_homogenous_type_per_column=True))
            constraints_checker = dfc.ConstraintsChecker(constraints)
            constraints_checker.check(df)


# --- Edge cases ---


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame([[], [], []]),
        pd.DataFrame(columns=[1, 2, 3]),
        pd.DataFrame(index=[1, 2, 3]),
    ],
)
def test_empty_df_index_constraints(df):
    constraints = dfc.Constraints(
        on_index=dfc.IndexConstraints(
            types=(pd.Index,),
            dtypes=(int, float, object),
            dtype_object_constrain_types=(str,),
            enforce_monotonic_increasing=True,
            enforce_unique=True,
            enforce_not_multi_index=True,
        )
    )
    constraints_checker = dfc.ConstraintsChecker(constraints)
    constraints_checker.check(df)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame([[], [], []]),
        pd.DataFrame(columns=[1, 2, 3]),
        pd.DataFrame(index=[1, 2, 3]),
    ],
)
def test_empty_df_column_constraints(df):
    constraints = dfc.Constraints(
        on_columns=dfc.IndexConstraints(
            types=(pd.Index,),
            dtypes=(int, float, object),
            dtype_object_constrain_types=(str,),
            enforce_monotonic_increasing=True,
            enforce_unique=True,
            enforce_not_multi_index=True,
        )
    )
    constraints_checker = dfc.ConstraintsChecker(constraints)
    constraints_checker.check(df)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame([[], [], []]),
        pd.DataFrame(columns=[1, 2, 3]),
        pd.DataFrame(index=[1, 2, 3]),
    ],
)
def test_empty_df_element_constraints(df):
    constraints = dfc.Constraints(
        on_elements=dfc.ElementConstraints(
            dtypes=(int, float, object), dtype_object_constrain_types=(str,), enforce_homogenous_type_per_column=True
        )
    )
    constraints_checker = dfc.ConstraintsChecker(constraints)
    constraints_checker.check(df)
