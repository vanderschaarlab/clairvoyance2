import pandas as pd
import pytest

from clairvoyance2.data.internal_utils import (
    check_index_regular,
    df_align_and_overwrite,
)


class TestCheckIndexRegular:
    def test_empty_index(self):
        index = pd.Index([], dtype=int)
        is_regular, diff = check_index_regular(index)
        assert is_regular is True
        assert diff is None

    def test_single_item_index(self):
        index = pd.Index([5], dtype=int)
        is_regular, diff = check_index_regular(index)
        assert is_regular is True
        assert diff is None

    def test_two_item_index(self):
        index = pd.Index([5, 10], dtype=int)
        is_regular, diff = check_index_regular(index)
        assert is_regular is True
        assert diff == 5

    def test_regular_range_index(self):
        index = pd.RangeIndex(start=0, stop=10, step=1)
        is_regular, diff = check_index_regular(index)
        assert is_regular is True
        assert diff == 1

    def test_regular_numeric_index_float(self):
        index = pd.Index([1.0, 2.0, 3.0, 4.0], dtype=float)
        is_regular, diff = check_index_regular(index)
        assert is_regular is True
        assert diff == 1.0

    def test_regular_datetime_index(self):
        index = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"])
        is_regular, diff = check_index_regular(index)
        assert is_regular is True
        assert isinstance(diff, pd.Timedelta)
        assert diff == pd.Timedelta(1, "d")

    def test_irregular_numeric_index_float(self):
        index = pd.Index([1.0, 7.0, 9.0, 24.0], dtype=float)
        is_regular, diff = check_index_regular(index)
        assert is_regular is False
        assert diff is None

    def test_irregular_datetime_index(self):
        index = pd.to_datetime(["2000-01-01", "2000-01-20", "2000-09-03"])
        is_regular, diff = check_index_regular(index)
        assert is_regular is False
        assert diff is None


class TestDfAlignAndOverwrite:
    @pytest.mark.parametrize(
        "df_to_update, df_with_new_data, expected_df",
        [
            # Case: all elements updated.
            (
                # df_to_update:
                pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]),
                # df_with_new_data:
                pd.DataFrame({"a": [10, 20, 30], "b": [-10, -20, -30]}, index=[1, 2, 3]),
                # expected_df:
                pd.DataFrame({"a": [10, 20, 30], "b": [-10, -20, -30]}, index=[1, 2, 3]),
            ),
            # Case: some elements updated.
            (
                # df_to_update:
                pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]),
                # df_with_new_data:
                pd.DataFrame({"a": [30], "b": [-30]}, index=[3]),
                # expected_df:
                pd.DataFrame({"a": [1, 2, 30], "b": [-1, -2, -30]}, index=[1, 2, 3]),
            ),
            # Case: some elements updated, index extended.
            (
                # df_to_update:
                pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]),
                # df_with_new_data:
                pd.DataFrame({"a": [30, 70], "b": [-30, -70]}, index=[3, 7]),
                # expected_df:
                pd.DataFrame({"a": [1, 2, 30, 70], "b": [-1, -2, -30, -70]}, index=[1, 2, 3, 7]),
            ),
            # Case: some elements updated in index extended.
            (
                # df_to_update:
                pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]),
                # df_with_new_data:
                pd.DataFrame({"a": [70, 80], "b": [-70, -80]}, index=[7, 8]),
                # expected_df:
                pd.DataFrame({"a": [1, 2, 3, 70, 80], "b": [-1, -2, -3, -70, -80]}, index=[1, 2, 3, 7, 8]),
            ),
        ],
    )
    def test_success(self, df_to_update, df_with_new_data, expected_df):
        df = df_align_and_overwrite(df_to_update=df_to_update, df_with_new_data=df_with_new_data)
        assert (df == expected_df).all().all()
