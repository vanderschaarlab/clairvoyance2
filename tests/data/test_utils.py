import pandas as pd

from clairvoyance2.data.utils import check_index_regular, split_multi_index_dataframe


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


def test_split_multi_index_dataframe():
    # Arrange.
    multi_df = pd.DataFrame(
        {"a": [1, 2, 3, 7, 8, 9, -1, -2, -3], "b": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0]},
        index=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    )

    # Act.
    split_dataframes = tuple(split_multi_index_dataframe(multi_df))

    # Assert.
    expected_df_0 = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    expected_df_1 = pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0]})
    expected_df_2 = pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]})
    for idx, expected_df in enumerate([expected_df_0, expected_df_1, expected_df_2]):
        assert (expected_df == split_dataframes[idx]).all().all()
        assert (expected_df.columns == split_dataframes[idx].columns).all()
        assert (expected_df.index == split_dataframes[idx].index).all()
