import typing

import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data.dataformat import TimeSeries

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def df_numeric():
    return pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 2]})


# NOTE: For now, only integration tests that make sure the whole dataformat ecosystem works.
class TestIntegration:
    # TODO: Test timeseries with DatetimeIndex.
    @pytest.mark.parametrize(
        "data",
        [
            np.asarray([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]]),
            pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]}),
            pd.DataFrame(
                {"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]},
                index=pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
            ),
            pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]}),
            pd.DataFrame({"col_1": [], "col_2": []}),  # Empty case.
            pd.DataFrame({"col_1": [], "col_2": []}, dtype=int),  # Empty case w/ dtype set
            pd.DataFrame({"col_1": [2, 2, 2], "col_2": ["a", "b", "c"]}),
        ],
    )
    def test_init_success(self, data):
        TimeSeries(data=data)

    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, ["some", "list"], 3]}),
            # ^ Non-homogenous type columns.
            pd.DataFrame({"col_1": [1.0, 2.0, 3.0]}, columns=[("tu", "ple")]),
            # Wrong kind of column index.
        ],
    )
    def test_init_exception(self, data):
        with pytest.raises((TypeError, ValueError)):
            TimeSeries(data=data)

    def test_df(self, df_numeric):
        ts = TimeSeries(data=df_numeric)

        df_call_output = ts.df

        assert id(df_call_output) == id(df_numeric)

    @pytest.mark.parametrize(
        "df, expected_result",
        [
            (pd.DataFrame({"col_1": [1.0, np.nan, 3.0], "col_2": [1, 2, np.nan]}), True),
            (pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]}), False),
        ],
    )
    def test_has_missing(self, df, expected_result):
        ts = TimeSeries(data=df)
        has_missing = ts.has_missing
        assert has_missing is expected_result

    @pytest.mark.slow
    def test_plot_runs(self, df_numeric):
        TimeSeries(data=df_numeric).plot()

    def test_features(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]})
        ts = TimeSeries(data=data)

        features = ts.features

        assert isinstance(features, dict)
        assert len(features) == 2
        assert features["col_1"].numeric_compatible
        assert features["col_2"].categorical_compatible

    # --- Indexing-related ---

    def test_len(self, df_numeric):
        ts = TimeSeries(data=df_numeric)
        length = len(ts)
        assert length == 3

    def test_len_empty(self, df_numeric):
        ts = TimeSeries(data=df_numeric[:0].copy())
        length = len(ts)
        assert ts.empty
        assert length == 0

    class TestGetItem:
        def test_single_indexer_item(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ts = TimeSeries(data=data)

            ts_0 = ts[0]
            ts_2 = ts[2]

            assert isinstance(ts_0, TimeSeries)
            assert isinstance(ts_2, TimeSeries)
            assert (ts_0.df.values == [1.0, 1]).all()
            assert (ts_2.df.values == [3.0, 3]).all()

        def test_single_indexer_item_raises_error_from_pd(self, df_numeric):
            # NOTE: Error comes from pandas.
            ts = TimeSeries(data=df_numeric)

            with pytest.raises(KeyError) as excinfo:
                _ = ts["unknown_key"]
            assert "unknown_key" in str(excinfo.value).lower()

        @pytest.mark.parametrize("key", [12, 12.5])
        def test_single_indexer_item_raises_key_error(self, key, df_numeric):
            ts = TimeSeries(data=df_numeric)

            with pytest.raises(KeyError):
                _ = ts[key]

        def test_single_indexer_slice(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ts = TimeSeries(data=data)

            sliced = ts[1:]

            assert isinstance(sliced, TimeSeries)
            assert (sliced.df == pd.DataFrame({"col_1": [2.0, 3.0], "col_2": [2, 3]}, index=[1, 2])).all().all()

        @pytest.mark.parametrize(
            "key, expected_length",
            [
                (slice(0, 100), 3),
                (slice(0, 1), 2),
                (slice(0, 0), 1),
                (slice(0.5, 0.8), 0),
            ],
        )
        def test_single_indexer_slice_length(self, key, expected_length, df_numeric):
            ts = TimeSeries(data=df_numeric)

            sliced = ts[key]

            assert isinstance(sliced, TimeSeries)
            assert len(sliced.df) == expected_length

        def test_single_indexer_iterable_len_1(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [-1, -2, -3]})
            ts = TimeSeries(data=data)

            sliced = ts[(1,)]

            assert isinstance(sliced, TimeSeries)
            assert (sliced.df == pd.DataFrame({"col_1": [2.0], "col_2": [-2]}, index=[1])).all().all()

        def test_single_indexer_iterable_len_2(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ts = TimeSeries(data=data)

            sliced = ts[(0, 2)]

            assert isinstance(sliced, TimeSeries)
            assert (sliced.df == pd.DataFrame({"col_1": [1.0, 3.0], "col_2": [1, 3]}, index=[0, 2])).all().all()

        def test_single_indexer_iterable_len_2_alternative_syntax(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ts = TimeSeries(data=data)

            sliced = ts[0, 2]

            assert isinstance(sliced, TimeSeries)
            assert (sliced.df == pd.DataFrame({"col_1": [1.0, 3.0], "col_2": [1, 3]}, index=[0, 2])).all().all()

        def test_single_indexer_iterable_len_3(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [-1, -2, -3]})
            ts = TimeSeries(data=data)

            sliced = ts[(0, 1, 2)]

            assert isinstance(sliced, TimeSeries)
            assert (sliced.df == pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [-1, -2, -3]})).all().all()

        def test_two_indexers_item_item(self):
            df = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11, 12, 13], "col_3": [-1.0, -2.0, -3.0]})
            ts = TimeSeries(data=df)
            ts_new = ts[2, "col_3"]

            assert isinstance(ts_new, TimeSeries)
            assert len(ts_new) == 1
            assert (ts_new.df.values == [-3.0]).all()

        def test_two_indexers_item_slice(self):
            df = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11, 12, 13], "col_3": [-1.0, -2.0, -3.0]})
            ts = TimeSeries(data=df)
            ts_new = ts[2, :"col_2"]

            assert isinstance(ts_new, TimeSeries)
            assert len(ts_new) == 1
            assert (ts_new.df.values == [3.0, 13]).all()

        def test_two_indexers_item_iterable(self):
            df = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11, 12, 13], "col_3": [-1.0, -2.0, -3.0]})
            ts = TimeSeries(data=df)
            ts_new = ts[2, ("col_1", "col_3")]

            assert isinstance(ts_new, TimeSeries)
            assert len(ts_new) == 1
            assert (ts_new.df == pd.DataFrame({"col_1": [3.0], "col_3": [-3.0]}, index=[2])).all().all()

        def test_two_indexers_slice_slice(self):
            df = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 2], "col_3": [3.0, 3.0, 3.0]})
            ts = TimeSeries(data=df)
            ts_new = ts[1:, "col_2":]

            assert len(ts_new.df) == 2
            assert list(ts_new.time_index) == [1, 2]
            assert list(ts_new.df.columns) == ["col_2", "col_3"]
            assert "col_2" in ts_new.features and "col_3" in ts_new.features

        def test_two_indexers_iterable_iterable(self):
            df = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 2], "col_3": [3.0, 3.0, 3.0]})
            ts = TimeSeries(data=df)
            ts_new = ts[(0, 1), ("col_2",)]

            assert isinstance(ts_new, TimeSeries)
            assert len(ts_new) == 2
            assert (ts_new.df == pd.DataFrame({"col_2": [1, 2]}, index=[0, 1])).all().all()

        def test_two_indexers_iterable_slice(self):
            df = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 2], "col_3": [3.0, 3.0, 3.0]})
            ts = TimeSeries(data=df)
            ts_new = ts[(0, 1), "col_2":]

            assert isinstance(ts_new, TimeSeries)
            assert len(ts_new) == 2
            assert (ts_new.df == pd.DataFrame({"col_2": [1, 2], "col_3": [3.0, 3.0]}, index=[0, 1])).all().all()

    def test_iter(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
        ts = TimeSeries(data=data)

        items = []
        for s in ts:
            items.append(s)

        assert all(isinstance(i, TimeSeries) for i in items)
        assert (items[0].df.values == [1.0, 1]).all()
        assert (items[2].df.values == [3.0, 3]).all()

    def test_reversed(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
        ts = TimeSeries(data=data)

        items = []
        for s in reversed(ts):
            items.append(s)

        assert all(isinstance(i, TimeSeries) for i in items)
        assert (items[0].df.values == [3.0, 3]).all()
        assert (items[2].df.values == [1.0, 1]).all()

    def test_contains(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]}, index=[0, 1, 2])
        ts = TimeSeries(data=data)

        is_1_in_ts = 1 in ts
        is_9_in_ts = 9 in ts

        assert is_1_in_ts is True
        assert is_9_in_ts is False

    def test_py_sequence_some_methods_not_implemented(self, df_numeric):
        ts = TimeSeries(data=df_numeric)

        with pytest.raises(NotImplementedError):
            ts.count(None)

        with pytest.raises(NotImplementedError):
            ts.index(None)

    # --- Indexing-related (End) ---

    def test_implements_interfaces(self):
        assert issubclass(TimeSeries, typing.Container)
        assert issubclass(TimeSeries, typing.Iterable)
        assert issubclass(TimeSeries, typing.Collection)
        assert issubclass(TimeSeries, typing.Sized)
        assert issubclass(TimeSeries, typing.Sequence)

    class TestToNumpy:
        def test_no_change_shape(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ts = TimeSeries(data)

            array = ts.to_numpy(padding_indicator=-999.0, max_len=None)

            assert (ts.df.values == array).all()

        def test_no_change_shape_empty(self):
            data = pd.DataFrame({"col_1": [], "col_2": []})
            ts = TimeSeries(data)

            array = ts.to_numpy(padding_indicator=-999.0, max_len=None)

            assert (ts.df.values == array).all()
            assert array.size == 0
            assert array.shape == (0, 2)

        def test_extend(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ts = TimeSeries(data)

            array = ts.to_numpy(padding_indicator=-999.0, max_len=10)

            assert array.shape == (10, 2)
            assert (ts.df.values == array[:3, :]).all()  # pylint: disable=unsubscriptable-object
            assert (array[3:, :] == -999.0).all()  # pylint: disable=unsubscriptable-object

        def test_extend_empty(self):
            data = pd.DataFrame({"col_1": [], "col_2": []})
            ts = TimeSeries(data)

            array = ts.to_numpy(padding_indicator=-999.0, max_len=10)

            assert array.shape == (10, 2)
            assert (array == -999.0).all()  # pylint: disable=E

        def test_shrink(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ts = TimeSeries(data)

            array = ts.to_numpy(padding_indicator=-999.0, max_len=2)

            assert array.shape == (2, 2)
            assert (ts.df.values[:2, :] == array).all()

        def test_raise_found_padding_indicator(self):
            data = pd.DataFrame({"col_1": [1.0, -999.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ts = TimeSeries(data)

            with pytest.raises(ValueError) as excinfo:
                ts.to_numpy(padding_indicator=-999.0, max_len=None)
            assert "found in" in str(excinfo.value).lower()

        def test_time_index_no_change_shape(self):
            ts = TimeSeries(pd.DataFrame({"a": [10, 20, 30, 40]}, index=[1, 2, 3, 4]))

            np_time_index = ts.to_numpy_time_index(max_len=None)

            assert (np_time_index == np.asarray([[1, 2, 3, 4]]).T).all()

        def test_time_index_extend(self):
            ts = TimeSeries(pd.DataFrame({"a": [10, 20, 30, 40]}, index=[1, 2, 3, 4]))
            pi = -777.0

            np_time_index = ts.to_numpy_time_index(padding_indicator=pi, max_len=6)

            assert (np_time_index == np.asarray([[1, 2, 3, 4, pi, pi]]).T).all()

        def test_time_index_success_shrink(self):
            ts = TimeSeries(pd.DataFrame({"a": [10, 20, 30, 40]}, index=[1, 2, 3, 4]))

            np_time_index = ts.to_numpy_time_index(max_len=2)

            assert (np_time_index == np.asarray([[1, 2]]).T).all()

        def test_time_index_raise_found_padding_indicator(self):
            ts = TimeSeries(pd.DataFrame({"a": [10, 20, 30, 40]}, index=[1, 2, 3, 777]))

            with pytest.raises(ValueError) as excinfo:
                ts.to_numpy_time_index(padding_indicator=777.0, max_len=None)
            assert "found in" in str(excinfo.value).lower()

    def test_copy(self):
        data = pd.DataFrame({"col_1": [1.0, -999.0, np.nan], "col_2": ["a", "b", "a"]})
        ts = TimeSeries(data, missing_indicator=np.nan)

        ts_copy = ts.copy()
        ts_copy.df.loc[0, "col_1"] = 12345.0

        assert id(ts_copy) != id(ts)
        assert id(ts_copy.df) != id(ts.df)
        assert id(ts_copy.df) != id(ts.df)
        assert ts.df.loc[0, "col_1"] == 1.0
        assert ts_copy.df.loc[0, "col_1"] == 12345.0

    def test_n_timesteps(self, df_numeric):
        ts = TimeSeries(data=df_numeric)
        length = ts.n_timesteps
        assert length == 3

    def test_new_like(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]}, index=[2, 3, 8])
        ts = TimeSeries(data)

        new = ts.new_like(ts, data=data + 1)

        assert isinstance(new, TimeSeries)
        assert new.n_features == ts.n_features
        assert new.df.shape == ts.df.shape
        assert list(new.features.keys()) == list(ts.features.keys())
        assert (new.df.dtypes == ts.df.dtypes).all()
        assert (new.df == data + 1).all().all()

    def test_new_empty_like(self):
        data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]}, index=[2, 3, 8])
        ts = TimeSeries(data)

        new = ts.new_empty_like(ts)

        assert isinstance(new, TimeSeries)
        assert new.n_features == ts.n_features
        assert new.df.shape == (0, ts.df.shape[1])
        assert list(new.features.keys()) == list(ts.features.keys())
        assert (new.df.dtypes == ts.df.dtypes).all()
        assert new.df.empty is True

    class TestMutation:
        def test_mutate_df_loc(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ts = TimeSeries(data)

            ts.df.loc[0, "col_1"] = 999.0

            assert ts.df.loc[0, "col_1"] == 999.0
            assert ts._data.loc[0, "col_1"] == 999.0  # pylint: disable=protected-access

        def test_mutate_df_slice_all(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ts = TimeSeries(data)

            ts.df[:] = np.asarray([[9.0, 8.0, 6.0], [-1.0, -2.0, -3.0]]).T

            assert (ts.df == pd.DataFrame({"col_1": [9.0, 8.0, 6.0], "col_2": [-1.0, -2.0, -3.0]})).all().all()
            temp = ts._data  # pylint: disable=protected-access
            assert (temp == pd.DataFrame({"col_1": [9.0, 8.0, 6.0], "col_2": [-1.0, -2.0, -3.0]})).all().all()

        def test_mutate_df_reassign(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [11.0, 22.0, 33.0]})
            ts = TimeSeries(data)

            ts.df = pd.DataFrame({"col_1": [-1.0, -2.0, -3.0], "col_2": [-11.0, -22.0, -33.0]})

            assert (ts.df == pd.DataFrame({"col_1": [-1.0, -2.0, -3.0], "col_2": [-11.0, -22.0, -33.0]})).all().all()
            temp = ts._data  # pylint: disable=protected-access
            assert (temp == pd.DataFrame({"col_1": [-1.0, -2.0, -3.0], "col_2": [-11.0, -22.0, -33.0]})).all().all()
