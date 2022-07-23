import typing

import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data.dataformat import StaticSamples, TimeSeries, TimeSeriesSamples
from clairvoyance2.data.feature import FeatureType

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def df_numeric():
    return pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 2]})


@pytest.fixture
def three_numeric_dfs():
    df_0 = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    df_1 = pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0]})
    df_2 = pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]})
    return (df_0, df_1, df_2)


# NOTE: For now, only integration tests that make sure the whole dataformat ecosystem works.
class TestIntegration:
    class TestTimeSeries:
        # TODO: Test timeseries with DatetimeIndex.
        @pytest.mark.parametrize(
            "data, categorical_features",
            [
                (np.asarray([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]]), tuple()),
                (pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]}), tuple()),
                (
                    pd.DataFrame(
                        {"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]},
                        index=pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                    ),
                    tuple(),
                ),
                (pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]}), ["col_2"]),
                (
                    pd.DataFrame({"col_1": [2, 2, 2], "col_2": ["a", "b", "c"]}),
                    {"col_1": [1, 2], "col_2": ["a", "b", "c"]},
                ),
            ],
        )
        def test_init_success(self, data, categorical_features):
            TimeSeries(data=data, categorical_features=categorical_features)

        @pytest.mark.parametrize(
            "data, categorical_features",
            [
                (np.asarray([[1.0, 2.0, 3.0], [11.0, "a", 13.0]]), tuple()),
                # ^ Non-homogenous type columns.
                (pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, ["some", "list"], 3]}), tuple()),
                # ^ Non-homogenous type columns.
                (pd.DataFrame({"col_1": [1.0, 2.0, 3.0]}, columns=[("tu", "ple")]), tuple()),
                # Wrong kind of column index.
                (pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]}), tuple()),
                # ^ Categorical column given but not specified.
                (pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]}), {"col_2": ["a"]}),
                # ^ Wrong categories provided.
            ],
        )
        def test_init_exception(self, data, categorical_features):
            with pytest.raises((TypeError, ValueError)):
                TimeSeries(data=data, categorical_features=categorical_features)

        def test_df(self, df_numeric):
            ts = TimeSeries(data=df_numeric)

            df_call_output = ts.df

            assert id(df_call_output) == id(df_numeric)

        @pytest.mark.slow
        def test_plot_runs(self, df_numeric):
            TimeSeries(data=df_numeric).plot()

        def test_features(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]})
            categorical_features = ["col_2"]
            ts = TimeSeries(data=data, categorical_features=categorical_features)

            features = ts.features

            assert isinstance(features, dict)
            assert len(features) == 2
            assert features["col_1"].feature_type == FeatureType.NUMERIC
            assert features["col_2"].feature_type == FeatureType.CATEGORICAL

        # --- Indexing-related ---

        def test_len(self, df_numeric):
            ts = TimeSeries(data=df_numeric)
            length = len(ts)
            assert length == 3

        class TestGetItem:
            def test_nonslice_key(self):
                data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
                ts = TimeSeries(data=data)

                ts_0 = ts[0]
                ts_2 = ts[2]

                assert isinstance(ts_0, pd.Series)
                assert isinstance(ts_2, pd.Series)
                assert (ts_0.values == [1.0, 1]).all()
                assert (ts_2.values == [3.0, 3]).all()

            def test_nonslice_key_raises_type_error(self, df_numeric):
                ts = TimeSeries(data=df_numeric)

                with pytest.raises(TypeError) as excinfo:
                    _ = ts["str_key"]
                assert "key is of inappropriate type" in str(excinfo.value).lower()

            @pytest.mark.parametrize("key", [12, 12.5])
            def test_nonslice_key_raises_key_error(self, key, df_numeric):
                ts = TimeSeries(data=df_numeric)

                with pytest.raises(KeyError):
                    _ = ts[key]

            @pytest.mark.parametrize(
                "key, expected_length",
                [
                    (slice(0, 100), 3),
                    (slice(0, 1), 2),
                    (slice(0, 0), 1),
                    (slice(0.5, 0.8), 0),
                ],
            )
            def test_slice_key(self, key, expected_length, df_numeric):
                # NOTE: This behavior may change. Currently using .loc[key, :].
                ts = TimeSeries(data=df_numeric)

                sliced = ts[key]

                assert isinstance(sliced, TimeSeries)
                assert len(sliced._data) == expected_length  # pylint: disable=protected-access

        def test_iter(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ts = TimeSeries(data=data)

            items = []
            for s in ts:
                items.append(s)

            assert all(isinstance(i, pd.Series) for i in items)
            assert (items[0].values == [1.0, 1]).all()
            assert (items[2].values == [3.0, 3]).all()

        def test_reversed(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ts = TimeSeries(data=data)

            items = []
            for s in reversed(ts):
                items.append(s)

            assert all(isinstance(i, pd.Series) for i in items)
            assert (items[0].values == [3.0, 3]).all()
            assert (items[2].values == [1.0, 1]).all()

        def test_contains(self):
            # NOTE: Only testing the __contains__() test for index.
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]}, index=[0, 1, 2])
            ts = TimeSeries(data=data)

            is_1_in_ts = 1 in ts
            is_9_in_ts = 9 in ts

            assert is_1_in_ts is True
            assert is_9_in_ts is False

        def test_index_count_methods_not_implemented(self, df_numeric):
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

    class TestTimeSeriesSamples:
        # TODO: Test initialisation from 3D numpy array.
        class TestInit:
            def test_success_numeric_only(self, three_numeric_dfs):
                TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            def test_success_numeric_and_categorical(self):
                categorical_def = {"b": ["1", "2"]}
                ts_0 = TimeSeries(
                    data=pd.DataFrame({"a": [1, 2, 3], "b": ["1", "2", "1"]}), categorical_features=categorical_def
                )
                ts_1 = TimeSeries(
                    data=pd.DataFrame({"a": [7, 8, 9], "b": ["1", "1", "1"]}), categorical_features=categorical_def
                )
                TimeSeriesSamples(data=[ts_0, ts_1])

            def test_success_numeric_and_categorical_explicit_categories_passed(self):
                categorical_def = {"b": ["1", "2"]}
                ts_0 = TimeSeries(
                    data=pd.DataFrame({"a": [1, 2, 3], "b": ["1", "2", "1"]}), categorical_features=categorical_def
                )
                ts_1 = TimeSeries(
                    data=pd.DataFrame({"a": [7, 8, 9], "b": ["1", "1", "1"]}), categorical_features=categorical_def
                )
                TimeSeriesSamples(data=[ts_0, ts_1], categorical_features=categorical_def)

            def test_fails_categories_mismatch(self):
                ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}))
                ts_1 = TimeSeries(
                    data=pd.DataFrame({"a": [7, 8, 9], "b": ["1", "1", "1"]}), categorical_features={"b": ["1"]}
                )

                with pytest.raises(TypeError) as excinfo:
                    TimeSeriesSamples(data=[ts_0, ts_1])
                assert "incompatible" in str(excinfo.value).lower()

        def test_data(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            _data = tss._data  # pylint: disable=protected-access

            assert isinstance(_data, pd.DataFrame)
            assert len(_data) == 2
            assert isinstance(_data.iloc[0, 0], pd.Series)

        def test_df_getter(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            df = tss.df
            _data = tss._data  # pylint: disable=protected-access

            assert id(df) == id(_data)

        def test_df_setter(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            with pytest.raises(AttributeError) as excinfo:
                tss.df = "some_value"
            assert "not set" in str(excinfo.value).lower()

        def test_features(self):
            ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 2]}), categorical_features=["b"])
            ts_1 = TimeSeries(data=pd.DataFrame({"a": [7, 8, 9], "b": [2, 2, 2]}), categorical_features=["b"])
            tss = TimeSeriesSamples(data=[ts_0, ts_1])

            features = tss.features

            assert isinstance(features, dict)
            assert len(features) == 2
            assert features["a"].feature_type == FeatureType.NUMERIC
            assert features["b"].feature_type == FeatureType.CATEGORICAL

        # --- Indexing-related ---

        def test_len(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])
            length = len(tss)
            assert length == 2

        def test_iter(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            items = []
            for s in tss:
                items.append(s)

            assert all(isinstance(i, TimeSeries) for i in items)
            assert len(items) == 2
            assert all(items[0]._data.columns == tss._data.columns)  # pylint: disable=protected-access

        def test_reversed(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            items = []
            for s in reversed(tss):
                items.append(s)

            assert all(isinstance(i, TimeSeries) for i in items)
            assert len(items) == 2
            assert all(items[0]._data.columns == tss._data.columns)  # pylint: disable=protected-access

        def test_getitem_nonslice_key(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])
            tss_0 = tss[0]
            assert isinstance(tss_0, TimeSeries)

        def test_getitem_slice_key(self, three_numeric_dfs):
            tss = TimeSeriesSamples(
                [
                    TimeSeries(three_numeric_dfs[0]),
                    TimeSeries(three_numeric_dfs[1]),
                    TimeSeries(three_numeric_dfs[2]),
                ]
            )
            sliced = tss[slice(1, 100)]
            assert isinstance(sliced, TimeSeriesSamples)
            assert len(sliced._data_internal_) == 2  # pylint: disable=protected-access
            assert all([1, 2] == sliced._data.index)  # pylint: disable=protected-access

        def test_contains(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            is_1_in_ts = 1 in tss
            is_9_in_ts = 9 in tss

            assert is_1_in_ts is True
            assert is_9_in_ts is False

        def test_index_count_methods_not_implemented(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

            with pytest.raises(NotImplementedError):
                tss.count(None)

            with pytest.raises(NotImplementedError):
                tss.index(None)

        # --- Indexing-related (End) ---

        def test_n_samples(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])
            n_samples = tss.n_samples
            assert n_samples == 2

        def test_n_timesteps_per_sample(self):
            ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2]}))
            ts_1 = TimeSeries(data=pd.DataFrame({"a": [7, 8, 9]}))
            ts_2 = TimeSeries(data=pd.DataFrame({"a": [11, 12, 12, 19]}))
            tss = TimeSeriesSamples(data=[ts_0, ts_1, ts_2])

            n_timesteps = tss.n_timesteps_per_sample

            assert n_timesteps == [2, 3, 4]

        @pytest.mark.slow
        def test_plot_runs(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])
            tss.plot(n=1)
            tss.plot()

    class TestStaticSamples:
        def test_init(self, df_numeric):
            StaticSamples(df_numeric)

        # --- Indexing-related ---

        def test_len(self, df_numeric):
            ss = StaticSamples(df_numeric)
            length = len(ss)
            assert length == 3

        def test_nonslice_key(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            ss_0 = ss[0]
            ss_2 = ss[2]

            assert isinstance(ss_0, pd.Series)
            assert isinstance(ss_2, pd.Series)
            assert (ss_0.values == [1.0, 1]).all()
            assert (ss_2.values == [3.0, 3]).all()

        def test_slice_key(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            sliced = ss[2:]

            assert isinstance(sliced, StaticSamples)
            assert len(sliced._data) == 1  # pylint: disable=protected-access

        def test_iter(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            items = []
            for s in ss:
                items.append(s)

            assert all(isinstance(i, pd.Series) for i in items)
            assert (items[0].values == [1.0, 1]).all()
            assert (items[2].values == [3.0, 3]).all()

        def test_reversed(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            items = []
            for s in reversed(ss):
                items.append(s)

            assert all(isinstance(i, pd.Series) for i in items)
            assert (items[0].values == [3.0, 3]).all()
            assert (items[2].values == [1.0, 1]).all()

        def test_contains(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": [1, 2, 3]})
            ss = StaticSamples(data)

            is_1_in_ts = 1 in ss
            is_9_in_ts = 9 in ss

            assert is_1_in_ts is True
            assert is_9_in_ts is False

        def test_index_count_methods_not_implemented(self, df_numeric):
            ss = StaticSamples(df_numeric)

            with pytest.raises(NotImplementedError):
                ss.count(None)

            with pytest.raises(NotImplementedError):
                ss.index(None)

        # --- Indexing-related (End) ---

        def test_n_samples(self, df_numeric):
            ss = StaticSamples(df_numeric)
            n_samples = ss.n_samples
            assert n_samples == 3

        def test_features(self):
            data = pd.DataFrame({"col_1": [1.0, 2.0, 3.0], "col_2": ["a", "b", "c"]})
            categorical_features = ["col_2"]
            ss = StaticSamples(data=data, categorical_features=categorical_features)

            features = ss.features

            assert isinstance(features, dict)
            assert len(features) == 2
            assert features["col_1"].feature_type == FeatureType.NUMERIC
            assert features["col_2"].feature_type == FeatureType.CATEGORICAL
