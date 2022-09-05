import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data.dataformat import TimeSeries, TimeSeriesSamples

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


@pytest.fixture
def three_numeric_dfs():
    df_0 = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    df_1 = pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0]})
    df_2 = pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]})
    return (df_0, df_1, df_2)


# NOTE: For now, only integration tests that make sure the whole dataformat ecosystem works.
class TestIntegration:
    # TODO: Test initialisation from 3D numpy array.
    class TestInit:
        def test_success_numeric_only(self, three_numeric_dfs):
            TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        def test_success_numeric_and_categorical(self):
            ts_0 = TimeSeries(
                data=pd.DataFrame({"a": [1, 2, 3], "b": ["1", "2", "1"]}),
            )
            ts_1 = TimeSeries(
                data=pd.DataFrame({"a": [7, 8, 9], "b": ["1", "1", "1"]}),
            )
            TimeSeriesSamples(data=[ts_0, ts_1])

        @pytest.mark.parametrize("sample_indices", [[3, 4], [0, 1]])
        @pytest.mark.parametrize(
            "data",
            [
                [
                    pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 9.0, 1.0]}),
                    pd.DataFrame({"a": [7, 8, 9], "b": [3.0, 3.0, 4.0]}),
                ],
                np.zeros(shape=(2, 5, 3)),
            ],
        )
        def test_success_with_sample_indices(self, data, sample_indices):
            tss = TimeSeriesSamples(data=data, sample_indices=sample_indices)

            assert len(tss) == 2
            assert tss.sample_indices == sample_indices
            assert list(tss.df.index) == sample_indices
            assert list(tss.sample_index) == sample_indices

        def test_fails_sample_indices_wrong_length(
            self,
        ):
            data = [
                pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 9.0, 1.0]}),
                pd.DataFrame({"a": [7, 8, 9], "b": [3.0, 3.0, 4.0]}),
            ]
            sample_indices = [1, 7, 9]

            with pytest.raises(ValueError) as excinfo:
                TimeSeriesSamples(data=data, sample_indices=sample_indices)
            assert "did not match" in str(excinfo.value).lower() and "TimeSeriesSamples" in str(excinfo.value)

        def test_empty_false(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0])])
            assert tss.empty is False

    def test_has_missing(self):
        ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}))
        ts_1 = TimeSeries(data=pd.DataFrame({"a": [-1, -2, -3], "b": [-1.0, -2.0, -3.0]}))
        ts_missing = TimeSeries(data=pd.DataFrame({"a": [-1, np.nan, -3], "b": [-1.0, -2.0, np.nan]}))

        tss_no_missing = TimeSeriesSamples(data=[ts_0, ts_1])
        tss_missing = TimeSeriesSamples(data=[ts_0, ts_missing])

        expect_false = tss_no_missing.has_missing
        expect_true = tss_missing.has_missing

        assert expect_false is False
        assert expect_true is True

    def test_data(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        _data = tss.df

        assert isinstance(_data, pd.DataFrame)
        assert len(_data) == 2
        assert isinstance(_data.iloc[0, 0], pd.Series)

    def test_df_getter(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        df = tss.df
        _data = tss.df

        assert id(df) == id(_data)

    def test_df_setter(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        with pytest.raises(AttributeError) as excinfo:
            tss.df = "some_value"
        assert "not set" in str(excinfo.value).lower()

    def test_features(self):
        ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 2]}))
        ts_1 = TimeSeries(data=pd.DataFrame({"a": [7, 8, 9], "b": [2, 2, 2]}))
        tss = TimeSeriesSamples(data=[ts_0, ts_1])

        features = tss.features

        assert isinstance(features, dict)
        assert len(features) == 2
        assert features["a"].numeric_compatible
        assert features["b"].categorical_compatible

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
        assert all(items[0].df.columns == tss.df.columns)

    def test_reversed(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        items = []
        for s in reversed(tss):
            items.append(s)

        assert all(isinstance(i, TimeSeries) for i in items)
        assert len(items) == 2
        assert all(items[0].df.columns == tss.df.columns)

    class TestGetItem:
        def test_single_indexer_item(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])
            ts_0 = tss[0]
            assert isinstance(ts_0, TimeSeries)
            assert (ts_0.df == three_numeric_dfs[0]).all().all()

        def test_single_indexer_iterable_len_1(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])
            tss_0 = tss[(0,)]
            assert isinstance(tss_0, TimeSeriesSamples)
            assert tss_0.n_samples == 1
            assert (tss_0._internal[0].df == three_numeric_dfs[0]).all().all()  # pylint: disable=protected-access

        def test_single_indexer_iterable_len_1_alternative_syntax(self, three_numeric_dfs):
            tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])
            tss_0 = tss[[0]]
            assert isinstance(tss_0, TimeSeriesSamples)
            assert tss_0.n_samples == 1
            assert (tss_0._internal[0].df == three_numeric_dfs[0]).all().all()  # pylint: disable=protected-access

        def test_single_indexer_iterable_len_2(self, three_numeric_dfs):
            tss = TimeSeriesSamples(
                [
                    TimeSeries(three_numeric_dfs[0]),
                    TimeSeries(three_numeric_dfs[1]),
                    TimeSeries(three_numeric_dfs[2]),
                ]
            )
            tss_i = tss[(0, 2)]
            assert isinstance(tss_i, TimeSeriesSamples)
            assert tss_i.n_samples == 2
            assert (tss_i._internal[0].df == three_numeric_dfs[0]).all().all()  # pylint: disable=protected-access
            assert (tss_i._internal[1].df == three_numeric_dfs[2]).all().all()  # pylint: disable=protected-access

        def test_single_indexer_iterable_len_3(self, three_numeric_dfs):
            tss = TimeSeriesSamples(
                [
                    TimeSeries(three_numeric_dfs[0]),
                    TimeSeries(three_numeric_dfs[1]),
                    TimeSeries(three_numeric_dfs[2]),
                    TimeSeries(three_numeric_dfs[2]),
                ]
            )
            tss_i = tss[[0, 1, 3]]
            assert isinstance(tss_i, TimeSeriesSamples)
            assert tss_i.n_samples == 3
            assert (tss_i._internal[0].df == three_numeric_dfs[0]).all().all()  # pylint: disable=protected-access
            assert (tss_i._internal[1].df == three_numeric_dfs[1]).all().all()  # pylint: disable=protected-access
            assert (tss_i._internal[2].df == three_numeric_dfs[2]).all().all()  # pylint: disable=protected-access

        def test_single_indexer_slice(self, three_numeric_dfs):
            tss = TimeSeriesSamples(three_numeric_dfs)
            sliced = tss[slice(1, 100)]
            assert isinstance(sliced, TimeSeriesSamples)
            assert len(sliced._internal) == 2  # pylint: disable=protected-access
            assert all([1, 2] == sliced.sample_index)

        def test_two_indexers_item_item(self, three_numeric_dfs):
            tss = TimeSeriesSamples(three_numeric_dfs)
            tss_new = tss[2, "b"]

            assert isinstance(tss_new, TimeSeries)
            assert (three_numeric_dfs[2].loc[:, ["b"]] == tss_new.df).all().all()

        def test_two_indexers_item_slice(self):
            tss = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [11, 12, 13]}),
                    pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": [8, 12, 8]}),
                    pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [10, 9, 11]}),
                ]
            )
            tss_new = tss[1, "b":]

            assert isinstance(tss_new, TimeSeries)
            assert (
                (pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": [8, 12, 8]}).loc[:, "b":] == tss_new.df)
                .all()
                .all()
            )

        def test_two_indexers_item_iterable(self):
            tss = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [11, 12, 13]}),
                    pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": [8, 12, 8]}),
                    pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [10, 9, 11]}),
                ]
            )
            tss_new = tss[1, ("a", "c")]

            assert isinstance(tss_new, TimeSeries)
            assert (pd.DataFrame({"a": [7, 8, 9], "c": [8, 12, 8]}) == tss_new.df).all().all()

        def test_two_indexers_slice_slice(self):
            tss = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [11, 12, 13]}),
                    pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": [8, 12, 8]}),
                    pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [10, 9, 11]}),
                ]
            )
            tss_new = tss[1:, "b":]

            def get(idx):
                return tss_new._internal[idx]  # pylint: disable=protected-access

            assert isinstance(tss_new, TimeSeriesSamples)
            assert tss_new.n_samples == 2
            assert len(tss_new.df) == 2
            assert (get(0).df == pd.DataFrame({"b": [7.0, 8.0, 9.0], "c": [8, 12, 8]})).all().all()
            assert (get(1).df == pd.DataFrame({"b": [11.0, 12.0, 13.0], "c": [10, 9, 11]})).all().all()

        def test_two_indexers_iterable_iterable(self):
            tss = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [11, 12, 13]}),
                    pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": [8, 12, 8]}),
                    pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [10, 9, 11]}),
                ]
            )
            tss_new = tss[(0, 1), ("c",)]

            def get(idx):
                return tss_new._internal[idx]  # pylint: disable=protected-access

            assert isinstance(tss_new, TimeSeriesSamples)
            assert tss_new.n_samples == 2
            assert len(tss_new.df) == 2
            assert (get(0).df == pd.DataFrame({"c": [11, 12, 13]})).all().all()
            assert (get(1).df == pd.DataFrame({"c": [8, 12, 8]})).all().all()

        def test_two_indexers_iterable_slice(self):
            tss = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": [11, 12, 13]}),
                    pd.DataFrame({"a": [7, 8, 9], "b": [7.0, 8.0, 9.0], "c": [8, 12, 8]}),
                    pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [10, 9, 11]}),
                ]
            )
            tss_new = tss[(2,), "a":]

            def get(idx):
                return tss_new._internal[idx]  # pylint: disable=protected-access

            assert isinstance(tss_new, TimeSeriesSamples)
            assert tss_new.n_samples == 1
            assert len(tss_new.df) == 1
            assert (
                (get(0).df == pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [10, 9, 11]})).all().all()
            )

    def test_contains(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        is_1_in_ts = 1 in tss
        is_9_in_ts = 9 in tss

        assert is_1_in_ts is True
        assert is_9_in_ts is False

    def test_py_sequence_some_methods_not_implemented(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        with pytest.raises(NotImplementedError):
            tss.count(None)

        with pytest.raises(NotImplementedError):
            tss.index(None)

    def test_new_like(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        new = tss.new_like(tss, data=[TimeSeries(three_numeric_dfs[1]), TimeSeries(three_numeric_dfs[2])])

        assert isinstance(new, TimeSeriesSamples)
        assert new.n_features == tss.n_features
        assert new.n_samples == tss.n_samples
        assert new.df.shape == tss.df.shape
        assert list(new.features.keys()) == list(tss.features.keys())
        assert (new.df.dtypes == tss.df.dtypes).all()
        assert (new[0].df == TimeSeries(three_numeric_dfs[1]).df).all().all()
        assert (new[1].df == TimeSeries(three_numeric_dfs[2]).df).all().all()

    def test_new_empty_like(self, three_numeric_dfs):
        tss = TimeSeriesSamples([TimeSeries(three_numeric_dfs[0]), TimeSeries(three_numeric_dfs[1])])

        new = tss.new_empty_like(tss)

        assert isinstance(new, TimeSeriesSamples)
        assert new.n_features == tss.n_features
        assert new.n_samples == tss.n_features  # NOTE: This behavior!
        assert list(new.features.keys()) == list(tss.features.keys())
        assert (new.df.dtypes == tss.df.dtypes).all()
        assert new[0].df.empty is True
        assert new[1].df.empty is True

    class TestMutation:
        def test_mutate_df_directly(self):
            tss = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [11.0, 12.0, 13.0]}),
                    pd.DataFrame({"a": [7.0, 8.0, 9.0], "b": [7.0, 8.0, 9.0], "c": [8.0, 12.0, 8.0]}),
                    pd.DataFrame({"a": [-1.0, -2.0, -3.0], "b": [11.0, 12.0, 13.0], "c": [10.0, 9.0, 11.0]}),
                ]
            )

            # NOTE: This is a case where pandas doesn't return a copy but a view, so the value can be modified
            # directly. Pandas makes no guarantees of returning views vs copies in general.
            tss.df.loc[1, "a"][2] = 999.0

            assert tss.df.loc[1, "a"][2] == 999.0

        def test_assign_df_not_allowed(self, three_numeric_dfs):
            tss = TimeSeriesSamples(three_numeric_dfs)

            with pytest.raises(AttributeError) as excinfo:
                tss.df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
            assert "df" in str(excinfo.value)

        class TestMutateInnerTimeseries:
            def test_mutate_df_loc(self):
                tss = TimeSeriesSamples(
                    [
                        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [11.0, 12.0, 13.0]}),
                        pd.DataFrame({"a": [7.0, 8.0, 9.0], "b": [7.0, 8.0, 9.0], "c": [8.0, 12.0, 8.0]}),
                        pd.DataFrame({"a": [-1.0, -2.0, -3.0], "b": [11.0, 12.0, 13.0], "c": [10.0, 9.0, 11.0]}),
                    ]
                )

                ts = tss[1]
                ts.df.loc[1, "b"] = 999.0

                assert tss.df.loc[1, "b"][1] == 999.0
                assert tss._internal[1].df.loc[1, "b"] == 999.0  # pylint: disable=protected-access

            def test_mutate_df_slice_all(self):
                tss = TimeSeriesSamples(
                    [
                        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [11.0, 12.0, 13.0]}),
                        pd.DataFrame({"a": [7.0, 8.0, 9.0], "b": [7.0, 8.0, 9.0], "c": [8.0, 12.0, 8.0]}),
                        pd.DataFrame({"a": [-1.0, -2.0, -3.0], "b": [11.0, 12.0, 13.0], "c": [10.0, 9.0, 11.0]}),
                    ]
                )

                ts = tss[1]
                arr = np.asarray([[9.0, 8.0, 6.0], [-1.0, -2.0, -3.0], [99.0, 88.0, 22.0]]).T
                ts.df[:] = arr

                assert (tss.df.loc[1, "a"] == np.asarray([9.0, 8.0, 6.0])).all()
                assert (tss.df.loc[1, "b"] == np.asarray([-1.0, -2.0, -3.0])).all()
                assert (tss.df.loc[1, "c"] == np.asarray([99.0, 88.0, 22.0])).all()
                assert (tss._internal[1].df.values == arr).all()  # pylint: disable=protected-access

            def test_mutate_df_reassign(self):
                tss = TimeSeriesSamples(
                    [
                        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [11.0, 12.0, 13.0]}),
                        pd.DataFrame({"a": [7.0, 8.0, 9.0], "b": [7.0, 8.0, 9.0], "c": [8.0, 12.0, 8.0]}),
                        pd.DataFrame({"a": [-1.0, -2.0, -3.0], "b": [11.0, 12.0, 13.0], "c": [10.0, 9.0, 11.0]}),
                    ]
                )

                ts = tss[1]
                df = pd.DataFrame({"a": [-7.0, -8.0, -9.0], "b": [-7.0, -8.0, -9.0], "c": [-8.0, -12.0, -8.0]})
                ts.df = df

                assert (tss.df.loc[1, "a"] == pd.Series([-7.0, -8.0, -9.0])).all()
                assert (tss._internal[1].df == df).all().all()  # pylint: disable=protected-access

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

    class TestToNumpy:
        @pytest.mark.parametrize("max_len, expect_padding", [(2, False), (3, True), (4, True), (5, True)])
        def test_max_len_set(self, max_len, expect_padding):
            tss = TimeSeriesSamples(
                [
                    TimeSeries(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [11.0, 12.0, 13.0]})),
                    TimeSeries(pd.DataFrame({"a": [-1.0, -2.0], "b": [-11.0, -12.0]})),
                    TimeSeries(pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0], "b": [111.0, 122.0, 133.0, 144.0]})),
                ]
            )

            array = tss.to_numpy(padding_indicator=-999.0, max_len=max_len)

            assert array.shape == (3, max_len, 2)
            if expect_padding:
                assert -999.0 in array  # pylint: disable=unsupported-membership-test
            else:
                assert -999.0 not in array  # pylint: disable=unsupported-membership-test

        def test_max_len_none(self):
            tss = TimeSeriesSamples(
                [
                    TimeSeries(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [11.0, 12.0, 13.0]})),
                    TimeSeries(pd.DataFrame({"a": [-1.0, -2.0], "b": [-11.0, -12.0]})),
                    TimeSeries(pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0], "b": [111.0, 122.0, 133.0, 144.0]})),
                ]
            )

            array = tss.to_numpy(padding_indicator=-999.0, max_len=None)

            assert array.shape == (3, 4, 2)
            assert -999.0 in array  # pylint: disable=unsupported-membership-test

        @pytest.mark.parametrize("max_len, expect_padding", [(2, False), (3, True), (4, True), (5, True)])
        def test_time_index_max_len_set(self, max_len, expect_padding):
            tss = TimeSeriesSamples(
                [
                    TimeSeries(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [11.0, 12.0, 13.0]}, index=[1, 2, 3])),
                    TimeSeries(pd.DataFrame({"a": [-1.0, -2.0], "b": [-11.0, -12.0]}, index=[10, 11])),
                    TimeSeries(
                        pd.DataFrame(
                            {"a": [10.0, 20.0, 30.0, 40.0], "b": [111.0, 122.0, 133.0, 144.0]},
                            index=[11, 12, 13, 14],
                        ),
                    ),
                ]
            )

            array = tss.to_numpy_time_index(padding_indicator=777.0, max_len=max_len)

            assert array.shape == (3, max_len, 1)
            if expect_padding:
                assert 777.0 in array
            else:
                assert 777.0 not in array

        def test_time_index_max_len_none(self):
            tss = TimeSeriesSamples(
                [
                    TimeSeries(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [11.0, 12.0, 13.0]})),
                    TimeSeries(pd.DataFrame({"a": [-1.0, -2.0], "b": [-11.0, -12.0]})),
                    TimeSeries(pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0], "b": [111.0, 122.0, 133.0, 144.0]})),
                ]
            )

            array = tss.to_numpy_time_index(padding_indicator=777.0, max_len=None)

            assert array.shape == (3, 4, 1)
            assert 777.0 in array

    class TestIsRegularMethod:
        def test_true(self):
            ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}, index=[8, 10, 12]))
            ts_1 = TimeSeries(data=pd.DataFrame({"a": [-1, -2, -3], "b": [-1.0, -2.0, -3.0]}, index=[0, 2, 4]))
            tss = TimeSeriesSamples(data=[ts_0, ts_1])

            is_regular, diff = tss.is_regular()

            assert is_regular is True
            assert diff == 2

        def test_true_one_time_series_only(self):
            ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}, index=[8, 10, 12]))
            tss = TimeSeriesSamples(data=[ts_0])

            is_regular, diff = tss.is_regular()

            assert is_regular is True
            assert diff == 2

        def test_false(self):
            ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}, index=[8, 20, 21]))
            ts_1 = TimeSeries(data=pd.DataFrame({"a": [-1, -2, -3], "b": [-1.0, -2.0, -3.0]}, index=[8, 20, 25]))
            tss = TimeSeriesSamples(data=[ts_0, ts_1])

            is_regular, diff = tss.is_regular()

            assert is_regular is False
            assert diff is None

        def test_false_one_time_series_only(self):
            ts_0 = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}, index=[8, 20, 21]))
            tss = TimeSeriesSamples(data=[ts_0])

            is_regular, diff = tss.is_regular()

            assert is_regular is False
            assert diff is None

    def test_copy(self, three_numeric_dfs):
        tss = TimeSeriesSamples(
            [
                TimeSeries(three_numeric_dfs[0]),
                TimeSeries(three_numeric_dfs[1]),
                TimeSeries(three_numeric_dfs[2]),
            ],
        )
        tss_copy = tss.copy()
        tss_copy[0].df.loc[0, "a"] = 12345

        assert id(tss_copy) != id(tss)
        assert id(tss_copy.df) != id(tss.df)
        assert id(tss_copy[0]) != id(tss[0])
        assert id(tss_copy.df) != id(tss.df)
        assert tss[0].df.loc[0, "a"] == 1
        assert tss_copy[0].df.loc[0, "a"] == 12345

    def test_to_multi_index_dataframe(self, three_numeric_dfs):
        tss = TimeSeriesSamples(
            [
                TimeSeries(three_numeric_dfs[0]),
                TimeSeries(three_numeric_dfs[1]),
                TimeSeries(three_numeric_dfs[2]),
            ],
        )
        multi = tss.to_multi_index_dataframe()

        expected_df = pd.DataFrame(
            {"a": [1, 2, 3, 7, 8, 9, -1, -2, -3], "b": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0]},
            index=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        )

        assert (multi.index == expected_df.index).all()
        assert (multi.columns == expected_df.columns).all()
        assert (multi == expected_df).all().all()

    def test_sample_indices(self, three_numeric_dfs):
        tss = TimeSeriesSamples(
            [
                TimeSeries(three_numeric_dfs[0]),
                TimeSeries(three_numeric_dfs[1]),
                TimeSeries(three_numeric_dfs[2]),
            ],
        )

        sample_indices = tss.sample_indices

        assert sample_indices == [0, 1, 2]
