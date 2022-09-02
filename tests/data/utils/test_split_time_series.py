import pandas as pd
import pytest

from clairvoyance2.data import Dataset, StaticSamples, TimeSeries, TimeSeriesSamples
from clairvoyance2.data.utils import split_time_series


class TestIntegration:
    class TestSplit:
        @pytest.mark.parametrize(
            "in_ts_df, at_iloc, out_ts0_df, out_ts1_df",
            [
                # Case: split at iloc 1.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    1,
                    pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                    pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                ),
                # Case: split at highest iloc.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    3,
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                    pd.DataFrame({"a": [4], "b": [4.4]}, index=[3]),
                ),
                # Case: split somewhere in between.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    2,
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                ),
                # Check also datetime index.
                (
                    pd.DataFrame(
                        {"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]},
                        index=pd.to_datetime(["2000-01-01", "2000-01-05", "2000-01-08", "2000-01-14"]),
                    ),
                    2,
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=pd.to_datetime(["2000-01-01", "2000-01-05"])),
                    pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=pd.to_datetime(["2000-01-08", "2000-01-14"])),
                ),
                # Test shortest possible time series.
                (
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    1,
                    pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                    pd.DataFrame({"a": [2], "b": [2.2]}, index=[1]),
                ),
            ],
        )
        def test_success(self, in_ts_df, at_iloc, out_ts0_df, out_ts1_df):
            ts = TimeSeries(data=in_ts_df)

            ts0, ts1 = split_time_series.split(ts, at_iloc=at_iloc)

            assert ts0.n_timesteps == out_ts0_df.shape[0]
            assert ts1.n_timesteps == out_ts1_df.shape[0]
            assert (ts0.df == out_ts0_df).all().all()
            assert (ts1.df == out_ts1_df).all().all()

        @pytest.mark.parametrize(
            "in_ts_df, at_iloc, out_ts0_df, out_ts1_df",
            [
                # Case: split at iloc 1.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    1,
                    pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                ),
                # Case: split at highest iloc.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    3,
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                    pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                ),
                # Case: split somewhere in between.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    2,
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                ),
                # Check also datetime index.
                (
                    pd.DataFrame(
                        {"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]},
                        index=pd.to_datetime(["2000-01-01", "2000-01-05", "2000-01-08", "2000-01-14"]),
                    ),
                    2,
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=pd.to_datetime(["2000-01-01", "2000-01-05"])),
                    pd.DataFrame(
                        {"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]},
                        index=pd.to_datetime(["2000-01-05", "2000-01-08", "2000-01-14"]),
                    ),
                ),
                # Test shortest possible time series.
                (
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    1,
                    pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                ),
            ],
        )
        def test_success_repeat_last(self, in_ts_df, at_iloc, out_ts0_df, out_ts1_df):
            ts = TimeSeries(data=in_ts_df)

            ts0, ts1 = split_time_series.split(ts, at_iloc=at_iloc, repeat_last_pre_step=True)

            assert ts0.n_timesteps == out_ts0_df.shape[0]
            assert ts1.n_timesteps == out_ts1_df.shape[0]
            assert (ts0.df == out_ts0_df).all().all()
            assert (ts1.df == out_ts1_df).all().all()

        def test_repeat_last_success_with_only_1_timestep(self):
            df = pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])
            ts = TimeSeries(data=df)

            ts0, ts1 = split_time_series.split(ts, at_iloc=1, repeat_last_pre_step=True)

            out_ts0_df = pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])
            out_ts1_df = pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])
            assert ts0.n_timesteps == out_ts0_df.shape[0]
            assert ts1.n_timesteps == out_ts1_df.shape[0]
            assert (ts0.df == out_ts0_df).all().all()
            assert (ts1.df == out_ts1_df).all().all()

        def test_fail_too_few_timesteps(self):
            df = pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])
            ts = TimeSeries(data=df)

            with pytest.raises(ValueError) as excinfo:
                _ = split_time_series.split(ts, at_iloc=1)
            assert "time steps < 2" in str(excinfo.value)

        def test_fail_too_few_timesteps_repeat_last(self):
            df = pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])
            df_no_timesteps = df.loc[:-1, :]
            ts = TimeSeries(data=df_no_timesteps)

            with pytest.raises(ValueError) as excinfo:
                _ = split_time_series.split(ts, at_iloc=1, repeat_last_pre_step=True)
            assert "time steps < 1" in str(excinfo.value)

        @pytest.mark.parametrize("at_iloc", [0, -1, 4, 5, 10])
        def test_fail_invalid_iloc(self, at_iloc):
            df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3])
            ts = TimeSeries(data=df)

            with pytest.raises(ValueError) as excinfo:
                _ = split_time_series.split(ts, at_iloc=at_iloc)
            assert "`at_iloc` to be in range" in str(excinfo.value)

    class TestSplitAtEachStep:
        @pytest.mark.parametrize(
            "in_ts_df, ts_pre_expected, ts_post_expected, count_expected",
            [
                # Case: typical case.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                        pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                        pd.DataFrame({"a": [4], "b": [4.4]}, index=[3]),
                    ],
                    # count_expected:
                    3,
                ),
                # Case: shortest possible.
                (
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [2], "b": [2.2]}, index=[1]),
                    ],
                    # count_expected:
                    1,
                ),
            ],
        )
        def test_success(self, in_ts_df, ts_pre_expected, ts_post_expected, count_expected):
            ts = TimeSeries(data=in_ts_df)

            ts_pre, ts_post, count = split_time_series.split_at_each_step(ts)

            assert isinstance(ts_pre, tuple)
            assert isinstance(ts_post, tuple)
            assert len(ts_pre) == len(ts_post) == count == count_expected == ts.n_timesteps - 1
            for idx in range(len(ts_pre)):
                assert (ts_pre[idx].df == ts_pre_expected[idx]).all().all()
                assert (ts_post[idx].df == ts_post_expected[idx]).all().all()

        @pytest.mark.parametrize(
            "in_ts_df, ts_pre_expected, ts_post_expected, count_expected",
            [
                # Case: typical case.
                (
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                        pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                        pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                        pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                        pd.DataFrame({"a": [4], "b": [4.4]}, index=[3]),
                    ],
                    # count_expected:
                    4,
                ),
                # Case: shortest possible.
                (
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                        pd.DataFrame({"a": [2], "b": [2.2]}, index=[1]),
                    ],
                    # count_expected:
                    2,
                ),
            ],
        )
        def test_success_repeat_last(self, in_ts_df, ts_pre_expected, ts_post_expected, count_expected):
            ts = TimeSeries(data=in_ts_df)

            ts_pre, ts_post, count = split_time_series.split_at_each_step(ts, repeat_last_pre_step=True)

            assert isinstance(ts_pre, tuple)
            assert isinstance(ts_post, tuple)
            assert len(ts_pre) == len(ts_post) == count == count_expected == ts.n_timesteps
            for idx in range(len(ts_pre)):
                assert (ts_pre[idx].df == ts_pre_expected[idx]).all().all()
                assert (ts_post[idx].df == ts_post_expected[idx]).all().all()

        @pytest.mark.parametrize(
            "in_ts_df, min_pre_len, min_post_len, ts_pre_expected, ts_post_expected",
            [
                # Case: set min_pre_len.
                (
                    # in_ts_df:
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # min_pre_len:
                    2,
                    # min_post_len:
                    1,
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                        pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                        pd.DataFrame({"a": [4], "b": [4.4]}, index=[3]),
                    ],
                ),
                # Case: set min_post_len.
                (
                    # in_ts_df:
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # min_pre_len:
                    1,
                    # min_post_len:
                    2,
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                    ],
                ),
                # Case: set both.
                (
                    # in_ts_df:
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # min_pre_len:
                    2,
                    # min_post_len:
                    2,
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                    ],
                ),
            ],
        )
        def test_success_set_min_lens(self, in_ts_df, min_pre_len, min_post_len, ts_pre_expected, ts_post_expected):
            ts = TimeSeries(data=in_ts_df)

            ts_pre, ts_post, count = split_time_series.split_at_each_step(ts, min_pre_len, min_post_len)

            assert isinstance(ts_pre, tuple)
            assert isinstance(ts_post, tuple)
            assert len(ts_pre) == len(ts_post) == count
            for idx in range(len(ts_pre)):
                assert (ts_pre[idx].df == ts_pre_expected[idx]).all().all()
                assert (ts_post[idx].df == ts_post_expected[idx]).all().all()

        def test_repeat_last_success_with_only_1_timestep(self):
            df = pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])
            ts = TimeSeries(data=df)

            ts_pre, ts_post, count = split_time_series.split_at_each_step(
                ts, min_pre_len=1, min_post_len=1, repeat_last_pre_step=True
            )
            print(ts_pre)
            print(ts_post)

            ts_pre_expected = [pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])]
            ts_post_expected = [pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])]
            assert isinstance(ts_pre, tuple)
            assert isinstance(ts_post, tuple)
            assert len(ts_pre) == len(ts_post) == count
            for idx in range(len(ts_pre)):
                assert (ts_pre[idx].df == ts_pre_expected[idx]).all().all()
                assert (ts_post[idx].df == ts_post_expected[idx]).all().all()

        @pytest.mark.parametrize(
            "in_ts_df, min_pre_len, min_post_len, ts_pre_expected, ts_post_expected",
            [
                # Case: set min_pre_len.
                (
                    # in_ts_df:
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # min_pre_len:
                    2,
                    # min_post_len:
                    1,
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                        pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                        pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                        pd.DataFrame({"a": [4], "b": [4.4]}, index=[3]),
                    ],
                ),
                # Case: set min_post_len.
                (
                    # in_ts_df:
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # min_pre_len:
                    1,
                    # min_post_len:
                    2,
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1], "b": [1.1]}, index=[0]),
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                        pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                        pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                    ],
                ),
                # Case: set both.
                (
                    # in_ts_df:
                    pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.1, 2.2, 3.3, 4.4]}, index=[0, 1, 2, 3]),
                    # min_pre_len:
                    2,
                    # min_post_len:
                    2,
                    # ts_pre_expected:
                    [
                        pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}, index=[0, 1]),
                        pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}, index=[0, 1, 2]),
                    ],
                    # ts_post_expected:
                    [
                        pd.DataFrame({"a": [2, 3, 4], "b": [2.2, 3.3, 4.4]}, index=[1, 2, 3]),
                        pd.DataFrame({"a": [3, 4], "b": [3.3, 4.4]}, index=[2, 3]),
                    ],
                ),
            ],
        )
        def test_success_set_min_lens_repeat_last(
            self, in_ts_df, min_pre_len, min_post_len, ts_pre_expected, ts_post_expected
        ):
            ts = TimeSeries(data=in_ts_df)

            ts_pre, ts_post, count = split_time_series.split_at_each_step(
                ts, min_pre_len, min_post_len, repeat_last_pre_step=True
            )

            assert isinstance(ts_pre, tuple)
            assert isinstance(ts_post, tuple)
            assert len(ts_pre) == len(ts_post) == count
            for idx in range(len(ts_pre)):
                assert (ts_pre[idx].df == ts_pre_expected[idx]).all().all()
                assert (ts_post[idx].df == ts_post_expected[idx]).all().all()

        @pytest.mark.parametrize(
            "df, min_pre_len, min_post_len",
            [
                (
                    pd.DataFrame({"a": [1], "b": [1.1]}),
                    1,
                    1,
                ),
                (
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}),
                    2,
                    1,
                ),
                (
                    pd.DataFrame({"a": [1, 2], "b": [1.1, 2.2]}),
                    1,
                    2,
                ),
                (
                    pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}),
                    2,
                    2,
                ),
            ],
        )
        def test_fail_too_few_timesteps(self, df, min_pre_len, min_post_len):
            ts = TimeSeries(data=df)

            with pytest.raises(ValueError) as excinfo:
                (*_,) = split_time_series.split_at_each_step(ts, min_pre_len, min_post_len)
            assert f"time steps < {min_pre_len + min_post_len}" in str(excinfo.value)

        def test_fail_too_few_timesteps_repeat_last(self):
            df = pd.DataFrame({"a": [1], "b": [1.1]}, index=[0])
            df_no_timesteps = df.loc[:-1, :]
            ts = TimeSeries(data=df_no_timesteps)

            with pytest.raises(ValueError) as excinfo:
                _ = split_time_series.split_at_each_step(ts, min_pre_len=1, min_post_len=1, repeat_last_pre_step=True)
            assert "time steps < 1" in str(excinfo.value)

    class TestSplitAtEachStepTSS:
        def test_success_check_samples_count_map(self):
            dfs = [
                pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 40]}),
                pd.DataFrame({"a": [7, 8], "b": [7.0, 8.0]}),
                pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]}),
            ]
            tss = TimeSeriesSamples(data=dfs)

            tss0, tss1, samples_map = split_time_series.split_at_each_step(tss)

            assert isinstance(tss0, TimeSeriesSamples)
            assert isinstance(tss1, TimeSeriesSamples)
            assert tss0.feature_names == tss.feature_names
            assert tss1.feature_names == tss.feature_names
            assert tss0.categorical_def == tss.categorical_def  # pylint: disable=no-member
            assert tss1.categorical_def == tss.categorical_def  # pylint: disable=no-member
            assert tss0.n_samples == 6
            assert tss1.n_samples == 6
            assert samples_map == {0: [0, 1, 2], 1: [3], 2: [4, 5]}

        def test_success_repeat_last_check_samples_count_map(self):
            dfs = [
                pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 40]}),
                pd.DataFrame({"a": [7, 8], "b": [7.0, 8.0]}),
                pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0]}),
            ]
            tss = TimeSeriesSamples(data=dfs)

            tss0, tss1, samples_map = split_time_series.split_at_each_step(tss, repeat_last_pre_step=True)

            assert isinstance(tss0, TimeSeriesSamples)
            assert isinstance(tss1, TimeSeriesSamples)
            assert tss0.feature_names == tss.feature_names
            assert tss1.feature_names == tss.feature_names
            assert tss0.categorical_def == tss.categorical_def  # pylint: disable=no-member
            assert tss1.categorical_def == tss.categorical_def  # pylint: disable=no-member
            assert tss0.n_samples == 9
            assert tss1.n_samples == 9
            assert samples_map == {0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7, 8]}

    class TestSplitAtEachStepDataset:
        def test_success_no_static_samples(self):
            dfs_cov = [
                pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 4.0], "c": [1.1, 2.1, 3.1, 4.1]}),
                pd.DataFrame({"a": [7, 8], "b": [7.0, 8.0], "c": [-7.0, -8.0]}),
                pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [11.0, 12.0, 13.0]}),
            ]
            dfs_targ = [
                pd.DataFrame({"d": [1, 2, 3, 4], "e": [1.0, 2.0, 3.0, 40]}),
                pd.DataFrame({"d": [7, 8], "e": [7.0, 8.0]}),
                pd.DataFrame({"d": [-1, -2, -3], "e": [11.0, 12.0, 13.0]}),
            ]
            dfs_treat = [
                pd.DataFrame({"f": [1, 2, 3, 4]}),
                pd.DataFrame({"f": [7, 8]}),
                pd.DataFrame({"f": [-1, -2, -3]}),
            ]
            t_cov = TimeSeriesSamples(data=dfs_cov)
            t_targ = TimeSeriesSamples(data=dfs_targ)
            t_treat = TimeSeriesSamples(data=dfs_treat)
            data = Dataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat)

            data_0, data_1, samples_map = split_time_series.split_at_each_step(data)

            assert isinstance(data_0, Dataset)
            assert isinstance(data_1, Dataset)
            assert data_0.temporal_covariates.n_features == 3
            assert data_0.temporal_targets.n_features == 2
            assert data_0.temporal_treatments.n_features == 1
            assert data_0.temporal_covariates.n_samples == 6
            assert data_1.temporal_covariates.n_samples == 6
            assert data_0.temporal_targets.n_samples == 6
            assert data_1.temporal_targets.n_samples == 6
            assert data_0.temporal_treatments.n_samples == 6
            assert data_1.temporal_treatments.n_samples == 6
            assert data_0.static_covariates is None
            assert data_1.static_covariates is None
            assert samples_map == {0: [0, 1, 2], 1: [3], 2: [4, 5]}

        def test_success_repeat_last_no_static_samples(self):
            dfs_cov = [
                pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 4.0], "c": [1.1, 2.1, 3.1, 4.1]}),
                pd.DataFrame({"a": [7, 8], "b": [7.0, 8.0], "c": [-7.0, -8.0]}),
                pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [11.0, 12.0, 13.0]}),
            ]
            dfs_targ = [
                pd.DataFrame({"d": [1, 2, 3, 4], "e": [1.0, 2.0, 3.0, 40]}),
                pd.DataFrame({"d": [7, 8], "e": [7.0, 8.0]}),
                pd.DataFrame({"d": [-1, -2, -3], "e": [11.0, 12.0, 13.0]}),
            ]
            dfs_treat = [
                pd.DataFrame({"f": [1, 2, 3, 4]}),
                pd.DataFrame({"f": [7, 8]}),
                pd.DataFrame({"f": [-1, -2, -3]}),
            ]
            t_cov = TimeSeriesSamples(data=dfs_cov)
            t_targ = TimeSeriesSamples(data=dfs_targ)
            t_treat = TimeSeriesSamples(data=dfs_treat)
            data = Dataset(temporal_covariates=t_cov, temporal_targets=t_targ, temporal_treatments=t_treat)

            data_0, data_1, samples_map = split_time_series.split_at_each_step(data, repeat_last_pre_step=True)

            assert isinstance(data_0, Dataset)
            assert isinstance(data_1, Dataset)
            assert data_0.temporal_covariates.n_features == 3
            assert data_0.temporal_targets.n_features == 2
            assert data_0.temporal_treatments.n_features == 1
            assert data_0.temporal_covariates.n_samples == 9
            assert data_1.temporal_covariates.n_samples == 9
            assert data_0.temporal_targets.n_samples == 9
            assert data_1.temporal_targets.n_samples == 9
            assert data_0.temporal_treatments.n_samples == 9
            assert data_1.temporal_treatments.n_samples == 9
            assert data_0.static_covariates is None
            assert data_1.static_covariates is None
            assert samples_map == {0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7, 8]}

        def test_success_with_static_samples(self):
            dfs_cov = [
                pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 4.0], "c": [1.1, 2.1, 3.1, 4.1]}),
                pd.DataFrame({"a": [7, 8], "b": [7.0, 8.0], "c": [-7.0, -8.0]}),
                pd.DataFrame({"a": [-1, -2, -3], "b": [11.0, 12.0, 13.0], "c": [11.0, 12.0, 13.0]}),
            ]
            dfs_targ = [
                pd.DataFrame({"d": [1, 2, 3, 4], "e": [1.0, 2.0, 3.0, 40]}),
                pd.DataFrame({"d": [7, 8], "e": [7.0, 8.0]}),
                pd.DataFrame({"d": [-1, -2, -3], "e": [11.0, 12.0, 13.0]}),
            ]
            dfs_treat = [
                pd.DataFrame({"f": [1, 2, 3, 4]}),
                pd.DataFrame({"f": [7, 8]}),
                pd.DataFrame({"f": [-1, -2, -3]}),
            ]
            t_cov = TimeSeriesSamples(data=dfs_cov)
            t_targ = TimeSeriesSamples(data=dfs_targ)
            t_treat = TimeSeriesSamples(data=dfs_treat)
            df_s_cov = pd.DataFrame({"s_a": [0.4, 0.2, 0.2], "s_b": [4, 2, 3]})
            s_cov = StaticSamples(df_s_cov)
            data = Dataset(
                temporal_covariates=t_cov, static_covariates=s_cov, temporal_targets=t_targ, temporal_treatments=t_treat
            )

            data_0, data_1, samples_map = split_time_series.split_at_each_step(data)

            assert isinstance(data_0, Dataset)
            assert isinstance(data_1, Dataset)
            assert data_0.temporal_covariates.n_features == 3
            assert data_0.temporal_targets.n_features == 2
            assert data_0.temporal_treatments.n_features == 1
            assert data_0.temporal_covariates.n_samples == 6
            assert data_1.temporal_covariates.n_samples == 6
            assert data_0.temporal_targets.n_samples == 6
            assert data_1.temporal_targets.n_samples == 6
            assert data_0.temporal_treatments.n_samples == 6
            assert data_1.temporal_treatments.n_samples == 6
            assert samples_map == {0: [0, 1, 2], 1: [3], 2: [4, 5]}
            # ---
            assert data_0.static_covariates is not None
            assert data_1.static_covariates is not None
            assert data_0.static_covariates.n_samples == 6
            assert data_1.static_covariates.n_samples == 6
            assert list(data_0.static_covariates.df.columns) == ["s_a", "s_b"]
            assert list(data_1.static_covariates.df.columns) == ["s_a", "s_b"]
            assert (data_0.static_covariates.df[0:3].values == pd.DataFrame({"s_a": [0.4], "s_b": [4]}).values).all()
            assert (data_0.static_covariates.df[3:4].values == pd.DataFrame({"s_a": [0.2], "s_b": [2]}).values).all()
            assert (data_0.static_covariates.df[4:].values == pd.DataFrame({"s_a": [0.2], "s_b": [3]}).values).all()
            assert (data_1.static_covariates.df[0:3].values == pd.DataFrame({"s_a": [0.4], "s_b": [4]}).values).all()
            assert (data_1.static_covariates.df[3:4].values == pd.DataFrame({"s_a": [0.2], "s_b": [2]}).values).all()
            assert (data_1.static_covariates.df[4:].values == pd.DataFrame({"s_a": [0.2], "s_b": [3]}).values).all()
