import numpy as np
import pandas as pd
import pytest

from clairvoyance2.data import TimeSeries, TimeSeriesSamples


class TestIntegration:
    class TestUpdateTSFromArray:
        @pytest.mark.parametrize(
            "time_index, update_array, padding_indicator, expected_ts_df",
            [
                (
                    # time_index:
                    pd.Index([1, 3, 7]),
                    # update_array:
                    np.asarray([[10, 30, 70], [-10, -30, -70]]).T,
                    # padding_indicator:
                    999,
                    # expected_ts_df:
                    pd.DataFrame({"a": [10, 2, 30, 70], "b": [-10, -2, -30, -70]}, index=[1, 2, 3, 7]),
                ),
                (
                    # time_index:
                    pd.Index([1, 3, 7]),
                    # update_array:
                    np.asarray([[10, 999, 70], [-10, 999, -70]]).T,
                    # padding_indicator:
                    999,
                    # expected_ts_df:
                    pd.DataFrame({"a": [10, 2, 3, 70], "b": [-10, -2, -3, -70]}, index=[1, 2, 3, 7]),
                ),
                (
                    # time_index:
                    pd.Index([1, 3, 7]),
                    # update_array:
                    np.asarray([[10, 999, 999], [-10, 999, 999]]).T,
                    # padding_indicator:
                    999,
                    # expected_ts_df:
                    pd.DataFrame({"a": [10, 2, 3], "b": [-10, -2, -3]}, index=[1, 2, 3]),
                ),
                (
                    # time_index:
                    pd.Index([1, 3, 7]),
                    # update_array:
                    np.asarray([[10, np.nan, np.nan], [-10, np.nan, np.nan]]).T,
                    # padding_indicator:
                    np.nan,
                    # expected_ts_df:
                    pd.DataFrame({"a": [10, 2, 3], "b": [-10, -2, -3]}, index=[1, 2, 3]),
                ),
                (
                    # time_index:
                    pd.Index([1, 3, 7]),
                    # update_array:
                    np.asarray([[999, 999, 999], [999, 999, 999]]).T,
                    # padding_indicator:
                    999,
                    # expected_ts_df:
                    pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]),
                ),
            ],
        )
        def test_success(self, time_index, update_array, padding_indicator, expected_ts_df):
            ts = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]))

            ts.update_from_array(update_array=update_array, time_index=time_index, padding_indicator=padding_indicator)

            assert (ts.df == expected_ts_df).all().all()

    class TestUpdateTSNStep:
        def test_fail_wrong_dims(self):
            ts = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]))
            update_array = np.ones(shape=(4, 3, 2))
            with pytest.raises(ValueError) as excinfo:
                ts.update_from_array_n_step_ahead(update_array, n_step=1, padding_indicator=999.0)
            assert "dimensions" in str(excinfo.value)

        def test_fail_too_short(self):
            ts = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]))
            update_array = np.asarray([[88], [11]]).T  # (1, 2)
            with pytest.raises(ValueError) as excinfo:
                ts.update_from_array_n_step_ahead(update_array, n_step=1, padding_indicator=999.0)
            assert "at least" in str(excinfo.value)

        def test_fail_padding_found(self):
            pi = 999.0
            ts = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]))
            update_array = np.asarray([[88, pi, 87], [11, 12, pi]]).T  # (3, 2)
            with pytest.raises(ValueError) as excinfo:
                ts.update_from_array_n_step_ahead(update_array, n_step=1, padding_indicator=pi)
            assert "padding" in str(excinfo.value)

        @pytest.mark.parametrize(
            "n_step, update_array, expected_df",
            [
                (
                    1,
                    np.asarray([[88, 89, 87], [11, 12, 14]]).T,
                    pd.DataFrame({"a": [88, 89, 87], "b": [11, 12, 14]}, index=[2, 3, 4]),
                ),
                (
                    3,
                    np.asarray([[88, 89, 87], [11, 12, 14]]).T,
                    pd.DataFrame({"a": [88, 89, 87], "b": [11, 12, 14]}, index=[4, 5, 6]),
                ),
            ],
        )
        def test_success(self, n_step, update_array, expected_df):
            ts = TimeSeries(data=pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]))
            ts.update_from_array_n_step_ahead(update_array, n_step=n_step, padding_indicator=999.0)

            assert (ts.df == expected_df).all().all()

    class TestUpdateTSSFromSequenceOfArrays:
        def test_success(self):
            pi = 999
            tss = TimeSeriesSamples(
                data=[
                    pd.DataFrame({"a": [1, 2, 3], "b": [-1, -2, -3]}, index=[1, 2, 3]),
                    pd.DataFrame({"a": [1.1, 2.2], "b": [-1.1, -2.2]}, index=[11, 22]),
                ]
            )
            time_index_sequence = [pd.Index([1, 3, 4]), pd.Index([11, 22, 33])]
            update_arrays = np.asarray(
                [
                    np.asarray([[pi, 30, 40], [pi, -30, -40]]).T,
                    np.asarray([[pi, 22.2, pi], [pi, -22.2, -pi]]).T,
                ]
            )

            tss.update_from_sequence_of_arrays(
                update_array_sequence=update_arrays, time_index_sequence=time_index_sequence, padding_indicator=pi
            )

            assert (
                (tss[0].df == pd.DataFrame({"a": [1, 2, 30, 40], "b": [-1, -2, -30, -40]}, index=[1, 2, 3, 4]))
                .all()
                .all()
            )
            assert (tss[1].df == pd.DataFrame({"a": [1.1, 22.2], "b": [-1.1, -22.2]}, index=[11, 22])).all().all()
