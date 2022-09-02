import pandas as pd
import pytest

from clairvoyance2.data import TimeSeries, TimeSeriesSamples
from clairvoyance2.data.utils import get_n_step_ahead_index, time_index_equal


class TestIntegration:
    class TestTimeIndexEqual:
        @pytest.mark.parametrize(
            "index_1, index_2",
            [
                ([0, 1, 2], [0, 1, 2]),
                (
                    pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                    pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                ),
            ],
        )
        def test_ts_case_true(self, index_1, index_2):
            ts1 = TimeSeries(pd.DataFrame({"a": [1, 2, 3]}, index=index_1))
            ts2 = TimeSeries(pd.DataFrame({"a": [-1, -2, -3]}, index=index_2))

            assert time_index_equal(ts1, ts2)

        @pytest.mark.parametrize(
            "index_1, index_2",
            [
                ([0, 1, 2], [0, 3, 8]),
                (
                    pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                    pd.to_datetime(["2000-01-01", "2000-02-02", "2000-03-03"]),
                ),
            ],
        )
        def test_ts_case_false(self, index_1, index_2):
            ts1 = TimeSeries(pd.DataFrame({"a": [1, 2, 3]}, index=index_1))
            ts2 = TimeSeries(pd.DataFrame({"a": [-1, -2, -3]}, index=index_2))

            assert not time_index_equal(ts1, ts2)

        @pytest.mark.parametrize(
            "index_1_1, index_1_2, index_2_1, index_2_2",
            [
                ([0, 1, 2], [7, 11], [0, 1, 2], [7, 11]),
                (
                    pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                    pd.to_datetime(["2000-12-07", "2000-12-11"]),
                    pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                    pd.to_datetime(["2000-12-07", "2000-12-11"]),
                ),
            ],
        )
        def test_tss_case_true(self, index_1_1, index_1_2, index_2_1, index_2_2):
            ts1 = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3]}, index=index_1_1),
                    pd.DataFrame({"a": [1, 2]}, index=index_1_2),
                ]
            )
            ts2 = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3]}, index=index_2_1),
                    pd.DataFrame({"a": [1, 2]}, index=index_2_2),
                ]
            )

            assert time_index_equal(ts1, ts2)

        @pytest.mark.parametrize(
            "index_1_1, index_1_2, index_2_1, index_2_2",
            [
                ([0, 1, 2], [7, 11], [0, 1, 2], [7, 12]),
                (
                    pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                    pd.to_datetime(["2000-12-07", "2000-12-11"]),
                    pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                    pd.to_datetime(["2000-12-07", "2000-12-12"]),
                ),
            ],
        )
        def test_tss_case_false(self, index_1_1, index_1_2, index_2_1, index_2_2):
            ts1 = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3]}, index=index_1_1),
                    pd.DataFrame({"a": [1, 2]}, index=index_1_2),
                ]
            )
            ts2 = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3]}, index=index_2_1),
                    pd.DataFrame({"a": [1, 2]}, index=index_2_2),
                ]
            )

            assert not time_index_equal(ts1, ts2)

        def test_tss_case_false_samples_do_not_match(self):
            index_1_1 = [0, 1, 2]
            index_1_2 = [7, 11]
            index_2_1 = index_1_1
            ts1 = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3]}, index=index_1_1),
                    pd.DataFrame({"a": [1, 2]}, index=index_1_2),
                ]
            )
            ts2 = TimeSeriesSamples(
                [
                    pd.DataFrame({"a": [1, 2, 3]}, index=index_2_1),
                ]
            )

            assert not time_index_equal(ts1, ts2)

        def test_raises_type_exception(self):
            with pytest.raises(TypeError):
                time_index_equal(123, "string")


class TestGetNStepAheadIndex:
    @pytest.mark.parametrize(
        "original_index, n_step, expected_new_index",
        [
            # --- int index ---
            (
                # original_index:
                pd.Index([1, 2, 3]),
                # n_step:
                1,
                # expected_new_index:
                pd.Index([2, 3, 4]),
            ),
            (
                # original_index:
                pd.Index([4, 8]),
                # n_step:
                1,
                # expected_new_index:
                pd.Index([8, 12]),
            ),
            (
                # original_index:
                pd.Index([3, 6]),
                # n_step:
                3,
                # expected_new_index:
                pd.Index([12, 15]),
            ),
            # --- float index ---
            (
                # original_index:
                pd.Index([1.1, 2.1]),
                # n_step:
                2,
                # expected_new_index:
                pd.Index([3.1, 4.1]),
            ),
            # --- datetime index ---
            (
                # original_index:
                pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
                # n_step:
                2,
                # expected_new_index:
                pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]),
            ),
        ],
    )
    def test_int_index(self, original_index, n_step, expected_new_index):
        new_idx = get_n_step_ahead_index(original_index, n_step=n_step)
        assert len(new_idx) == len(original_index)
        assert len(new_idx) == len(expected_new_index)
        assert (new_idx == expected_new_index).all()

    def test_fails_not_regular(self):
        with pytest.raises(RuntimeError) as excinfo:
            get_n_step_ahead_index(pd.Index([1, 2, 11]), n_step=1)
        assert "regular" in str(excinfo.value)

    def test_fails_too_short(self):
        with pytest.raises(RuntimeError) as excinfo:
            get_n_step_ahead_index(pd.Index([7]), n_step=1)
        assert "two elements" in str(excinfo.value)
