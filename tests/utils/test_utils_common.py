import pandas as pd
import pytest

from clairvoyance2.utils.common import rolling_window, split_multi_index_dataframe


class TestRollingWindow:
    class TestExpandNeither:
        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40, 50, 60], 3, ([10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60])),
                # Case: window = len(sequence) - 1
                ([10, 20, 30, 40, 50, 60], 5, ([10, 20, 30, 40, 50], [20, 30, 40, 50, 60])),
                # Case: =1 window.
                (
                    [10, 20, 30, 40, 50, 60],
                    1,
                    (
                        [10],
                        [20],
                        [30],
                        [40],
                        [50],
                        [60],
                    ),
                ),
            ],
        )
        def test_window_lt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="neither") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40, 50, 60], 6, ([10, 20, 30, 40, 50, 60],)),
                # Case: =1 window.
                ([10], 1, ([10],)),
            ],
        )
        def test_window_eq_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="neither") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40, 50, 60], 10, tuple()),
                # Case: len(sequence) == 1.
                (
                    [10],
                    2,
                    tuple(),
                ),
            ],
        )
        def test_window_gt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="neither") == expected_sequences_out

    class TestExpandBoth:
        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40, 50], 3, ([10], [10, 20], [10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50], [50])),
                # Case: window = len(sequence) - 1
                ([10, 20, 30, 40], 3, ([10], [10, 20], [10, 20, 30], [20, 30, 40], [30, 40], [40])),
                # Case: =1 window.
                (
                    [10, 20, 30, 40],
                    1,
                    (
                        [10],
                        [20],
                        [30],
                        [40],
                    ),
                ),
            ],
        )
        def test_window_lt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="both") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40], 4, ([10], [10, 20], [10, 20, 30], [10, 20, 30, 40], [20, 30, 40], [30, 40], [40])),
                # Case: =1 window.
                ([10], 1, ([10],)),
            ],
        )
        def test_window_eq_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="both") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40], 10, ([10], [10, 20], [10, 20, 30], [10, 20, 30, 40], [20, 30, 40], [30, 40], [40])),
                # Case: len(sequence) == 1.
                ([10], 2, ([10],)),
            ],
        )
        def test_window_gt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="both") == expected_sequences_out

    class TestExpandLeft:
        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40, 50], 3, ([10], [10, 20], [10, 20, 30], [20, 30, 40], [30, 40, 50])),
                # Case: window = len(sequence) - 1
                ([10, 20, 30, 40], 3, ([10], [10, 20], [10, 20, 30], [20, 30, 40])),
                # Case: =1 window.
                (
                    [10, 20, 30, 40],
                    1,
                    (
                        [10],
                        [20],
                        [30],
                        [40],
                    ),
                ),
            ],
        )
        def test_window_lt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="left") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40], 4, ([10], [10, 20], [10, 20, 30], [10, 20, 30, 40])),
                # Case: =1 window.
                ([10], 1, ([10],)),
            ],
        )
        def test_window_eq_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="left") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40], 10, ([10], [10, 20], [10, 20, 30], [10, 20, 30, 40])),
                # Case: len(sequence) == 1.
                ([10], 2, ([10],)),
            ],
        )
        def test_window_gt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="left") == expected_sequences_out

    class TestExpandRight:
        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40, 50], 3, ([10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50], [50])),
                # Case: window = len(sequence) - 1
                ([10, 20, 30, 40], 3, ([10, 20, 30], [20, 30, 40], [30, 40], [40])),
                # Case: =1 window.
                (
                    [10, 20, 30, 40],
                    1,
                    (
                        [10],
                        [20],
                        [30],
                        [40],
                    ),
                ),
            ],
        )
        def test_window_lt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="right") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40], 4, ([10, 20, 30, 40], [20, 30, 40], [30, 40], [40])),
                # Case: =1 window.
                ([10], 1, ([10],)),
            ],
        )
        def test_window_eq_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="right") == expected_sequences_out

        @pytest.mark.parametrize(
            "sequence_in, window, expected_sequences_out",
            [
                # Case: >1 window.
                ([10, 20, 30, 40], 10, ([10, 20, 30, 40], [20, 30, 40], [30, 40], [40])),
                # Case: len(sequence) == 1.
                ([10], 2, ([10],)),
            ],
        )
        def test_window_gt_sequence(self, sequence_in, window, expected_sequences_out):
            assert rolling_window(sequence_in, window=window, expand="right") == expected_sequences_out

    @pytest.mark.parametrize("window", [0, -3])
    def test_raises_value_error_wrong_window(self, window):
        with pytest.raises(ValueError) as excinfo:
            _ = rolling_window([1, 2, 3], window=window)
        assert "window" in str(excinfo.value)


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
