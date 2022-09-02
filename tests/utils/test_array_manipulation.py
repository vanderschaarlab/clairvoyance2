from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

from clairvoyance2.utils.array_manipulation import compute_deltas, n_step_shifted


def are_equal(actual, expected) -> bool:
    return (
        (actual if isinstance(actual, np.ndarray) else actual.numpy())
        == (expected if isinstance(expected, np.ndarray) else expected.numpy())
    ).all()


class TestIntegration:
    class TestNStepShifted:
        class TestValidationFails:
            @pytest.mark.parametrize(
                "shift_back, shift_forward",
                [
                    (np.ones(shape=(2,)), np.ones(shape=(2, 2, 2))),
                    (np.ones(shape=(2,)), torch.ones(size=(2, 2, 2))),
                    (torch.ones(size=(2,)), np.ones(shape=(2, 2, 2))),
                    # ---
                    (np.ones(shape=(2, 2)), np.ones(shape=(2, 2, 2))),
                    (np.ones(shape=(2, 2)), torch.ones(size=(2, 2, 2))),
                    (torch.ones(size=(2, 2)), np.ones(shape=(2, 2, 2))),
                    # ---
                    (np.ones(shape=(2, 2, 2)), np.ones(shape=(2,))),
                    (np.ones(shape=(2, 2, 2)), torch.ones(size=(2,))),
                    (torch.ones(size=(2, 2, 2)), np.ones(shape=(2,))),
                ],
            )
            def test_wrong_ndim(self, shift_back, shift_forward):
                with pytest.raises(ValueError) as excinfo:
                    n_step_shifted(shift_back, shift_forward, n_step=1)
                assert "3 dimensions" in str(excinfo.value)

            @pytest.mark.parametrize(
                "shift_back, shift_forward",
                [
                    (np.ones(shape=(1, 3, 2)), np.ones(shape=(1, 4, 2))),
                    (np.ones(shape=(1, 3, 2)), torch.ones(size=(1, 4, 2))),
                    (torch.ones(size=(1, 3, 2)), np.ones(shape=(1, 4, 2))),
                ],
            )
            def test_mismatch_timestep_dim(self, shift_back, shift_forward):
                with pytest.raises(ValueError) as excinfo:
                    n_step_shifted(shift_back, shift_forward, n_step=1)
                assert "must have equal size of dimension 1" in str(excinfo.value)

            @pytest.mark.parametrize(
                "shift_back, shift_forward",
                [
                    (np.ones(shape=(1, 3, 2)), np.ones(shape=(1, 4, 2))),
                    (np.ones(shape=(1, 4, 2)), torch.ones(size=(1, 3, 2))),
                ],
            )
            def test_too_short(self, shift_back, shift_forward):
                with pytest.raises(ValueError) as excinfo:
                    n_step_shifted(shift_back, shift_forward, n_step=4)
                assert "too short" in str(excinfo.value)

        class TestSuccess:
            def are_equal(self, actual, expected) -> bool:
                return (
                    (actual if isinstance(actual, np.ndarray) else actual.numpy())
                    == (expected if isinstance(expected, np.ndarray) else expected.numpy())
                ).all()

            @pytest.mark.parametrize(
                "shift_back, shift_forward, expected_shift_back, expected_shift_forward",
                [
                    # 1 sample.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, 4], [10, 20, 30, 40]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92, 93, 94], [991, 992, 993, 994]]).T,
                            ]
                        ),
                        # expected_shift_back:
                        np.asarray(
                            [
                                np.asarray([[3, 4], [30, 40]]).T,
                            ]
                        ),
                        # expected_shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92], [991, 992]]).T,
                            ]
                        ),
                    ),
                    # 2 samples.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, 4], [10, 20, 30, 40]]).T,
                                np.asarray([[-1, -2, -3, -4], [-10, -20, -30, -40]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92, 93, 94], [991, 992, 993, 994]]).T,
                                np.asarray([[-91, -92, -93, -94], [-991, -992, -993, -994]]).T,
                            ]
                        ),
                        # expected_shift_back:
                        np.asarray(
                            [
                                np.asarray([[3, 4], [30, 40]]).T,
                                np.asarray([[-3, -4], [-30, -40]]).T,
                            ]
                        ),
                        # expected_shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92], [991, 992]]).T,
                                np.asarray([[-91, -92], [-991, -992]]).T,
                            ]
                        ),
                    ),
                ],
            )
            def test_success_shift_forward_non_empty_do_not_append(
                self, shift_back, shift_forward, expected_shift_back, expected_shift_forward
            ):
                for tensors_, shift_forward_ in zip(
                    (shift_back, torch.tensor(shift_back)), (shift_forward, torch.tensor(shift_forward))
                ):
                    # ^ Check both np.ndarray and torch.tensor cases.rs]):
                    t, o = n_step_shifted(
                        tensors_,
                        shift_forward_,
                        n_step=2,
                        validate_not_all_padding_=False,
                    )
                    assert are_equal(t, expected_shift_back)
                    assert are_equal(o, expected_shift_forward)

        class TestAllPaddingCheck:
            @pytest.mark.parametrize(
                "shift_back, shift_forward, padding_indicator, n_step, expectation",
                [
                    # Case: no padding values.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, 4], [10, 20, 30, 40]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92, 93, 94], [991, 992, 993, 994]]).T,
                            ]
                        ),
                        # padding_indicator:
                        777.0,
                        # n_step:
                        2,
                        # expectation:
                        does_not_raise(),
                    ),
                    # Case: not all padding in shift_back.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, 777.0], [10, 20, 30, 777.0]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92, 93, 94], [991, 992, 993, 994]]).T,
                            ]
                        ),
                        # padding_indicator:
                        777.0,
                        # n_step:
                        2,
                        # expectation:
                        does_not_raise(),
                    ),
                    # Case: all padding in shift_back.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, 777.0], [10, 20, 30, 777.0]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92, 93, 94], [991, 992, 993, 994]]).T,
                            ]
                        ),
                        # padding_indicator:
                        777.0,
                        # n_step:
                        3,
                        # expectation:
                        pytest.raises(ValueError),
                    ),
                    # Case: all padding in shift_back - check when padding indicator is nan.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, np.nan], [10, 20, 30, np.nan]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[91, 92, 93, 94], [991, 992, 993, 994]]).T,
                            ]
                        ),
                        # padding_indicator:
                        np.nan,
                        # n_step:
                        3,
                        # expectation:
                        pytest.raises(ValueError),
                    ),
                    # Case: not all padding in shift_forward.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, 4], [10, 20, 30, 40]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[31, 32, 33, 34], [331, 777.0, 333, 334]]).T,
                            ]
                        ),  # Throw in some padding values which shouldn't cause a problem.
                        # padding_indicator:
                        777.0,
                        # n_step:
                        1,
                        # expectation:
                        does_not_raise(),
                    ),
                    # Case: all padding in shift_forward.
                    (
                        # shift_back:
                        np.asarray(
                            [
                                np.asarray([[1, 2, 3, 4], [10, 20, 30, 40]]).T,
                            ]
                        ),
                        # shift_forward:
                        np.asarray(
                            [
                                np.asarray([[777.0, 92, 93, 94], [777.0, 992, 993, 994]]).T,  # <-- This one.
                            ]
                        ),
                        # padding_indicator:
                        777.0,
                        # n_step:
                        3,
                        # expectation:
                        pytest.raises(ValueError),
                    ),
                ],
            )
            def test_allow_all_padding_false(self, shift_back, shift_forward, padding_indicator, n_step, expectation):
                for tensors_, shift_forward_ in zip(
                    (shift_back, torch.tensor(shift_back)), (shift_forward, torch.tensor(shift_forward))
                ):
                    # ^ Check both np.ndarray and torch.tensor cases.
                    with expectation as excinfo:
                        _, _ = n_step_shifted(
                            tensors_,
                            shift_forward_,
                            n_step=n_step,
                            padding_indicator=padding_indicator,
                            validate_not_all_padding_=True,
                        )
                    if excinfo is not None:
                        assert "all padding" in str(excinfo.value)

    class TestComputeDeltas:
        @pytest.mark.parametrize(
            "tensor, expected_out, padding_indicator",
            [
                # Case: single sample.
                (
                    # tensor:
                    np.asarray(
                        [
                            np.asarray([[1, 3, 8, 777], [3, 23, 33, 777]]).T,
                        ]
                    ),
                    # expected_out:
                    np.asarray(
                        [
                            np.asarray([[0, 2, 5, 777], [0, 20, 10, 777]]).T,
                        ]
                    ),
                    # padding_indicator:
                    777,
                ),
                # Case: multiple samples.
                (
                    # tensor:
                    np.asarray(
                        [
                            np.asarray([[1, 3, 8, 777], [3, 23, 33, 777]]).T,
                            np.asarray([[1, 2, 777, 777], [8, 38, 777, 777]]).T,
                        ]
                    ),
                    # expected_out:
                    np.asarray(
                        [
                            np.asarray([[0, 2, 5, 777], [0, 20, 10, 777]]).T,
                            np.asarray([[0, 1, 777, 777], [0, 30, 777, 777]]).T,
                        ]
                    ),
                    # padding_indicator:
                    777,
                ),
                # Case: all padding.
                (
                    # tensor:
                    np.asarray(
                        [
                            np.asarray([[777, 777], [777, 777]]).T,
                        ]
                    ),
                    # expected_out:
                    np.asarray(
                        [
                            np.asarray([[777, 777], [777, 777]]).T,
                        ]
                    ),
                    # padding_indicator:
                    777,
                ),
                # Case: no padding.
                (
                    # tensor:
                    np.asarray(
                        [
                            np.asarray([[1, 3, 8], [3, 23, 33]]).T,
                        ]
                    ),
                    # expected_out:
                    np.asarray(
                        [
                            np.asarray([[0, 2, 5], [0, 20, 10]]).T,
                        ]
                    ),
                    # padding_indicator:
                    777,
                ),
                # Case: length 1.
                (
                    # tensor:
                    np.asarray(
                        [
                            np.asarray([[4], [31]]).T,
                        ]
                    ),
                    # expected_out:
                    np.asarray(
                        [
                            np.asarray([[0], [0]]).T,
                        ]
                    ),
                    # padding_indicator:
                    777,
                ),
            ],
        )
        def test_success(self, tensor, expected_out, padding_indicator):
            for t in (tensor, torch.tensor(tensor)):  # Check both np.ndarray and torch.tensor cases.
                out = compute_deltas(t, padding_indicator=padding_indicator)
                print(type(out))
                assert out.shape == t.shape
                assert are_equal(out, expected_out)
