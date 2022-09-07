import numpy as np
import pytest
import torch
import torch.nn as nn

from clairvoyance2.components.torch.rnn import (
    AutoregressiveMixin,
    RecurrentFFNet,
    RecurrentNet,
    apply_to_each_timestep,
    packed,
)

# pylint: disable=redefined-outer-name
# ^ Otherwise pylint trips up on pytest fixtures.


# TODO: Test unbatched case?
class TestRecurrentNet:
    @staticmethod
    def assert_basics(out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell):
        assert out.shape[0] == n_samples
        assert out.shape[1] == n_timesteps
        assert out.shape[-1] == out_dim
        if rnn_type == "LSTM":
            assert h_cell != 0
            h_n, c_n = h
            assert c_n.shape[1] == n_samples
            assert c_n.shape[0] == d_num_layers
            assert c_n.shape[2] == h_cell
        else:
            assert h_cell == 0
            h_n = h
        assert h_n.shape[1] == n_samples
        assert h_n.shape[0] == d_num_layers
        assert h_n.shape[2] == h_out

    @pytest.mark.parametrize("rnn_type", ["RNN", "LSTM", "GRU"])
    @pytest.mark.parametrize("input_size", [1, 5])
    @pytest.mark.parametrize("hidden_size", [1, 10])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_basics_common_params(self, rnn_type, input_size, hidden_size, num_layers, bidirectional):
        rnn = RecurrentNet(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            num_layers=num_layers,
            bidirectional=bidirectional,
            proj_size=0,
            bias=True,
            dropout=0.0,
        )
        n_samples = 5
        n_timesteps = 10
        x = torch.ones(size=(n_samples, n_timesteps, input_size))
        out, h = rnn(x, h=None)
        out_dim, (d_num_layers, h_out, h_cell) = rnn.get_output_and_h_dim()

        self.assert_basics(out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell)

    @pytest.mark.parametrize("proj_size", [None, 1, 3])
    def test_basics_lstm_proj_size(self, proj_size):
        rnn_type = "LSTM"
        input_size = 5
        hidden_size = 10
        num_layers = 2
        rnn = RecurrentNet(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=None,
            num_layers=num_layers,
            bidirectional=False,
            proj_size=proj_size,
            bias=True,
            dropout=0.0,
        )
        n_samples = 5
        n_timesteps = 10
        x = torch.ones(size=(n_samples, n_timesteps, input_size))
        out, h = rnn(x, h=None)
        out_dim, (d_num_layers, h_out, h_cell) = rnn.get_output_and_h_dim()

        self.assert_basics(out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell)

    @pytest.mark.parametrize("nonlinearity", [None, "tanh", "relu"])
    def test_basics_rnn_nonlinearity(self, nonlinearity):
        rnn_type = "RNN"
        input_size = 5
        hidden_size = 10
        num_layers = 2
        rnn = RecurrentNet(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            num_layers=num_layers,
            bidirectional=False,
            proj_size=None,
            bias=True,
            dropout=0.0,
        )
        n_samples = 5
        n_timesteps = 10
        x = torch.ones(size=(n_samples, n_timesteps, input_size))
        out, h = rnn(x, h=None)
        out_dim, (d_num_layers, h_out, h_cell) = rnn.get_output_and_h_dim()

        self.assert_basics(out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell)

    @pytest.mark.parametrize("rnn_type", ["RNN", "LSTM", "GRU"])
    @pytest.mark.parametrize("n_samples", [1, 7])
    @pytest.mark.parametrize("n_timesteps", [1, 7])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_basics_n_samples_n_timesteps(self, rnn_type, num_layers, bidirectional, n_samples, n_timesteps):
        input_size = 5
        hidden_size = 10
        rnn = RecurrentNet(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            num_layers=num_layers,
            bidirectional=bidirectional,
            proj_size=0,
            bias=True,
            dropout=0.0,
        )
        x = torch.ones(size=(n_samples, n_timesteps, input_size))
        out, h = rnn(x, h=None)
        out_dim, (d_num_layers, h_out, h_cell) = rnn.get_output_and_h_dim()

        self.assert_basics(out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell)


class TestRecurrentFFNet:
    @staticmethod
    def assert_basics(
        out, ff_out_size, rnn_out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell
    ):
        assert out.shape[0] == n_samples
        assert out.shape[1] == n_timesteps
        assert out.shape[2] == ff_out_size
        TestRecurrentNet.assert_basics(
            rnn_out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell
        )

    @pytest.mark.parametrize("rnn_type", ["RNN", "LSTM", "GRU"])
    @pytest.mark.parametrize("input_size", [1, 5])
    @pytest.mark.parametrize("hidden_size", [1, 10])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("ff_out_size", [1, 3])
    def test_basics_common_params(self, rnn_type, input_size, hidden_size, num_layers, bidirectional, ff_out_size):
        rnn_ff = RecurrentFFNet(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            num_layers=num_layers,
            bidirectional=bidirectional,
            proj_size=0,
            bias=True,
            dropout=0.0,
            ff_out_size=ff_out_size,
            ff_hidden_dims=[3],
        )
        n_samples = 5
        n_timesteps = 10
        x = torch.ones(size=(n_samples, n_timesteps, input_size))
        out, rnn_out, h = rnn_ff(x, h=None)
        out_dim, (d_num_layers, h_out, h_cell) = rnn_ff.rnn.get_output_and_h_dim()

        self.assert_basics(
            out, ff_out_size, rnn_out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell
        )

    @pytest.mark.parametrize("rnn_type", ["RNN", "LSTM", "GRU"])
    @pytest.mark.parametrize("input_size", [1, 5])
    @pytest.mark.parametrize("hidden_size", [1, 10])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("ff_out_size", [1, 3])
    def test_basics_common_params_x_has_padding_case(
        self, rnn_type, input_size, hidden_size, num_layers, bidirectional, ff_out_size
    ):
        rnn_ff = RecurrentFFNet(
            rnn_type=rnn_type,
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity="tanh",
            num_layers=num_layers,
            bidirectional=bidirectional,
            proj_size=0,
            bias=True,
            dropout=0.0,
            ff_out_size=ff_out_size,
            ff_hidden_dims=[3],
        )
        n_samples = 5
        n_timesteps = 10

        padding_indicator = 999.0
        x = torch.ones(size=(n_samples, n_timesteps, input_size))
        x[2, 3:, :] = padding_indicator
        x[4, 2:, :] = padding_indicator

        out, rnn_out, h = rnn_ff(x, h=None, padding_indicator=padding_indicator)
        out_dim, (d_num_layers, h_out, h_cell) = rnn_ff.rnn.get_output_and_h_dim()

        self.assert_basics(
            out, ff_out_size, rnn_out, h, n_samples, n_timesteps, out_dim, rnn_type, d_num_layers, h_out, h_cell
        )


@pytest.fixture
def test_module_x2_input():
    class TestModule(nn.Module):
        def forward(self, x):
            return x * 2

    return TestModule()


@pytest.fixture
def test_module_linear_2_1():
    class TestModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    return TestModule()


class TestApplyToEachTimestep:
    def test_fails_module_input_size_check(self, test_module_x2_input):
        input_ = torch.tensor(
            np.asarray(
                [
                    np.asarray([[1, 1], [2, 2]]).T,
                ]
            ),
            dtype=torch.float,
        )

        with pytest.raises(RuntimeError) as excinfo:
            _ = apply_to_each_timestep(
                module=test_module_x2_input,
                input_tensor=input_,
                output_size=2,
                concat_tensors=[],
                padding_indicator=None,
                expected_module_input_size=4,
            )
        assert "expected size" in str(excinfo.value)

    def test_fails_module_output_size(self, test_module_x2_input):
        input_ = torch.tensor(
            np.asarray(
                [
                    np.asarray([[1, 1], [2, 2]]).T,
                ]
            ),
            dtype=torch.float,
        )

        with pytest.raises(RuntimeError):
            # Error raised by torch.
            _ = apply_to_each_timestep(
                module=test_module_x2_input,
                input_tensor=input_,
                output_size=4,
                concat_tensors=[],
                padding_indicator=None,
                expected_module_input_size=None,
            )

    def test_no_padding(self, test_module_x2_input):
        input_ = torch.tensor(
            np.asarray(
                [
                    np.asarray([[1, 1, 1, 1], [2, 2, 2, 2]]).T,
                    np.asarray([[-1, -1, -1, -1], [-2, -2, -2, -2]]).T,
                    np.asarray([[10, 10, 10, 10], [20, 20, 20, 20]]).T,
                ]
            ),
            dtype=torch.float,
        )

        result = apply_to_each_timestep(
            module=test_module_x2_input,
            input_tensor=input_,
            output_size=2,
            concat_tensors=[],
            padding_indicator=None,
            expected_module_input_size=2,
        )

        assert result.shape == (3, 4, 2)
        assert (result == 2 * input_).all()

    def test_with_padding(self, test_module_x2_input):
        p_i = 999.0
        input_ = torch.tensor(
            np.asarray(
                [
                    np.asarray([[1, 1, 1, p_i], [2, 2, 2, p_i]]).T,
                    np.asarray([[-1, -1, p_i, p_i], [-2, -2, p_i, p_i]]).T,
                    np.asarray([[10, 10, 10, p_i], [20, 20, 20, p_i]]).T,
                ]
            ),
            dtype=torch.float,
        )

        result = apply_to_each_timestep(
            module=test_module_x2_input,
            input_tensor=input_,
            output_size=2,
            concat_tensors=[],
            padding_indicator=p_i,
            expected_module_input_size=2,
        )

        assert result.shape == (3, 4, 2)
        assert (
            result
            == torch.tensor(
                np.asarray(
                    [
                        np.asarray([[2, 2, 2, p_i], [4, 4, 4, p_i]]).T,
                        np.asarray([[-2, -2, p_i, p_i], [-4, -4, p_i, p_i]]).T,
                        np.asarray([[20, 20, 20, p_i], [40, 40, 40, p_i]]).T,
                    ]
                ),
                dtype=torch.float,
            )
        ).all()

    def test_with_padding_nan_case(self, test_module_x2_input):
        p_i = np.nan
        input_ = torch.tensor(
            np.asarray(
                [
                    np.asarray([[1, 1, 1, p_i], [2, 2, 2, p_i]]).T,
                    np.asarray([[-1, -1, p_i, p_i], [-2, -2, p_i, p_i]]).T,
                    np.asarray([[10, 10, 10, p_i], [20, 20, 20, p_i]]).T,
                ]
            ),
            dtype=torch.float,
        )

        result = apply_to_each_timestep(
            module=test_module_x2_input,
            input_tensor=input_,
            output_size=2,
            concat_tensors=[],
            padding_indicator=p_i,
            expected_module_input_size=2,
        )

        expected = torch.tensor(
            np.asarray(
                [
                    np.asarray([[2, 2, 2, p_i], [4, 4, 4, p_i]]).T,
                    np.asarray([[-2, -2, p_i, p_i], [-4, -4, p_i, p_i]]).T,
                    np.asarray([[20, 20, 20, p_i], [40, 40, 40, p_i]]).T,
                ]
            ),
            dtype=torch.float,
        )
        assert result.shape == (3, 4, 2)
        assert (result.isnan() == expected.isnan()).all()
        assert (result[~result.isnan()] == expected[~expected.isnan()]).all()

    def test_with_padding_case_module_linear(self, test_module_linear_2_1):
        p_i = 999.0
        input_ = torch.tensor(
            np.asarray(
                [
                    np.asarray([[1, 1, 1, p_i], [2, 2, 2, p_i]]).T,
                    np.asarray([[-1, -1, p_i, p_i], [-2, -2, p_i, p_i]]).T,
                    np.asarray([[10, 10, 10, p_i], [20, 20, 20, p_i]]).T,
                ]
            ),
            dtype=torch.float,
        )

        result = apply_to_each_timestep(
            module=test_module_linear_2_1,
            input_tensor=input_,
            output_size=1,
            concat_tensors=[],
            padding_indicator=p_i,
            expected_module_input_size=2,
        )

        assert result.shape == (3, 4, 1)
        assert (  # Check padding was in the right locations.
            (result == p_i)
            == torch.tensor(
                np.asarray(
                    [
                        np.asarray([[False, False, False, True]]).T,
                        np.asarray([[False, False, True, True]]).T,
                        np.asarray([[False, False, False, True]]).T,
                    ]
                ),
                dtype=torch.float,
            )
        ).all()

    class TestConcatTensors:
        def test_all_padding(self, test_module_x2_input):
            p_i = 999.0
            input_ = torch.tensor(
                np.asarray(
                    [
                        np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                        np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                        np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                    ]
                ),
                dtype=torch.float,
            )

            result = apply_to_each_timestep(
                module=test_module_x2_input,
                input_tensor=input_,
                output_size=2,
                concat_tensors=[],
                padding_indicator=p_i,
                expected_module_input_size=2,
            )

            assert result.shape == (3, 2, 2)
            assert (
                result
                == torch.tensor(
                    np.asarray(
                        [
                            np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                            np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                            np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                        ]
                    ),
                    dtype=torch.float,
                )
            ).all()

        def test_with_padding_concatenate_items(self, test_module_x2_input):
            p_i = 999.0
            input_ = torch.tensor(
                np.asarray(
                    [
                        np.asarray([[1, 1, 1, p_i], [2, 2, 2, p_i]]).T,
                        np.asarray([[-1, -1, p_i, p_i], [-2, -2, p_i, p_i]]).T,
                        np.asarray([[10, 10, 10, p_i], [20, 20, 20, p_i]]).T,
                    ]
                ),
                dtype=torch.float,
            )
            concat_tensors = [
                torch.tensor(
                    np.asarray(
                        [
                            np.asarray([4, 8]).T,
                            np.asarray([-4, -8]).T,
                            np.asarray([40, 80]).T,
                        ]
                    ),
                    dtype=torch.float,
                ),
                torch.tensor(
                    np.asarray(
                        [
                            np.asarray([7, 9]).T,
                            np.asarray([-7, -9]).T,
                            np.asarray([70, 90]).T,
                        ]
                    ),
                    dtype=torch.float,
                ),
            ]

            result = apply_to_each_timestep(
                module=test_module_x2_input,
                input_tensor=input_,
                output_size=6,
                concat_tensors=concat_tensors,
                padding_indicator=p_i,
                expected_module_input_size=6,
            )

            assert result.shape == (3, 4, 6)
            assert (
                result
                == torch.tensor(
                    np.asarray(
                        [
                            np.asarray(
                                [
                                    [2, 2, 2, p_i],
                                    [4, 4, 4, p_i],
                                    [8, 8, 8, p_i],
                                    [16, 16, 16, p_i],
                                    [14, 14, 14, p_i],
                                    [18, 18, 18, p_i],
                                ]
                            ).T,
                            np.asarray(
                                [
                                    [-2, -2, p_i, p_i],
                                    [-4, -4, p_i, p_i],
                                    [-8, -8, p_i, p_i],
                                    [-16, -16, p_i, p_i],
                                    [-14, -14, p_i, p_i],
                                    [-18, -18, p_i, p_i],
                                ]
                            ).T,
                            np.asarray(
                                [
                                    [20, 20, 20, p_i],
                                    [40, 40, 40, p_i],
                                    [80, 80, 80, p_i],
                                    [160, 160, 160, p_i],
                                    [140, 140, 140, p_i],
                                    [180, 180, 180, p_i],
                                ]
                            ).T,
                        ]
                    ),
                    dtype=torch.float,
                )
            ).all()

        def test_all_padding_concatenate_items(self, test_module_x2_input):
            p_i = 999.0
            input_ = torch.tensor(
                np.asarray(
                    [
                        np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                        np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                        np.asarray([[p_i, p_i], [p_i, p_i]]).T,
                    ]
                ),
                dtype=torch.float,
            )
            concat_tensors = [
                torch.tensor(
                    np.asarray(
                        [
                            np.asarray([4]).T,
                            np.asarray([-4]).T,
                            np.asarray([40]).T,
                        ]
                    ),
                    dtype=torch.float,
                ),
                torch.tensor(
                    np.asarray(
                        [
                            np.asarray([7]).T,
                            np.asarray([-7]).T,
                            np.asarray([70]).T,
                        ]
                    ),
                    dtype=torch.float,
                ),
            ]

            result = apply_to_each_timestep(
                module=test_module_x2_input,
                input_tensor=input_,
                output_size=4,
                concat_tensors=concat_tensors,
                padding_indicator=p_i,
                expected_module_input_size=4,
            )

            assert result.shape == (3, 2, 4)
            assert (
                result
                == torch.tensor(
                    np.asarray(
                        [
                            np.asarray([[p_i, p_i], [p_i, p_i], [p_i, p_i], [p_i, p_i]]).T,
                            np.asarray([[p_i, p_i], [p_i, p_i], [p_i, p_i], [p_i, p_i]]).T,
                            np.asarray([[p_i, p_i], [p_i, p_i], [p_i, p_i], [p_i, p_i]]).T,
                        ]
                    ),
                    dtype=torch.float,
                )
            ).all()


class ModuleForAutoregressive_NFeaturesUnchanged(nn.Module):
    def forward(self, x, dummy1, dummy2=0):
        self.dummy1 = dummy1  # pylint: disable=attribute-defined-outside-init
        self.dummy2 = dummy2  # pylint: disable=attribute-defined-outside-init
        return x * 2, "other_stuff"


class ModuleForAutoregressive_NFeatures2(nn.Module):
    def forward(self, x, dummy1, dummy2=0):
        self.dummy1 = dummy1  # pylint: disable=attribute-defined-outside-init
        self.dummy2 = dummy2  # pylint: disable=attribute-defined-outside-init
        return (x * 2)[:, :, :2], "other_stuff"


class TestAutoregressiveMixin:
    @pytest.mark.parametrize("feed_first_n", [None, 2])
    def test_success_one_sample(self, feed_first_n):
        class A(ModuleForAutoregressive_NFeaturesUnchanged, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=feed_first_n)
                ModuleForAutoregressive_NFeaturesUnchanged.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                out, _ = self.forward(x, **kwargs)
                return out

        a = A()
        x = torch.ones(1, 5, 2)

        out = a.autoregress(x, dummy1=1, dummy2=2)

        assert a.dummy1 == 1
        assert a.dummy2 == 2
        assert out.shape == (1, 5, 2)
        assert a.x_used_in_autoregress.shape == (1, 5, 2)
        assert (out == torch.tensor([[[2.0, 2.0], [4.0, 4.0], [8.0, 8.0], [16.0, 16.0], [32.0, 32.0]]])).all()
        assert (
            a.x_used_in_autoregress == torch.tensor([[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0], [16.0, 16.0]]])
        ).all()

    @pytest.mark.parametrize("feed_first_n", [None, 2])
    def test_success_multiple_samples(self, feed_first_n):
        class A(ModuleForAutoregressive_NFeaturesUnchanged, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=feed_first_n)
                ModuleForAutoregressive_NFeaturesUnchanged.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                out, _ = self.forward(x, **kwargs)
                return out

        a = A()
        x = torch.ones(3, 5, 2)
        x[1, :, :] = x[1, :, :] * -1
        x[2, :, :] = 0.0

        out = a.autoregress(x, dummy1=1, dummy2=2)

        assert a.dummy1 == 1
        assert a.dummy2 == 2
        assert out.shape == (3, 5, 2)
        assert a.x_used_in_autoregress.shape == (3, 5, 2)
        assert (
            out
            == torch.tensor(
                [
                    [[2.0, 2.0], [4.0, 4.0], [8.0, 8.0], [16.0, 16.0], [32.0, 32.0]],
                    [[-2.0, -2.0], [-4.0, -4.0], [-8.0, -8.0], [-16.0, -16.0], [-32.0, -32.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()
        assert (
            a.x_used_in_autoregress
            == torch.tensor(
                [
                    [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [8.0, 8.0], [16.0, 16.0]],
                    [[-1.0, -1.0], [-2.0, -2.0], [-4.0, -4.0], [-8.0, -8.0], [-16.0, -16.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test_success_feed_first_is_relevant(self):
        class A(ModuleForAutoregressive_NFeatures2, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=2)
                ModuleForAutoregressive_NFeatures2.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                out, _ = self.forward(x, **kwargs)
                return out

        a = A()
        x = torch.ones(1, 5, 4)

        out = a.autoregress(x, dummy1=1, dummy2=2)

        assert a.dummy1 == 1
        assert a.dummy2 == 2
        assert out.shape == (1, 5, 2)
        assert a.x_used_in_autoregress.shape == (1, 5, 4)
        assert (out == torch.tensor([[[2.0, 2.0], [4.0, 4.0], [8.0, 8.0], [16.0, 16.0], [32.0, 32.0]]])).all()
        assert (
            a.x_used_in_autoregress
            == torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 1.0, 1.0],
                        [4.0, 4.0, 1.0, 1.0],
                        [8.0, 8.0, 1.0, 1.0],
                        [16.0, 16.0, 1.0, 1.0],
                    ]
                ]
            )
        ).all()

    def test_fail_x_wrong_ndim(self):
        class A(ModuleForAutoregressive_NFeaturesUnchanged, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=None)
                ModuleForAutoregressive_NFeaturesUnchanged.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                out, _ = self.forward(x, **kwargs)
                return out

        a = A()
        x = torch.ones(1, 5, 4, 7)

        with pytest.raises(RuntimeError) as excinfo:
            a.autoregress(x, dummy1=1, dummy2=2)
        assert "`x`" in str(excinfo.value) and "3 dimensions" in str(excinfo.value)

    def test_fail_x_and_feed_first_n_mismatch(self):
        class A(ModuleForAutoregressive_NFeaturesUnchanged, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=3)
                ModuleForAutoregressive_NFeaturesUnchanged.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                out, _ = self.forward(x, **kwargs)
                return out

        a = A()
        x = torch.ones(1, 5, 2)

        with pytest.raises(RuntimeError) as excinfo:
            a.autoregress(x, dummy1=1, dummy2=2)
        assert "`x`" in str(excinfo.value) and "`feed_first_n`" in str(excinfo.value)

    def test_fail_output_wrong_ndim(self):
        class Temp(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], x.shape[1], x.shape[2], 7)

        class A(Temp, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=None)
                Temp.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                return self.forward(x)

        a = A()
        x = torch.ones(1, 5, 2)

        with pytest.raises(RuntimeError) as excinfo:
            a.autoregress(x)
        assert "output" in str(excinfo.value) and "3 dimensions" in str(excinfo.value)

    def test_fail_output_feed_first_n_mismatch(self):
        class Temp(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], x.shape[1], 1)

        class A(Temp, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=3)
                Temp.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                return self.forward(x)

        a = A()
        x = torch.ones(1, 5, 3)

        with pytest.raises(RuntimeError) as excinfo:
            a.autoregress(x)
        assert "output" in str(excinfo.value) and "`feed_first_n`" in str(excinfo.value)

    def test_fail_output_x_mismatch(self):
        class Temp(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0], x.shape[1], 1)

        class A(Temp, AutoregressiveMixin):
            def __init__(self) -> None:
                AutoregressiveMixin.__init__(self, feed_first_n=None)
                Temp.__init__(self)

            def _forward_for_autoregress(self, x: torch.Tensor, timestep_idx: int, **kwargs) -> torch.Tensor:
                return self.forward(x)

        a = A()
        x = torch.ones(1, 5, 3)

        with pytest.raises(RuntimeError) as excinfo:
            a.autoregress(x)
        assert "output" in str(excinfo.value) and "`x`" in str(excinfo.value) and "same size" in str(excinfo.value)


@pytest.fixture
def test_module_for_packed():
    class TestModule(nn.RNN):
        def __init__(self):
            super().__init__(input_size=2, hidden_size=2)

    return TestModule()


class TestPacked:
    def test_fails_wrong_dim(self):
        pi = 999.0
        x = torch.ones(size=(2, 2))
        with pytest.raises(RuntimeError) as excinfo:
            with packed(x, padding_indicator=pi) as _:
                pass
        assert "3 dimensional" in str(excinfo.value)

    def test_fails_batch_first_false_not_implemented(self):
        pi = 999.0
        x = torch.ones(size=(3, 4, 2))
        with pytest.raises(NotImplementedError) as excinfo:
            with packed(x, padding_indicator=pi, batch_first=False) as _:
                pass
        assert "batch_first" in str(excinfo.value)

    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_packed_typical(self, test_module_for_packed, padding_indicator):
        x = torch.ones(size=(3, 4, 2))
        x[1, -2:, :] = padding_indicator
        x[2, -1:, :] = padding_indicator

        with packed(x, padding_indicator=padding_indicator) as p:
            p.packed, _ = test_module_for_packed(p.packed)

        assert p.unpacked.shape == x.shape
        if not np.isnan(padding_indicator):
            assert (p.unpacked[1, -2:, :] == padding_indicator).all()
            assert (p.unpacked[2, -1:, :] == padding_indicator).all()
            assert (p.unpacked[1, :2, :] != padding_indicator).all()
            assert (p.unpacked[2, :3, :] != padding_indicator).all()
        else:
            assert (p.unpacked[1, -2:, :].isnan()).all()
            assert (p.unpacked[2, -1:, :].isnan()).all()
            assert (~p.unpacked[1, :2, :].isnan()).all()
            assert (~p.unpacked[2, :3, :].isnan()).all()
        assert list(p.unpacked_lens) == [4, 2, 3]

    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_packed_only_some_features_have_padding_indicator(self, test_module_for_packed, padding_indicator):
        x = torch.ones(size=(3, 4, 2))
        x[1, -2:, 0] = padding_indicator
        x[2, -1:, 1] = padding_indicator

        with packed(x, padding_indicator=padding_indicator) as p:
            p.packed, _ = test_module_for_packed(p.packed)

        assert p.unpacked.shape == x.shape
        if not np.isnan(padding_indicator):
            assert (p.unpacked[1, -2:, :] == padding_indicator).all()
            assert (p.unpacked[2, -1:, :] == padding_indicator).all()
            assert (p.unpacked[1, :2, :] != padding_indicator).all()
            assert (p.unpacked[2, :3, :] != padding_indicator).all()
        else:
            assert (p.unpacked[1, -2:, :].isnan()).all()
            assert (p.unpacked[2, -1:, :].isnan()).all()
            assert (~p.unpacked[1, :2, :].isnan()).all()
            assert (~p.unpacked[2, :3, :].isnan()).all()
        assert list(p.unpacked_lens) == [4, 2, 3]

    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_packed_padding_not_at_end(self, padding_indicator):
        x = torch.ones(size=(3, 4, 2))
        x[2, 1, 1] = padding_indicator

        with pytest.raises(RuntimeError) as excinfo:
            with packed(x, padding_indicator=padding_indicator) as _:
                pass
        assert "not at the end" in str(excinfo.value)

    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_packed_has_all_padding_cases(self, test_module_for_packed, padding_indicator):
        x = torch.ones(size=(3, 4, 2))
        x[0, :, :] = padding_indicator  # All padding
        x[1, -2:, :] = padding_indicator
        x[2, -1:, :] = padding_indicator

        with packed(x, padding_indicator=padding_indicator) as p:
            p.packed, _ = test_module_for_packed(p.packed)

        assert p.unpacked.shape == x.shape
        if not np.isnan(padding_indicator):
            assert (p.unpacked[0, :, :] == padding_indicator).all()
            assert (p.unpacked[1, -2:, :] == padding_indicator).all()
            assert (p.unpacked[2, -1:, :] == padding_indicator).all()
            assert (p.unpacked[1, :2, :] != padding_indicator).all()
            assert (p.unpacked[2, :3, :] != padding_indicator).all()
        else:
            assert (p.unpacked[0, :, :].isnan()).all()
            assert (p.unpacked[1, -2:, :].isnan()).all()
            assert (p.unpacked[2, -1:, :].isnan()).all()
            assert (~p.unpacked[1, :2, :].isnan()).all()
            assert (~p.unpacked[2, :3, :].isnan()).all()
        assert list(p.unpacked_lens) == [0, 2, 3]
