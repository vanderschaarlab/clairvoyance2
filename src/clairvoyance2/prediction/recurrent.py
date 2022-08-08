from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from ..data import DEFAULT_PADDING_INDICATOR, Dataset, TimeSeries
from ..interface import PredictorModel, TDefaultParams, THorizon, TParams
from ..interface.requirements import (
    DatasetRequirements,
    PredictionRequirements,
    PredictionTarget,
    PredictionTask,
    Requirements,
)
from ..utils.converters import to_torch_dataset
from ..utils.dev import NEEDED, raise_not_implemented

RNNClass = Union[Type[nn.RNN], Type[nn.LSTM], Type[nn.GRU]]
RNNHidden = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

RNN_CLASS_MAP: Mapping[str, RNNClass] = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}

OPTIM_MAP: Mapping[str, Type[torch.optim.Optimizer]] = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    # TODO: Allow more.
}

_DEBUG = False


class _RecurrentModule(nn.Module):
    def __init__(
        self,
        rnn_class: RNNClass,
        input_size: int,
        hidden_size: int,
        nonlinearity: Optional[str],
        num_layers: int,
        bias: bool,
        dropout: float,
        bidirectional: bool,
        proj_size: Optional[int],
    ) -> None:
        super().__init__()

        kwargs: Dict[str, Any] = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,  # NOTE: We adopt batch first convention.
            dropout=dropout,
            bidirectional=bidirectional,
        )
        if rnn_class == nn.RNN:
            kwargs["nonlinearity"] = nonlinearity
        if rnn_class == nn.LSTM:
            kwargs["proj_size"] = proj_size

        self.rnn = rnn_class(**kwargs)

    def forward(self, x: torch.Tensor, h: Optional[RNNHidden]) -> Tuple[torch.Tensor, RNNHidden]:
        if h is not None:
            rnn_out, h_out = self.rnn(x, h)
        else:
            rnn_out, h_out = self.rnn(x)
        return rnn_out, h_out


class _DefaultParams(NamedTuple):
    rnn_model_str: str
    hidden_size: int
    num_layers: int
    bias: bool
    dropout: float
    bidirectional: bool
    nonlinearity: Optional[str]
    proj_size: Optional[int]
    max_len: Optional[int]
    device_str: str
    optimizer_str: str
    optimizer_kwargs: Mapping[str, Any]
    batch_size: int
    epochs: int


class RecurrentPredictor(PredictorModel):
    requirements: Requirements = Requirements(
        dataset_requirements=DatasetRequirements(
            requires_time_series_samples_regular=True,
            requires_time_series_index_numeric=True,
            requires_no_missing_data=True,
        ),
        prediction_requirements=PredictionRequirements(
            task=PredictionTask.REGRESSION,
            target=PredictionTarget.TIME_SERIES,
        ),
    )
    DEFAULT_PARAMS: TDefaultParams = _DefaultParams(
        rnn_model_str="LSTM",
        hidden_size=100,
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        nonlinearity=None,
        proj_size=1,
        max_len=None,
        device_str="cpu",
        optimizer_str="Adam",
        optimizer_kwargs=dict(lr=0.01),
        batch_size=32,
        epochs=100,
    )

    torch_dtype = torch.float
    padding_value = DEFAULT_PADDING_INDICATOR

    def __init__(
        self,
        params: Optional[TParams] = None,
    ) -> None:
        super().__init__(params)

        if self.params.rnn_model_str != "LSTM":
            # TODO: Need to implement others by adding a FF NN on the output (instead of inbuilt projection).
            raise_not_implemented("Non-LSTM recurrent prediction models")

        self.rnn: Optional[_RecurrentModule] = NEEDED
        self.optim: Optional[torch.optim.Optimizer] = NEEDED
        self.input_size: Optional[int] = NEEDED

        self.loss_fn = nn.MSELoss()
        self.device = torch.device(self.params.device_str)

    def _init_rnn(self, input_size: int) -> None:
        self.rnn = _RecurrentModule(
            rnn_class=RNN_CLASS_MAP[self.params.rnn_model_str],
            input_size=input_size,
            hidden_size=self.params.hidden_size,
            nonlinearity=self.params.nonlinearity,
            num_layers=self.params.num_layers,
            bias=self.params.bias,
            dropout=self.params.dropout,
            bidirectional=self.params.bidirectional,
            proj_size=self.params.proj_size,
        )
        self.input_size = input_size

    # TODO: Historic targets are not used, make them used (& in predict).
    # TODO: Time step deltas are not used, make them used (& in predict).
    def _fit(self, data: Dataset, horizon: THorizon = NEEDED) -> "RecurrentPredictor":
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None
        assert horizon is not None

        # Initialize the underlying RNN.
        self.params.proj_size = data.temporal_targets.n_features  # TODO: This will change once redone properly.
        self._init_rnn(input_size=data.temporal_covariates.n_features)
        assert self.rnn is not None
        assert self.input_size is not None
        self.rnn.to(self.device, dtype=self.torch_dtype)

        # Initialize optimizer.
        self.optim = OPTIM_MAP[self.params.optimizer_str](params=self.rnn.parameters(), **self.params.optimizer_kwargs)

        self.params.batch_size = min(self.params.batch_size, data.n_samples)
        # NOTE: ^ Override the params if batch_size had to be changed, to accurately record the fit-time params.

        torch_dataset = to_torch_dataset(data, padding_indicator=self.padding_value, max_len=self.params.max_len)
        dataloader = DataLoader(torch_dataset, batch_size=self.params.batch_size, shuffle=True)

        self.rnn.train()
        for epoch_idx in range(self.params.epochs):

            n_samples = 0.0
            epoch_loss = 0.0
            for batch_idx, (t_cov, t_targ, _) in enumerate(dataloader):  # pylint: disable=unused-variable
                n_samples += t_cov.shape[0]
                t_cov, t_targ = t_cov.to(self.device), t_targ.to(self.device)

                # TODO: Check for "too short" case.
                x = t_cov[:, : -horizon.n_step, :]
                y = t_targ[:, horizon.n_step :, :]
                max_len = x.shape[1]

                if _DEBUG is True:  # pragma: no cover
                    print("t_targ.shape", t_targ.shape)
                    print("x.shape", x.shape)
                    print("y.shape", y.shape)
                    # print(x)

                x_seq_lens = (x.detach()[:, :, 0] != self.padding_value).sum(axis=1)

                if _DEBUG is True:  # pragma: no cover
                    print("x_seq_lens.shape", x_seq_lens.shape)
                    print("x_seq_lens", x_seq_lens)
                    print(x_seq_lens.shape)
                    # for idx, f in enumerate(x_seq_lens):
                    #     y[idx, -(max_len - f) :, :] = self.padding_value
                    # y_seq_lens = (y.detach()[:, :, 0] != self.padding_value).sum(axis=1)
                    # print("y_seq_lens", y_seq_lens)

                x = pack_padded_sequence(x, x_seq_lens, batch_first=True, enforce_sorted=False)
                out, _ = self.rnn(x, h=None)
                out, _ = pad_packed_sequence(
                    out, batch_first=True, padding_value=self.padding_value, total_length=max_len
                )

                if _DEBUG is True:  # pragma: no cover
                    print("out.shape", out.shape)
                    # out_seq_lens = (out.detach()[:, :, 0] != self.padding_value).sum(axis=1)
                    # print("out_seq_lens", out_seq_lens)

                # This shouldn't be necessary but just in case:
                out[out == self.padding_value] = 0.0
                y[y == self.padding_value] = 0.0

                loss = self.loss_fn(out, y)  # Padding values are equal, so should be fine.

                # Optimization:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item() * n_samples

                if _DEBUG is True:  # pragma: no cover
                    print(f"{batch_idx}: {loss.item()}")

            epoch_loss /= n_samples
            print(f"Epoch: {epoch_idx}, Loss: {epoch_loss}")

        return self

    def _predict(self, data: Dataset, horizon: THorizon = NEEDED) -> Dataset:
        assert data.temporal_covariates is not None
        assert data.temporal_targets is not None
        assert self.rnn is not None
        assert horizon is not None

        batch_size = min(self.params.batch_size, data.n_samples)
        # NOTE: ^ Do not override the params even if batch_size had to be changed, as doesn't affect fit-time params.

        torch_dataset = to_torch_dataset(data, padding_indicator=self.padding_value, max_len=self.params.max_len)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

        self.rnn.eval()
        list_batches = []
        with torch.no_grad():
            for batch_idx, (t_cov, t_targ, _) in enumerate(dataloader):  # pylint: disable=unused-variable
                t_cov, t_targ = t_cov.to(self.device), t_targ.to(self.device)
                x = t_cov[:, : -horizon.n_step, :]
                max_len = x.shape[1]
                x_seq_lens = (x.detach()[:, :, 0] != self.padding_value).sum(axis=1)

                x = pack_padded_sequence(x, x_seq_lens, batch_first=True, enforce_sorted=False)
                out, _ = self.rnn(x, h=None)
                out, _ = pad_packed_sequence(
                    out, batch_first=True, padding_value=self.padding_value, total_length=max_len
                )

                list_batches.append(out.detach().cpu().numpy())

        result = np.concatenate(list_batches).astype(float)

        data = data.copy()
        assert data.temporal_targets is not None

        # TODO: Extract this into some kind of an helper function.
        max_len = max(data.temporal_targets.n_timesteps_per_sample)
        for sample_idx, sample_array in zip(data.temporal_targets.sample_indices, result):
            if _DEBUG is True:  # pragma: no cover
                print("sample_array\n", sample_array)
                print("sample_array.shape:", sample_array.shape)

            ts: TimeSeries = data.temporal_targets[sample_idx]
            assert (sample_array[ts.n_timesteps :, :] == self.padding_value).all()  # Check padding is correct.

            if _DEBUG is True:  # pragma: no cover
                print("ts.n_timesteps", ts.n_timesteps)
            is_regular, diff = ts.is_regular()
            assert is_regular

            last_index = ts.df.index[-1]
            if _DEBUG is True:  # pragma: no cover
                print("last_index", last_index)

            # TODO: Add below checks properly.
            # assert isinstance(last_index, (float, int))
            # assert isinstance(diff, (float, int))
            new_indices = [last_index + i * diff for i in range(1, horizon.n_step + 1)]  # type: ignore
            if _DEBUG is True:  # pragma: no cover
                print(new_indices)

            ts.df = ts.df.loc[horizon.n_step :, :].copy()
            for new_index in new_indices:
                ts.df.loc[new_index, :] = np.nan

            # Max length limit in the n-step ahead setting:
            if len(ts.df) > max_len - horizon.n_step:
                ts.df = ts.df.loc[: (max_len - horizon.n_step), :].copy()

            ts.df[:] = sample_array[: ts.n_timesteps, :]
            if _DEBUG is True:  # pragma: no cover
                print("ts.df\n", ts.df)
                print("---")

        return data
