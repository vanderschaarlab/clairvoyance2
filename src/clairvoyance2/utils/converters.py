from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..data import DEFAULT_PADDING_INDICATOR, Dataset


class ClairvoyanceTorchDataset(TorchDataset):
    def __init__(
        self,
        temporal_covariates: np.ndarray,
        temporal_targets: Optional[np.ndarray],
        static_covariates: Optional[np.ndarray],
    ) -> None:
        self.temporal_covariates = temporal_covariates
        self.temporal_targets = temporal_targets
        self.static_covariates = static_covariates
        super().__init__()

    def __len__(self) -> int:
        return self.temporal_covariates.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        temporal_covariates = self.temporal_covariates[idx, :, :]
        temporal_targets = self.temporal_targets[idx, :, :] if self.temporal_targets is not None else None
        static_covariates = self.static_covariates[idx, :] if self.static_covariates is not None else None

        len_ = temporal_covariates.shape[0]
        temporal_covariates_as_tensor = torch.tensor(temporal_covariates, dtype=torch.float)
        if static_covariates is not None:
            static_covariates_as_tensor = torch.tensor(static_covariates, dtype=torch.float)
        else:
            static_covariates_as_tensor = torch.tensor(np.full(shape=(len_,), fill_value=np.nan), dtype=torch.float)

        if temporal_targets is not None:
            temporal_targets_as_tensor = torch.tensor(temporal_targets, dtype=torch.float)
        else:
            temporal_targets_as_tensor = torch.tensor(np.full(shape=(len_,), fill_value=np.nan), dtype=torch.float)

        return temporal_covariates_as_tensor, temporal_targets_as_tensor, static_covariates_as_tensor


# TODO: Should also return time series indexes.
def to_torch_dataset(
    data: Dataset,
    padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    max_len: Optional[int] = None,
) -> TorchDataset:
    temporal_covariates, static_covariates, temporal_targets = data

    temporal_covariates_np = temporal_covariates.to_numpy(padding_indicator, max_len)
    temporal_targets_np = (
        temporal_targets.to_numpy(padding_indicator, max_len) if temporal_targets is not None else None
    )

    return ClairvoyanceTorchDataset(
        temporal_covariates=temporal_covariates_np,
        temporal_targets=temporal_targets_np,
        static_covariates=static_covariates.to_numpy() if static_covariates is not None else None,
    )
