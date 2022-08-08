from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, eq=True)
class NStepAheadHorizon:
    n_step: int

    def __post_init__(self):
        if self.n_step <= 0:
            raise ValueError("N step ahead horizon must be > 0.")


THorizon = Union[None, NStepAheadHorizon]  # Will expand.
