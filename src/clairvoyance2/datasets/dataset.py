import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, TypeVar

from . import TDataset
from .download import download_file

TUrl = TypeVar("TUrl", bound=str)
TDatasetFileDef = Tuple[TUrl, str]  # ("URL", "local_file_name")

DATASET_ROOT_DIR = os.path.join(os.path.expanduser("~"), ".clairvoyance/datasets/")


# TODO: Unit test.
class DatasetRetriever(ABC):
    dataset_subdir: str
    dataset_files: Optional[Sequence[TDatasetFileDef]]

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.dataset_root_dir, self.dataset_subdir)

    def __init__(self, data_home: Optional[str] = None) -> None:
        if data_home is None:
            self.dataset_root_dir = DATASET_ROOT_DIR
        else:
            self.dataset_root_dir = data_home

    def download_dataset(self) -> None:
        if self.dataset_files is not None:
            for dataset_file in self.dataset_files:
                url, file_name = dataset_file
                download_file(url, os.path.join(self.dataset_dir, file_name))

    @abstractmethod
    def prepare(self) -> TDataset:
        # Prepare the dataset and return it.
        ...

    def retrieve(self) -> TDataset:
        # Download dataset files (if required).
        if self.dataset_files is not None:
            if any([not os.path.exists(os.path.join(self.dataset_dir, f)) for _, f in self.dataset_files]):
                self.download_dataset()
        # Prepare and retrieve dataset.
        return self.prepare()
