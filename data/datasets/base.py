from typing import Callable
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.utils.data as data


class BaseDataset(data.Dataset, ABC):
    """
    Base dataset interface
    """

    @abstractmethod
    def __init__(
        self,
        root: str = "_data",
        split: str = "train",
        download: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        if download:
            self.download()

        self._transform = transform

        self._data: torch.Tensor | np.ndarray

    @property
    def data(self) -> torch.Tensor | np.ndarray:
        return self._data

    @data.setter
    def data(self, data: torch.Tensor | np.ndarray) -> None:
        self._data = data

    @property
    def transform(self) -> Callable | None:
        return self._transform

    @transform.setter
    def transform(self, transform: Callable) -> None:
        self._transform = transform

    def __getitem__(self, index) -> torch.Tensor | np.ndarray:
        data = self.data[index]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def download(self) -> None:
        """Download dataset"""
