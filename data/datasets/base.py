from typing import Callable
from abc import abstractmethod, ABC
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

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
        train_size: float = 0.8,
        val_size: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = 1 - train_size - val_size
        self.random_state = random_state
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

    def split_data(self, data: np.ndarray) -> np.ndarray:
        train_data, temp_data = train_test_split(data, train_size=self.train_size, shuffle=True, random_state=self.random_state)
        val_data, test_data = train_test_split(temp_data, test_size=self.test_size/(self.test_size + self.val_size), shuffle=True, random_state=self.random_state)

        if self.split == "train":
            return train_data
        elif self.split == "val":
            return val_data
        elif self.split == "test":
            return test_data
        else:
            raise ValueError(f"Invalid split name: {self.split}. Expected one of ['train', 'val', 'test']")

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
