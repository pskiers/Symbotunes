import os
import glob
import requests
from torch.utils.data import Dataset
from Pipeline.Pipeline import Pipeline
from typing import Callable, List
from base import BaseDataset

class FolkRnnDataset(BaseDataset):
    """
    Folk-Rnn Dataset class
    available on: https://github.com/IraKorshunova/folk-rnn
    """

    def __init__(
        self,
        root: str = "_data",
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        preload: bool = True,
        ** kwargs
    ) -> None:
        super().__init__(root, split, False, transform, target_transform, **kwargs)
        
        self.file_list = glob.glob(os.path.join(root, '*.mid'))
        self.pipeline = Pipeline(type='midi')

        self._data = []
        self._targets = []

        if preload:
            self._load_data()

    def _load_data(self):
        self._data = []
        # self._targets = []
        for midi_path in self.file_list:
            midi_data = self.pipeline.process(midi_path)
            self._data.append(midi_data)
            # handle targets

    def __getitem__(self, index: int) -> tuple:
        midi_data = self._data[index]
        # target = self._targets[index]

        if self.transform:
            midi_data = self.transform(midi_data)
        # if self.target_transform:
        #     target = self.target_transform(target)

        return midi_data

    def __len__(self) -> int:
        return len(self._data)

    def download(self) -> None:
        pass