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
        data_type: str = "tokenized_ABC",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        preload: bool = True,
        ** kwargs
    ) -> None:
        super().__init__(root, split, False, transform, target_transform, **kwargs)
        
        self.data_type = data_type
        if data_type == 'midi':
            self.file_list = glob.glob(os.path.join(self.root, '*.mid'))
            self.pipeline = Pipeline(type='midi')
        elif data_type == 'tokenized_ABC':
            self.data_file = os.path.join(self.root, 'train/data_v2.txt')
            self.pipeline = Pipeline(type='tok_ABC')

        self._data = []
        self._targets = []

        if preload:
            self._load_data()

    def _load_data(self):
        self._data = []
        # self._targets = []
        if self.data_type == 'midi':
            for midi_path in self.file_list:
                midi_data = self.pipeline.process(midi_path)
                self._data.append(midi_data)
        elif self.data_type == 'tokenized_ABC':
            self._data = self.pipeline.process(self.data_file)
            # handle targets

    def __getitem__(self, index: int) -> tuple:
        midi_data = self._data[index]
        # target = self._targets[index]

        if self.transform:
            midi_data = self.transform(midi_data)
        # if self.target_transform:
        #     target = self.target_transform(target)9

        return midi_data

    def __len__(self) -> int:
        return len(self._data)

    def download(self) -> None:
        pass