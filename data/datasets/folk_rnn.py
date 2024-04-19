import os
import glob
import requests
from typing import Callable

from Pipeline.Pipeline import Pipeline
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
        download: bool = True,
        **kwargs
    ) -> None:
        self.data_type = data_type
        super().__init__(root, split, download, transform, target_transform, **kwargs)

        if data_type == "midi":
            self.file_list = glob.glob(os.path.join(os.path.join(self.root, "session_test"),"*.mid"))
        elif data_type == "tokenized_ABC":
            self.data_file = os.path.join(self.root, "train/data_v2.txt")

        self.pipeline = Pipeline(config_path='../../models/folk_rnn/config.yaml', model_name='folk_rnn')
        self.data = []
        self.targets = []

        if preload:
            self._load_data()

    def _load_data(self):
        self._data = []
        # self._targets = []
        if self.data_type == "midi":
            for midi_path in self.file_list:
                midi_data = self.pipeline.process(midi_path)
                self._data.append(midi_data)
        elif self.data_type == "tokenized_ABC":
            self.data = self.pipeline.process(self.data_file)
            # handle targets

    def __getitem__(self, index: int) -> tuple:
        midi_data = self.data[index]
        # target = self._targets[index]

        if self.transform:
            midi_data = self.transform(midi_data)
        # if self.target_transform:
        #     target = self.target_transform(target)

        return midi_data

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        if self.data_type == "tokenized_ABC":
            self.url = "https://raw.githubusercontent.com/IraKorshunova/folk-rnn/master/data/data_v2"

            response = requests.get(self.url)

            if response.status_code == 200:
                dest_path = os.path.join(
                    self.root,
                    "train",
                )
                os.makedirs(dest_path, exist_ok=True)
                with open(os.path.join(dest_path, "data_v2.txt"), "wb") as file:
                    file.write(response.content)
