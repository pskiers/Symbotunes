import os
import glob
import requests
from typing import Callable

from .Pipeline.Pipeline import Pipeline
from .base import BaseDataset


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
        replace_if_exists: bool = True,
        **kwargs
    ) -> None:
        self.data_type = data_type
        super().__init__(root, split, download, replace_if_exists, transform, target_transform, **kwargs)
        if data_type == "midi":
            self.file_list = glob.glob(os.path.join(self.root, "*.mid"))
            self.pipeline = Pipeline(type="midi")
        elif data_type == "tokenized_ABC":
            self.data_file = os.path.join(self.root, "train/data_v2.txt")
            self.pipeline = Pipeline(type="tok_ABC")

        self.data = []  # type: ignore[assignment]
        self.targets = []  # type: ignore[assignment]

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
        #     target = self.target_transform(target)9

        return midi_data  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.data)

    def download(self) -> None:
        if self.data_type == "tokenized_ABC":
            self.url = "https://raw.githubusercontent.com/IraKorshunova/folk-rnn/master/data/data_v2"

            dest_path = os.path.join(
                self.root,
                "train",
            )
            file_path = os.path.join(dest_path, "data_v2.txt")
            if not self.replace_if_exists and os.path.exists(file_path):
                print("Dataset already exists. Skipping download.")
                return
            response = requests.get(self.url)

            if response.status_code == 200:
                os.makedirs(dest_path, exist_ok=True)
                with open(file_path, "wb") as file:
                    file.write(response.content)
