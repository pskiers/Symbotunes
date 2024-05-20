import os
import tarfile
import requests
from typing import Callable

from .Pipeline.Pipeline import Pipeline
from base import BaseDataset


class LakhMidiDataset(BaseDataset):
    """
    LakhMIDI Dataset class
    available on: https://colinraffel.com/projects/lmd/#get
    """

    def __init__(
        self,
        root: str = "_data",
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        preload: bool = True,
        download: bool = True,
        **kwargs
    ) -> None:
        super().__init__(root, split, download, transform, target_transform, **kwargs)

        self.pipeline = Pipeline(type="midi_path")

        self.data = []  # type: ignore[assignment]
        self.targets = []  # type: ignore[assignment]

        if preload:
            self._load_data()

    def _load_data(self):
        # self._targets = []
        self._data = self.pipeline.process(self.root)
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
        # Temporary substitution, so that we don't download 1.7 GB of midi each time.
        # self.url = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
        self.url = "https://drive.google.com/uc?export=download&id=1aV4rNwtb3b8f55bxmoTOmqc0zBPJ1MIp"

        response = requests.get(self.url)

        if response.status_code == 200:
            dest_path = os.path.join(
                self.root,
                "train",
            )
            os.makedirs(dest_path, exist_ok=True)
            tarball_path = os.path.join(dest_path, "lakh.tar.gz")
            with open(tarball_path, "wb") as file:
                file.write(response.content)
            with tarfile.open(tarball_path, "r:*") as file:
                file.extractall(dest_path)
            os.remove(tarball_path)