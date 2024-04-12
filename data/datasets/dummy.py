from typing import Callable
from datasets.base import BaseDataset


class Dummy(BaseDataset):
    def __init__(self, transform: Callable):
        super().__init__(transform=transform)

    def download(self) -> None:
        with open('datasets/data_v2', 'r') as f:
            self.raw_data = f.read()
