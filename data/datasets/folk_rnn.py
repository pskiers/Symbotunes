import os
import requests
from typing import Callable
import numpy as np

from .base import BaseDataset


class FolkRnnDataset(BaseDataset):
    """
    Folk-Rnn Dataset class available on: https://github.com/IraKorshunova/folk-rnn
    """

    def __init__(
        self,
        root: str = "_data",
        split: str = "train",
        data_type: str = "tokenized_ABC",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = True,
        replace_if_exists: bool = True,
        **kwargs,
    ) -> None:
        self.data_type = data_type
        super().__init__(root, split, download, replace_if_exists, transform, target_transform, **kwargs)
        # NOTE maybe add midi version of the dataset
        # if data_type == "midi":
        #     self.file_list = glob.glob(os.path.join(os.path.join(self.root, "session_test"),"*.mid"))
        if data_type == "tokenized_ABC":
            self.data_file = os.path.join(self.root, "train", "data_v2.txt")
            with open(self.data_file, "r", encoding="utf-8") as file:
                full_data = np.array(file.read().split("\n\n"))

            self.data = self.split_data(full_data)

        else:
            raise ValueError(f"{data_type} is not an allowed data type for this dataset")

    def download(self) -> None:
        if self.data_type == "tokenized_ABC":
            dest_path = os.path.join(self.root, "train")
            if os.path.exists(os.path.join(dest_path, "data_v2.txt")):
                return

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
            else:
                raise requests.HTTPError("Unable to download the dataset")
