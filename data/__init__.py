from torch.utils.data import DataLoader

from .datasets import get_dataset
from .tokenizers import get_tokenizer
from .transforms import *


def get_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader | list[DataLoader]]:
    """Get dataloaders from config. Returns val dataloader and train dataloader(s)"""
    # TODO
    raise NotImplementedError
