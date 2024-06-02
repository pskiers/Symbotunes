from .base import BaseDataset
from .folk_rnn import FolkRnnDataset
from .lakh import LakhMidiDataset


def get_dataset(name: str, config: dict) -> BaseDataset:
    config = config if config is not None else dict()
    match name:
        case "folk-rnn":
            return FolkRnnDataset(**config)
        case "lakh":
            return LakhMidiDataset(**config)
        case _:
            raise NotImplementedError()
