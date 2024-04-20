from typing import Callable
from torchvision import transforms
from .folk_rnn import FolkTransform
from .file_loaders import LoadMIDI


def get_transform(kwargs: dict | list) -> Callable:
    if isinstance(kwargs, list):
        return transforms.Compose([parse_transform(*next(iter(trans.items()))) for trans in kwargs])
    else:
        return transforms.Compose([parse_transform(n, k) for n, k in kwargs.items()])


def parse_transform(name: str, kwargs: dict) -> Callable:
    kwargs = kwargs if kwargs is not None else dict()
    match name:
        case "folk_rnn":
            return FolkTransform(**kwargs)
        case "load_midi":
            return LoadMIDI(**kwargs)
        case _:
            raise NotImplementedError()
