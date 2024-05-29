from typing import Type
from .base import BaseModel
from .folk_rnn import FolkRNN
from .music_vae import MusicVae


def get_model(name: str) -> Type[BaseModel]:
    match name:
        case "folk-rnn":
            return FolkRNN
        case "music-vae":
            return MusicVae
        case _:
            raise NotImplementedError(f"Model {name} is not available")
