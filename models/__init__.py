from typing import Type
from .base import BaseModel
from .folk_rnn import FolkRNN
from .music_vae import MusicVae
from .gpt import GPT2


def get_model(name: str) -> Type[BaseModel]:
    match name:
        case "folk-rnn":
            return FolkRNN
        case "music-vae":
            return MusicVae
        case "gpt2":
            return GPT2
        case _:
            raise NotImplementedError(f"Model {name} is not available")
