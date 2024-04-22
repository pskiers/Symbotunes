from typing import Type
from .base import BaseModel
from .folk_rnn import FolkRNN


def get_model(name: str) -> Type[BaseModel]:
    match name:
        case "folk-rnn":
            return FolkRNN
        case _:
            raise NotImplementedError(f"Model {name} is not available")
