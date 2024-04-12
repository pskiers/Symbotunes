import pytorch_lightning as pl
from .folk_rnn import FolkRNN


def get_model(name: str, kwargs: dict) -> pl.LightningModule:
    match name:
        case "folk-rnn":
            return FolkRNN(**kwargs)
        case _:
            raise NotImplementedError(f"Model {name} is not available")
