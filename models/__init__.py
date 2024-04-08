import pytorch_lightning as pl


def get_model(name: str, kwargs: dict) -> pl.LightningModule:
    match name:
        case _:
            raise NotImplementedError(f"Model {name} is not available")