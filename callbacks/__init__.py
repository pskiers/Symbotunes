from pytorch_lightning.callbacks import Callback

from .checkpoint_every_n_callback import CheckpointEveryNSteps
from .cuda_callback import CUDACallback
from .pl_callbacks import get_pl_callback
from .setup_callback import SetupCallback
from .model_checkpoint import ModelCheckpointAdjusted


def _get_callback(common: dict, name: str, kwargs: dict | None) -> Callback:
    kwargs = kwargs if kwargs is not None else dict()
    match name:
        case "checkpoint_every_n_steps":
            return CheckpointEveryNSteps(**kwargs)
        case "cuda_callback":
            return CUDACallback(**kwargs)
        case "setup_callback":
            return SetupCallback(**kwargs, **common)
        case "model_checkpoint":
            return ModelCheckpointAdjusted(**kwargs, **common)
        case _:
            return get_pl_callback(name, kwargs)


def get_callbacks(config: list, common: dict) -> list[Callback]:
    return [_get_callback(common, *next(iter(callback.items()))) for callback in config]
