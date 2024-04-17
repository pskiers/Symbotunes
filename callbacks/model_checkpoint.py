import os
from datetime import timedelta
from typing import Literal
from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointAdjusted(ModelCheckpoint):
    def __init__(
        self,
        nowname: str,
        logdir: str = "logs",
        filename: str | None = None,
        monitor: str | None = None,
        verbose: bool = False,
        save_last: None | bool | Literal["link"] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            os.path.join(logdir, nowname, "checkpoints"),
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            enable_version_counter,
        )
