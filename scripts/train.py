from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from os import environ
from pathlib import Path
import datetime

from data import get_dataloaders
from models import get_model
from callbacks import get_callbacks


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=Path, required=True, help="path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=False,
        help="path to model checkpoint file",
    )
    args = parser.parse_args()
    config_path = str(args.path)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None

    config = OmegaConf.load(config_path)
    lightning_config = config.pop("lightning", OmegaConf.create())  # type: ignore[call-arg, arg-type]

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_kwargs = OmegaConf.to_container(trainer_config)
    lightning_config.trainer = trainer_config

    dl_config_orig = config.pop("dataloaders")  # type: ignore[arg-type]
    dl_config = OmegaConf.to_container(dl_config_orig, resolve=True)
    train_dls, test_dl = get_dataloaders(dl_config)  # type: ignore[arg-type]

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path

    model = get_model(
        config.model.get("model_type"), config.model.get("params", dict())
    )

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now
    common_callbacks_kwargs = {
        "nowname": nowname,
        "config": config,
        "lightning_config": lightning_config,
        "dl_config": dl_config,
    }

    tags = []  # type: ignore[var-annotated]
    trainer_kwargs["logger"] = WandbLogger(  # type: ignore
        name=nowname, id=nowname, tags=tags, project="Symbotunes"
    )
    callback_cfg = config.get("callbacks", OmegaConf.create())  # type: ignore[arg-type]
    trainer_kwargs["callbacks"] = get_callbacks(config.callbacks, common_callbacks_kwargs)  # type: ignore

    trainer = pl.Trainer(**trainer_kwargs)  # type: ignore[arg-type]

    trainer.fit(
        model,
        train_dataloaders=train_dls,
        val_dataloaders=test_dl,
    )
