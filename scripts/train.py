from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
from os import environ
from pathlib import Path
import datetime

from data import get_dataloaders
from models import get_model
from callbacks import get_callbacks


if __name__ == "__main__":
    environ["WANDB__SERVICE_WAIT"] = "300"

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
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
    lightning_config = config.pop("lightning", OmegaConf.create())

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_opt = OmegaConf.to_container(trainer_config)
    lightning_config.trainer = trainer_config

    dl_config_orig = config.pop("dataloaders")
    dl_config = OmegaConf.to_container(dl_config_orig, resolve=True)
    train_dls, test_dl = get_dataloaders(dl_config)

    if checkpoint_path is not None:
        config.model.params["ckpt_path"] = checkpoint_path

    model = get_model(config.model.get("model_type"), config.model.get("params", dict()))

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = model.__class__.__name__ + "_" + now

    trainer_kwargs = dict()

    tags = []
    trainer_kwargs["logger"] = pl.loggers.WandbLogger(name=nowname, id=nowname, tags=tags)

    callback_cfg = config.get("callbacks", OmegaConf.create())
    trainer_kwargs["callbacks"] = get_callbacks(config.callbacks)

    trainer = pl.Trainer(**trainer_opt, **trainer_kwargs)

    trainer.fit(
        model,
        train_dataloaders=train_dls,
        val_dataloaders=test_dl,
    )
