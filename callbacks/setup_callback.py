import os
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf


class SetupCallback(Callback):
    def __init__(
        self,
        nowname: str,
        config: OmegaConf,
        lightning_config: OmegaConf,
        dl_config: OmegaConf,
        resume: bool = False,
        logdir: str = "logs",
    ):
        super().__init__()
        self.resume = resume
        self.nowname = nowname
        self.logdir = os.path.join(logdir, nowname)
        self.ckptdir = os.path.join(self.logdir, "checkpoints")
        self.cfgdir = os.path.join(self.logdir, "configs")
        self.config = config
        self.lightning_config = lightning_config
        self.dl_config = dl_config

    def on_exception(self, trainer, pl_module, exception):
        match exception:
            case KeyboardInterrupt():
                if trainer.global_rank == 0:
                    print("Summoning checkpoint.")
                    ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                    trainer.save_checkpoint(ckpt_path)
            case _:
                raise exception

    def setup(self, trainer, pl_module, stage):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "project.yaml"),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.dl_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.dl_config}),
                os.path.join(self.cfgdir, "lightning.yaml"),
            )
            print("Dataloading config")
            print(OmegaConf.to_yaml(self.dl_config))
            OmegaConf.save(
                OmegaConf.create({"dataloading": self.dl_config}),
                os.path.join(self.cfgdir, "dataloading.yaml"),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
