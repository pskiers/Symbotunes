from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule, ABC):
    """Base model for the repo"""

    @abstractmethod
    def sample(self, batch_size: int) -> list[torch.Tensor] | torch.Tensor:
        """Generate new samples, by sampling from model"""
