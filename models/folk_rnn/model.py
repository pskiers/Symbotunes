import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..base import BaseModel


class FolkRNN(BaseModel):
    """LSTM-based music generation model, see https://arxiv.org/pdf/1604.08723.pdf"""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        lstm_size: int,
        dropout: float,
        embedding_size: int | None = None,
        lr: float = 0.003,
        lr_decay: float = 0.97,
        lr_decay_start: int = 20,
        start_token: int = 135,
        end_token: int = 136,
        *args: torch.Any,
        **kwargs: torch.Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lstm_size = lstm_size
        self.num_layers = num_layers

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_start = lr_decay_start

        self.start_token = start_token
        self.end_token = end_token

        one_hot = embedding_size is None
        if one_hot:
            # If one-hot encoding is used, initialize weights as an identity matrix
            self.weights_emb = nn.Parameter(torch.eye(vocab_size, dtype=torch.float32))
            weights_emb = torch.eye(vocab_size, dtype=torch.float32)
        else:
            # Otherwise, initialize weights using Orthogonal initializer
            weights_emb = nn.init.orthogonal_(torch.empty(vocab_size, embedding_size))  # type: ignore
        self.embedding = nn.Embedding.from_pretrained(weights_emb, freeze=True)

        rnn_input_size = vocab_size if one_hot else embedding_size
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.out = nn.Linear(in_features=lstm_size, out_features=vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb_packed = pack_padded_sequence(emb, lengths, enforce_sorted=False, batch_first=True)
        hidden = torch.zeros((self.num_layers, x.shape[0], self.lstm_size), device=self.device)
        state = torch.zeros((self.num_layers, x.shape[0], self.lstm_size), device=self.device)
        out, _ = self.lstm(emb_packed, (hidden, state))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        out = self.out(out)
        return F.softmax(out, dim=2)

    def _step(self, batch) -> torch.Tensor:
        x, lengths = batch
        y = x[:, 1:]
        mask = (torch.arange(x.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        out = self(x, lengths.cpu())
        corresponding_outs = out.gather(dim=2, index=y.unsqueeze(-1)).squeeze(-1)
        log_probs = torch.log(corresponding_outs)
        loss = -torch.mean(torch.sum(log_probs * mask[:, :-1], dim=1), dim=0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    # fmt: off
    def configure_optimizers(self):
        optimizer = RMSprop(params=self.parameters(), lr=self.lr)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (
                1
                if epoch < self.lr_decay_start
                else self.lr_decay ** (epoch - self.lr_decay_start)
            ),
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    # fmt: on

    @torch.no_grad()
    def sample(self, batch_size: int) -> list[torch.Tensor]:
        batch = torch.tensor([[self.start_token] for _ in range(batch_size)], device=self.device)
        lengths = torch.tensor([1 for _ in range(batch_size)], device=torch.device("cpu"))
        samples: list[torch.Tensor] = []
        while batch.shape[0] > 0:
            out = self(batch, lengths)
            next_tokens = out[:, -1].argmax(dim=1).unsqueeze(1)
            batch = torch.concat(tensors=(batch, next_tokens), dim=1)
            ended = batch[:, -1] == self.end_token
            samples += [sample for sample in batch[ended]]
            batch = batch[~ended]
            lengths += 1
            lengths = lengths[~ended.cpu()]
        return samples
