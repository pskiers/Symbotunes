import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from ..base import BaseModel


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        z_size: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.linear_out = nn.Linear(in_features=hidden_size * 2, out_features=z_size * 2)

    @torch.no_grad()
    def reparametrisation_trick(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.normal(mean=torch.zeros_like(mu), std=torch.ones_like(log_var))
        sigma = torch.exp(log_var * 0.5)
        return mu + eps * sigma

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape0 = (1 + self.bidirectional) * self.num_layers
        h0 = torch.zeros((shape0, x.shape[0], self.hidden_size), device=x.device)
        c0 = torch.zeros((shape0, x.shape[0], self.hidden_size), device=x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.linear_out(out)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        log_var = nn.functional.softplus(log_var)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        z_size: int,
        linear_in_out_size: int,
        conductor_hidden: int,
        rnn_hidden_size: int,
        num_levels: int = 16,
        notes_per_bar: int = 16,
        conductor_num_layers: int = 1,
        rnn_num_layers: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_levels = num_levels
        self.notes_per_bar = notes_per_bar
        self.total_notes = num_levels * notes_per_bar
        self.num_tokens = num_tokens
        self.z_size = z_size
        self.linear_in_out_size = linear_in_out_size
        self.conductor_hidden = conductor_hidden
        self.rnn_hidden_size = rnn_hidden_size
        self.conductor_num_layers = conductor_num_layers
        self.rnn_num_layers = rnn_num_layers
        self.linear_in = nn.Linear(in_features=z_size, out_features=linear_in_out_size)
        self.conductor = nn.LSTM(
            input_size=linear_in_out_size,
            hidden_size=conductor_hidden,
            num_layers=conductor_num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.decoder_rnn = nn.LSTM(
            input_size=num_tokens + conductor_hidden,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.linear_out = nn.Linear(in_features=rnn_hidden_size, out_features=num_tokens)

    def forward(self, z: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        out = self.linear_in(z)
        h0 = torch.zeros((self.conductor_num_layers, z.shape[0], self.conductor_hidden), device=z.device)
        c0 = torch.zeros((self.conductor_num_layers, z.shape[0], self.conductor_hidden), device=z.device)

        counter = 0
        notes = torch.zeros(z.shape[0], self.total_notes, self.num_tokens, device=z.device)
        for level in range(self.num_levels):
            emb, (h0, c0) = self.conductor(z[:, self.num_levels * level, :], (h0, c0))

            decoder_h0 = torch.randn((self.conductor_num_layers, z.shape[0], self.conductor_hidden), device=z.device)
            decoder_c0 = torch.randn((self.conductor_num_layers, z.shape[0], self.conductor_hidden), device=z.device)
            if x is None:
                note = torch.zeros(z.shape[0], 1, self.num_tokens, device=z.device)
                for _ in range(self.notes_per_bar):
                    emb = torch.cat([emb, note], dim=-1)
                    emb = emb.view(z.shape[0], 1, -1)

                    note, (decoder_h0, decoder_c0) = self.decoder_rnn(emb, (decoder_h0, decoder_c0))

                    out = self.linear_out(note)
                    out = torch.sigmoid(out)

                    notes[:, counter, :] = out.squeeze()

                    note = out
                    counter = counter + 1
            else:
                emb = emb.expand(z.shape[0], self.notes_per_bar, emb.shape[2])

                e = torch.cat([emb, x[:, range(level * 16, level * 16 + 16), :]], dim=-1)

                notes2, (decoder_h0, decoder_c0) = self.decoder_rnn(e, (decoder_h0, decoder_c0))

                out = self.linear_out(notes2)
                out = torch.softmax(out, dim=2)

                # generates 16 notes per batch at a time
                notes[:, range(level * 16, level * 16 + 16), :] = out

        return notes


class MusicVae(BaseModel):
    """
    Music Vae model based on https://arxiv.org/pdf/1803.05428.
    Original code base: https://github.com/magenta/magenta/blob/main/magenta/models/music_vae
    """

    def __init__(
        self,
        encoder_config: dict,
        decoder_config: dict,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
        use_teacher_forcing: bool = True,
        *args: torch.Any,
        **kwargs: torch.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.kl_weight = kl_weight
        self.use_teacher_forcing = use_teacher_forcing
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        sigma = torch.exp(log_var * 2)
        z = self.encoder.reparametrisation_trick(mu, log_var)
        if self.use_teacher_forcing:
            out = self.decoder(z, x)
        else:
            out = self.decoder(z)
        return out, mu, sigma

    def _step(self, batch) -> torch.Tensor:
        x = nn.functional.one_hot(batch, num_classes=self.decoder.num_tokens).float()
        out, mu, sigma = self(x)
        recon_loss = -binary_cross_entropy(out, batch, reduction="none")
        recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(1)

        n_mu = torch.Tensor([0], device=self.device)
        n_sigma = torch.Tensor([1], device=self.device)

        p = Normal(n_mu, n_sigma)
        q = Normal(mu, sigma)
        kl_div = kl_divergence(q, p)
        elbo = torch.mean(recon_loss) - (self.kl_weight * torch.mean(kl_div))
        return elbo

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = RMSprop(params=self.parameters(), lr=self.lr)
        # scheduler = LambdaLR(
        #     optimizer,
        #     lr_lambda=lambda epoch: (
        #         1
        #         if epoch < self.lr_decay_start
        #         else self.lr_decay ** (epoch - self.lr_decay_start)
        #     ),
        # )
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    # fmt: on

    @torch.no_grad()
    def sample(self, batch_size: int) -> list[torch.Tensor]:
        n_mu = torch.Tensor([0], device=self.device)
        n_sigma = torch.Tensor([1], device=self.device)

        p = Normal(n_mu, n_sigma)
        batch = p.sample((batch_size, self.decoder.z_size)).to(self.device)
        samples = self.decoder(batch)
        return samples
