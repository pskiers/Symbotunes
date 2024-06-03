import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.optim.lr_scheduler import ExponentialLR
from ..base import BaseModel


class Encoder(nn.Module):
    def __init__(
        self, input_size: int, z_size: int, hidden_size: int = 256, num_layers: int = 1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.linear_out = nn.Linear(in_features=hidden_size * 2, out_features=z_size * 2)

    def reparametrisation_trick(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.normal(mean=torch.zeros_like(mu), std=torch.ones_like(log_var))
        sigma = torch.exp(log_var * 0.5)
        return mu + eps * sigma

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape0 = 2 * self.num_layers
        h0 = torch.zeros((shape0, x.shape[0], self.hidden_size), device=x.device)
        c0 = torch.zeros((shape0, x.shape[0], self.hidden_size), device=x.device)
        _, (out, _) = self.lstm(x, (h0, c0))  # we want the last state from both directions
        out = out.permute(1, 0, 2).reshape(
            out.shape[1], out.shape[0] * out.shape[2]
        )  # reshape the forward and backward directions into one tensor: shape (2, batch, hidden) -> (batch, hidden * 2)
        out = self.linear_out(out)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        log_var = nn.functional.softplus(log_var)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        z_size: int,
        conductor_in: int,
        conductor_hidden: int,
        lstm_hidden: int,
        conductor_num_layers: int = 1,
        lstm_num_layers: int = 1,
        num_subsequences: int = 16,
        notes_per_subsequence: int = 16,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_subsequences = num_subsequences
        self.n_per_subseq = notes_per_subsequence
        self.total_notes = num_subsequences * notes_per_subsequence
        self.num_tokens = num_tokens
        self.z_size = z_size
        self.conductor_in = conductor_in
        self.conductor_hidden = conductor_hidden
        self.lstm_hidden = lstm_hidden
        self.conductor_num_layers = conductor_num_layers
        self.lstm_num_layers = lstm_num_layers
        self.linear_in = nn.Linear(in_features=z_size, out_features=conductor_hidden * conductor_num_layers * 2)
        self.conductor = nn.LSTM(
            input_size=conductor_in,
            hidden_size=conductor_hidden,
            num_layers=conductor_num_layers,
            batch_first=True,
        )
        self.linear_pre_decoder = nn.Linear(
            in_features=conductor_hidden, out_features=lstm_hidden * lstm_num_layers * 2
        )
        self.decoder_lstm = nn.LSTM(
            input_size=lstm_hidden * lstm_num_layers * 2 + num_tokens,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.linear_out = nn.Linear(in_features=lstm_hidden, out_features=num_tokens)

    def forward(self, z: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = z.shape[0]
        out = self.linear_in(z)
        out = F.tanh(out)
        out = out.view(out.shape[0], self.conductor_num_layers, -1).permute(1, 0, 2)
        hidden_conductor = torch.chunk(out, chunks=2, dim=-1)
        hidden_conductor = [hidden_conductor[0].contiguous(), hidden_conductor[1].contiguous()]
        conductor_in = torch.zeros(
            batch_size, 1, self.conductor_in, device=z.device
        )  # conductor produces embeddings by processing the encoder state z recursively - the input is always zero
        notes = torch.zeros(batch_size, self.total_notes, self.num_tokens, device=z.device)
        for subsequence in range(self.num_subsequences):
            conductor_out, hidden_conductor = self.conductor(conductor_in, hidden_conductor)
            conductor_out = conductor_out.squeeze(1)
            decoder_in = self.linear_pre_decoder(conductor_out)
            decoder_state = decoder_in.view(decoder_in.shape[0], self.lstm_num_layers, -1).permute(1, 0, 2)
            decoder_h, decoder_c = torch.chunk(decoder_state, chunks=2, dim=-1)
            decoder_h, decoder_c = decoder_h.contiguous(), decoder_c.contiguous()

            if x is None:
                note = torch.zeros(batch_size, self.num_tokens, device=z.device)
                for i in range(self.n_per_subseq):
                    lstm_in = torch.cat([decoder_in, note], dim=-1)
                    lstm_in = lstm_in.unsqueeze(1)

                    out, (decoder_h, decoder_c) = self.decoder_lstm(lstm_in, (decoder_h, decoder_c))
                    out = out.squeeze(1)
                    out = self.linear_out(out)
                    note = F.softmax(out, dim=1)
                    notes[:, subsequence * self.n_per_subseq + i, :] = note
            else:
                decoder_in = decoder_in.unsqueeze(1).repeat(1, 16, 1)

                sequence_x = x[:, subsequence * self.n_per_subseq + 1 : (subsequence + 1) * self.n_per_subseq]
                first_token = torch.zeros(sequence_x.shape[0], 1, sequence_x.shape[2], device=sequence_x.device)
                sequence_x = torch.cat([first_token, sequence_x], dim=1)

                lstm_in = torch.cat([decoder_in, sequence_x], dim=-1)

                out, (decoder_h, decoder_c) = self.decoder_lstm(lstm_in, (decoder_h, decoder_c))

                out = self.linear_out(out)

                notes[:, subsequence * self.n_per_subseq : (subsequence + 1) * self.n_per_subseq] = out

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
        lr_decay: float = 0.9999,
        kl_weight: float = 1.0,
        use_teacher_forcing: bool = True,
        *args: torch.Any,
        **kwargs: torch.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.lr_decay = lr_decay
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

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, lengths = batch
        inputs = inputs.long()

        mask = torch.arange(inputs.size(1), device=lengths.device).expand(len(lengths), inputs.size(1)) < lengths.unsqueeze(1)
        x = nn.functional.one_hot(inputs, num_classes=self.decoder.num_tokens).float()

        out, mu, sigma = self(x)

        out = out.reshape(-1, self.decoder.num_tokens)
        targets = inputs.view(-1)
        mask = mask.reshape(-1)
        
        out = out[mask]
        targets = targets[mask]

        recon_loss = F.cross_entropy(out, targets)

        n_mu = torch.tensor([0], device=self.device)
        n_sigma = torch.tensor([1], device=self.device)
        p = Normal(n_mu, n_sigma)
        q = Normal(mu, sigma)
        kl_div = kl_divergence(q, p).mean()

        elbo = torch.mean(recon_loss) + (self.kl_weight * kl_div)
        return elbo, recon_loss, kl_div

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_div = self._step(batch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_recreation", recon_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss_kl_divergence", kl_div, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_div = self._step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/loss_recreation", recon_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val/loss_kl_divergence", kl_div, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_decay)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.no_grad()
    def sample(self, batch_size: int) -> list[torch.Tensor]:
        batch = torch.randn(batch_size, self.decoder.z_size).to(self.device)
        samples = self.decoder(batch)
        return samples
