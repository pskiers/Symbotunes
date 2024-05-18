import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..base import BaseModel

def default_hparams():
    return {
        'n_vocab': 0,
        'n_ctx': 1024,
        'n_embd': 768,
        'n_head': 12,
        'n_layer': 12,
    }

def gelu(x): # DONE
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Norm(nn.Module): # DONE +-
    def __init__(self, n_state, epsilon=1e-5):
        super(Norm, self).__init__()
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.ones(n_state)) # weight
        self.b = nn.Parameter(torch.zeros(n_state)) # bias

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.epsilon)
        return x * self.g + self.b

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        # self.nx = nx #
        # self.w = nn.Parameter(torch.randn(nx, nf) * 0.02)
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(nf))

    def forward(self, x): # DONE +-
        # start = x.size()[:-1] + (self.nf,)
        # c = torch.matmul(
        #     x.view(-1, self.nx),
        #     self.w.view(-1, self.nf)
        # ) + self.b
        # c = c.view(*start, self.nf)
        # return c

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
        x = x.view(*size_out)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, nx, n_ctx, n_head, scale=False):
        super(MultiheadAttention, self).__init__()
        n_state = nx
        assert n_state % n_head == 0
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        # self.attn_dropout = nn.Dropout(0.1) # TODO change it
        # self.resid_dropout = nn.Dropout(0.1) # TODO change it or delete
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def split_heads(self, x, k=False): # DONE
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def merge_heads(self, x): # DONE
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def forward(self, x, layer_past=None, attn_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        # present = torch.stack((key, value))
        present = torch.stack((key.transpose(-2, -1), value))

        attn_output = self._attn(query, key, value, attn_mask)
        attn_output = self.merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        # attn_output = self.resid_dropout(attn_output)
        return attn_output, present

    def _attn(self, q, k, v, attn_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (v.size(-1)) # (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns] # self.bias???
        w = w * b - 1e10 * (1 - b)
        if attn_mask is not None: # check if needed
            w = w + attn_mask
        w = F.softmax(w, dim=-1)
        # w = self.attn_dropout(w) # check if needed
        a = torch.matmul(w, v)
        return a
        # return a, w # sprawdź czy nie tylko return a

class MLP(nn.Module):
    def __init__(self, n_state, n_embd):
        super(MLP, self).__init__()
        nx = n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        # self.dropout = nn.Dropout(0.1) # TODO change it or delete

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2
        # return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, n_ctx, n_embd, n_head, scale=False):
        super(Block, self).__init__()
        nx = n_embd
        self.attn = MultiheadAttention(nx, n_ctx, n_head, scale)
        self.ln_1 = Norm(nx)
        self.mlp = MLP(4 * nx, n_embd)
        self.ln_2 = Norm(nx)

    def forward(self, x, layer_past=None, attn_mask=None):
        nx = x.shape[-1]
        a, present = self.attn(self.ln_1(x), layer_past, attn_mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2(BaseModel):
    def __init__(
        self, 
        n_vocab,
        n_ctx,
        n_embd,
        n_head,
        n_layer,
        lr: float = 0.003,
        lr_decay: float = 0.97,
        lr_decay_start: int = 20,
        *args: torch.Any,
        **kwargs: torch.Any
    ) -> None:
        super(GPT2, self).__init__()
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_start = lr_decay_start

        self.wpe = nn.Embedding(self.n_ctx, self.n_embd)
        self.wte = nn.Embedding(self.n_vocab, self.n_embd)
        block = Block(self.n_ctx, self.n_embd, self.n_head, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(self.n_layer)])
        # self.h = nn.ModuleList([Block(self.n_ctx, self.n_embd, self.n_head, scale=True) for _ in range(self.n_layer)])
        self.ln_f = Norm(self.n_embd)
        n_hidden = 2*n_vocab
        self.ll = nn.Linear(n_embd, n_hidden)
        self.ll2 = nn.Linear(n_hidden, n_vocab)

    def set_embeddings_weights(self, model_embeddings_weights): # ?
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            # position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = input_embeds + position_embeds + token_type_embeds

        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(*output_shape)

        out = F.leaky_relu(self.ll(hidden_states))
        out = self.ll2(out)

        return out, presents

# --- 
    def _step(self, batch) -> torch.Tensor:
        x, lengths = batch
        y = x[:, 1:]
        mask = (torch.arange(x.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        mask = mask[:, 1:]
        out, _ = self(x)
        # corresponding_outs = out.gather(dim=2, index=y.unsqueeze(-1)).squeeze(-1)
        # log_probs = torch.log(corresponding_outs)
        d1, d2, d3 = out.shape
        out = out[:, :-1, :].reshape(d1*(d2-1), d3)
        y = y.flatten()
        loss = F.cross_entropy(out, y, reduction='none')
        loss = loss.view(d1, d2-1)
        loss = loss*mask
        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        # loss = F.cross_entropy(logits.view(-1, self.n_vocab), labels.view(-1))
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @torch.no_grad()
    def sample(self, batch_size: int, length: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        input_ids = torch.tensor([[self.start_token] * batch_size], device=self.device).t()
        position_ids = torch.arange(0, length, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        output = input_ids

        past = None
        for i in range(length):
            logits, past = self(input_ids, position_ids[:, :i + 1], past=past)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            output = torch.cat([output, next_token], dim=-1)
            input_ids = next_token

        return output