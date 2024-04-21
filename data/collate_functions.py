from typing import Callable
import torch
from torch.nn.utils.rnn import pad_sequence


def get_collate_fn(name: str) -> Callable | None:
    name2func = {
        "pad_collate_single_sequence": pad_collate_single_sequence,
    }
    return name2func.get(name, None)


def pad_collate_single_sequence(batch, pad_value=0):
    x_lens = torch.tensor([len(x) for x in batch])
    xx_pad = pad_sequence(batch, batch_first=True, padding_value=pad_value)
    return xx_pad, x_lens
