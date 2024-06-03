from typing import Callable
from torchvision import transforms  # type: ignore[import]
from .folk_rnn import FolkTransform
from .file_loaders import LoadMIDI
from .midi_transforms import MidiTokenizer, MusicVAETokenizer, SampleBars, TokSequenceToTensor
from .sample_subsequence import SampleSubsequence


# fmt: off
def get_transform(kwargs: dict | list) -> Callable:
    if isinstance(kwargs, list):
        return transforms.Compose([parse_transform(*next(iter(trans.items()))) for trans in kwargs])
    else:
        return transforms.Compose([parse_transform(n, k) for n, k in kwargs.items()])
# fmt: on


def parse_transform(name: str, kwargs: dict) -> Callable:
    kwargs = kwargs if kwargs is not None else dict()
    match name:
        case "folk_rnn":
            return FolkTransform(**kwargs)
        case "load_midi":
            return LoadMIDI(**kwargs)
        case "midi_tokenizer":
            return MidiTokenizer(**kwargs)
        case "music_vae_tokenizer":
            return MusicVAETokenizer()
        case "sample_bars":
            return SampleBars(**kwargs)
        case "toksequence_to_tensor":
            return TokSequenceToTensor()
        case "sample_subsequence":
            return SampleSubsequence(**kwargs)
        case _:
            raise NotImplementedError()
