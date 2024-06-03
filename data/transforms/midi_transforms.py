# from random import randint # TODO should the transform sample the midi sequence at random?
from miditok import REMI, TokenizerConfig, TokSequence
from random import randint
import torch

TOKENIZER_PARAMS = {
    "pitch_range": (0, 127),
    "vocab_size": 130,  # Total vocab size: 130 for pitches, note-off, rest + 512 for drum patterns
    "max_bar_length": 16,  # Each bar has 16 events (16th notes)
    "num_bars": 16,  # Total number of bars for hierarchical model
    "hierarchical": True,  # Use hierarchical modeling
    "bar_token_count": 16,  # Each subsequence corresponds to a single bar
    "beat_resolution": 4,  # 16th note intervals (4 intervals per beat in 4/4 time)
    "note_on_count": 128,  # 128 note-on tokens
    # "use_rests": True,
    # "drum_pattern_count": 512  # 512 categorical tokens for drum patterns
}


class MidiTokenizer(object):
    def __init__(self, tokenizer_params: dict, max_tracks: int = 1) -> None:
        self.config = TokenizerConfig(**tokenizer_params)
        self.tokenizer = REMI(self.config)  # TODO handle other tokenizers
        self.max_tracks = max_tracks

    def __call__(self, path: str):
        tokenized_midi = self.tokenizer(path)[:self.max_tracks]
        return tokenized_midi[0] if len(tokenized_midi) == 1 else tokenized_midi


class MusicVAETokenizer(MidiTokenizer):
    def __init__(self) -> None:
        super().__init__(TOKENIZER_PARAMS)


class SampleBars(object):
    """Sample n bars from midi file or less if midi does not have enough bars"""

    def __init__(self, number_of_bars: int) -> None:
        self.num_bars = number_of_bars

    def __call__(self, tokens: TokSequence) -> TokSequence:
        token_names = tokens.tokens
        bar_positions = [i for i, token in enumerate(token_names) if token.startswith("Bar_")]

        if len(bar_positions) < self.num_bars:
            return tokens

        max_start_index = len(bar_positions) - self.num_bars
        start_index = randint(0, max_start_index)

        start_pos = bar_positions[start_index]
        end_pos = (
            bar_positions[start_index + self.num_bars]
            if (start_index + self.num_bars) < len(bar_positions)
            else len(token_names)
        )

        sampled_tokens = tokens[start_pos:end_pos]
        return sampled_tokens


class TokSequenceToTensor(object):
    def __call__(self, tokens: TokSequence) -> torch.Tensor:
        return torch.tensor(tokens.ids)
