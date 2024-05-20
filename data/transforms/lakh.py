# from random import randint # TODO should the transform sample the midi sequence at random?
from miditok import MIDILike, TokenizerConfig


TOKENIZER_PARAMS = {
    "vocab_size": 130 + 512,  # Total vocab size: 130 for pitches, note-off, rest + 512 for drum patterns
    "max_bar_length": 16,  # Each bar has 16 events (16th notes)
    "num_bars": 16,  # Total number of bars for hierarchical model
    "hierarchical": True,  # Use hierarchical modeling
    "bar_token_count": 16,  # Each subsequence corresponds to a single bar
    "beat_resolution": 4,  # 16th note intervals (4 intervals per beat in 4/4 time)
    "note_on_count": 128,  # 128 note-on tokens
    "use_rests": True,
    "drum_pattern_count": 512  # 512 categorical tokens for drum patterns
}


class LakhTransform(object):
    def __init__(self, sequence_size: int | None = None) -> None:
        self.tokenizer_params = TOKENIZER_PARAMS

    def __call__(self, path: str):
        config = TokenizerConfig(**self.tokenizer_params)
        tokenizer = MIDILike(config)
        tokens = tokenizer(path)

        return tokens
