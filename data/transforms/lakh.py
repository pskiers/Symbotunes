# from random import randint # TODO should the transform sample the midi sequence at random?
from miditok import REMI, TokenizerConfig, TokSequence
from random import randint

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


class LakhTransform(object):
    def __init__(self) -> None:
        self.tokenizer_params = TOKENIZER_PARAMS
        self.config = TokenizerConfig(**self.tokenizer_params)
        self.tokenizer = REMI(self.config)

    def _sample_bars(self, tokens: TokSequence, num_bars):
        tokens = tokens.tokens
        bar_positions = [i for i, token in enumerate(tokens) if token.startswith("Bar_")]

        if len(bar_positions) < num_bars:
            return None

        max_start_index = len(bar_positions) - num_bars
        start_index = randint(0, max_start_index)

        start_pos = bar_positions[start_index]
        end_pos = (
            bar_positions[start_index + num_bars] if (start_index + num_bars) < len(bar_positions) else len(tokens)
        )

        sampled_tokens = tokens[start_pos:end_pos]
        return sampled_tokens

    def __call__(self, path: str, number_of_bars: int = 16):
        tokenized_midi = self.tokenizer(path)[0]
        tokens = self._sample_bars(tokenized_midi, number_of_bars)
        int_tokens = [self.tokenizer.vocab[t] for t in tokens]
        return int_tokens
