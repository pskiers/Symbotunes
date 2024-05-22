# from random import randint # TODO should the transform sample the midi sequence at random?
from miditok import REMI, TokenizerConfig, TokSequence
from random import randint
from typing import List

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

    def _get_bars_from_tokenized_midi(self, midi: List[TokSequence]):
        """
        Tokenized MIDI contains multiple tracks, each with a TokSequence containing multiple tokens.
        Each track is separated by bars and analyzed. Only the best track is returned, which is determined by:
        - absence of drums
        - length of nonempty bars
        """
        best_bars = None
        for tok_seq in midi:
            bars: List[List[str]] = []
            current_bar: List[str] = []
            number_of_nonempty_bars = 0
            is_drum = False
            for tok in tok_seq.tokens:
                if "Drum" in tok:
                    is_drum = True
                    break
                if "Bar_" in tok:
                    if current_bar:
                        number_of_nonempty_bars += 0
                        bars.append(current_bar)
                        current_bar = []
                else:
                    current_bar.append(tok)
            if current_bar:
                number_of_nonempty_bars += 0
                bars.append(current_bar)
            if is_drum:
                # In the current implementation only melodic sequences are accepted
                continue
            if best_bars is None or len(bars) > len(best_bars):
                best_bars = bars

        return best_bars

    def __call__(self, path: str, number_of_bars: int = 16):
        tokenized_midi = self.tokenizer(path)
        bars = self._get_bars_from_tokenized_midi(tokenized_midi)
        if bars is None or len(bars) < number_of_bars:
            return None
        start_idx = randint(0, len(bars) - number_of_bars - 1)
        end_idx = start_idx + number_of_bars
        sampled_bars = bars[start_idx:end_idx]
        tokens = [x for xs in sampled_bars for x in xs]  # Flatten the bars

        return tokens
