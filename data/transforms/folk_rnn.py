from ..tokenizers import FolkTokenizer
from random import randint


class FolkTransform(object):
    def __init__(self, sequence_size: int | None = None) -> None:
        self.sequence_size = sequence_size

    def __call__(self, data: str):
        t = FolkTokenizer()
        tune = t(data)
        if self.sequence_size is not None:
            start_idx = randint(0, len(tune) - self.sequence_size - 1)
            end_idx = start_idx + self.sequence_size
            tune = tune[start_idx:end_idx]
        return tune
