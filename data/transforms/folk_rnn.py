from ..tokenizers import FolkTokenizer
from random import randint


class FolkTransform(object):
    def __init__(self, sequence_size: int) -> None:
        self.sequence_size = sequence_size

    def __call__(self, data: str):
        t = FolkTokenizer()
        tune = t(data)
        start_idx = randint(0, len(tune) - self.sequence_size - 1)
        end_idx = start_idx + self.sequence_size
        return tune[start_idx:end_idx]
