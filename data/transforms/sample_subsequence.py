from random import randint


class SampleSubsequence(object):
    def __init__(self, subsequence_len: int) -> None:
        self.subsequence_len = subsequence_len

    def __call__(self, sequence):
        if len(sequence) <= self.subsequence_len:
            return sequence
        start_idx = randint(0, len(sequence) - self.subsequence_len - 1)
        end_idx = start_idx + self.subsequence_len
        return sequence[start_idx:end_idx]
