from tokenizers.tokenizer import Tokenizer
from random import randint


class Transform():
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __call__(self, data: str):
        t = Tokenizer()
        tunes = data.split('\n\n')
        transformed_tunes = [t(tune)for tune in tunes]
        result = []
        for tune in transformed_tunes:
            start_idx = randint(0, len(tune) - self.batch_size)
            end_idx = start_idx + self.batch_size
            print(start_idx, end_idx)
            result.append(tune[start_idx:end_idx])
        return result
