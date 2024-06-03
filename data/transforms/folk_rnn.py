from ..tokenizers import FolkTokenizer
from random import randint


class FolkTransform(object):
    def __call__(self, data: str):
        t = FolkTokenizer()
        return t(data)
