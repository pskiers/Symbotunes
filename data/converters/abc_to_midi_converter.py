from typing import List
from music21 import converter


class ABCTOMidiConverter:
    def __init__(self, tokenizer, resolution=480, tempo=500000):
        self.resolution = resolution
        self.tempo = tempo
        self.encoder = tokenizer

    def _reformat_notes(self, notes: List[str]):
        notes.insert(1, "L: 1/8")
        notes.insert(3, "\n")
        notes.insert(2, "\n")
        notes.insert(1, "\n")
        return notes

    def _convert_abc_to_midi(self, notes_string: str, filename: str = "output.mid"):
        stream = converter.parse(notes_string)
        stream.write("midi", fp=filename)

    def __call__(self, encodings, filename="output.mid"):
        notes = self.encoder.inverse_transform(encodings)

        assert notes[0][0] == "M"
        assert notes[1][0] == "K"

        formatted_notes = self._reformat_notes(notes)
        notes_string = " ".join(formatted_notes)
        self._convert_abc_to_midi(notes_string, filename)
