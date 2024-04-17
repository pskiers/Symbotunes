from typing import List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from music21 import converter


class OneHotToMidiConverter:
    def __init__(self, tokens, resolution=480, tempo=500000):
        self.resolution = resolution
        self.tempo = tempo
        self.encoder = OneHotEncoder(categories=[tokens], sparse_output=False)
        reshaped_tokens = np.array(tokens).reshape(-1, 1)
        self.encoder.fit(reshaped_tokens)

    def _convert_ohc_to_str_list(self, input_notes):
        track_notes = self.encoder.inverse_transform(input_notes)
        return [item for sublist in track_notes for item in sublist]

    def _reformat_notes(self, notes: List[str]):
        notes.insert(1, 'L: 1/8')
        notes.insert(3, '\n')
        notes.insert(2, '\n')
        notes.insert(1, '\n')
        return notes

    def _convert_abc_to_midi(self, notes_string: str, filename: str = 'output.mid'):
        stream = converter.parse(notes_string)
        stream.write('midi', fp=filename)

    def __call__(self, encodings, filename='output.mid'):
        notes = self._convert_ohc_to_str_list(encodings)

        assert notes[0][0] == 'M'
        assert notes[1][0] == 'K'

        formatted_notes = self._reformat_notes(notes)
        notes_string = ' '.join(formatted_notes)
        self._convert_abc_to_midi(notes_string, filename)
