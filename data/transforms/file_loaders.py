from pretty_midi import PrettyMIDI


class LoadMIDI(object):
    def __init__(self, resolution: int = 220, initial_tempo: int = 120) -> None:
        self.resolution = resolution
        self.initial_tempo = initial_tempo

    def __call__(self, data: str):
        return PrettyMIDI(midi_file=data, resolution=self.resolution, initial_tempo=self.initial_tempo)
