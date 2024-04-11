from pretty_midi import PrettyMIDI

class Pipeline():
    def __init__(self,
                 type):
        self.pre_pipeline = []
        self.post_pipeline = []

        self.type = type
        {
            "midi": self.create_midi_pipeline
        }[type]()

    def create_midi_pipeline(self):
        self.pre_pipeline = [
            self.load_midi,
            # self.tokenize
        ]

        self.post_pipeline = [
        ]

    def process(self, path):
        result = None
        for func in self.pre_pipeline:
            if result is None:
                result = func(path) # loading function
            else:
                result = func(result)
        return result

    def post_process(self):
        raise NotImplementedError

    def tokenize(self):
        raise NotImplementedError

    def load_midi(self, midi_path):
        return PrettyMIDI(midi_file=midi_path)
