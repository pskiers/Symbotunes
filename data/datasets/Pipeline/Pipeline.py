from pretty_midi import PrettyMIDI

class Pipeline():
    def __init__(self,
                 type):
        self.pre_pipeline = []
        self.post_pipeline = []

        self.type = type
        {
            "midi": self.create_midi_pipeline,
            "tok_ABC": self.create_tok_ABC_pipeline
        }[type]()

    def create_midi_pipeline(self):
        self.pre_pipeline = [
            self.load_midi,
            # self.tokenize_MIDI
        ]

        self.post_pipeline = [
        ]

    def create_tok_ABC_pipeline(self):
        self.pre_pipeline = [
            self.load_tok_ABC,
            # self.tokenize_ABC
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
    
    def load_tok_ABC(self, ABC_file):
        with open(ABC_file, 'r', encoding='utf-8') as file:
            file_contents = file.read()
            return file_contents.split('\n\n')
