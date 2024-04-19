from pretty_midi import PrettyMIDI
import yaml
import glob
import os

class Pipeline():
    def __init__(self,
                 config_path, model_name):

        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.transform_configs = self.config['dataloaders']['validation']['transforms'][0][model_name]

        self.transform_functions = {
            'LoadMidi': self.load_midi,
            'LoadTokABC': self.load_tok_ABC
        }

        self.pre_pipeline = self.create_pipeline(self.transform_configs)
        self.post_pipeline = []

    def create_pipeline(self, transform_configs):
        pipeline = []
        for transform_config in transform_configs:
            if isinstance(transform_config, dict):
                transform_name, params = next(iter(transform_config.items()))
                if params is not None:
                    pipeline.append((self.transform_functions[transform_name], params))
                else:
                    pipeline.append((self.transform_functions[transform_name], {}))
        return pipeline

    def process(self, path):
        data = None
        for transform_func, params in self.pre_pipeline:
            data = transform_func(data if data is not None else path, **params)
        return data

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
