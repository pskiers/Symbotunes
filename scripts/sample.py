import os
import argparse
import torch
from omegaconf import OmegaConf
from pathlib import Path

from models import get_model
from data.tokenizers import FolkTokenizer
from data.converters.abc_to_midi_converter import ABCTOMidiConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="path to model checkpoint file",
    )
    parser.add_argument("--batch", "-b", type=int, required=True, help="amount of samples")
    parser.add_argument("--out", "-o", type=Path, required=False, help="output directory", default="samples")
    args = parser.parse_args()
    config_path = str(args.path)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None
    batch_size = args.batch
    out_path = args.out

    config = OmegaConf.load(config_path)

    model_type = get_model(config.model.get("model_type"))
    model = model_type.load_from_checkpoint(checkpoint_path, **config.model.get("params", dict()))
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    samples = model.sample(batch_size)

    # TODO handle other tokenizers. Maybe read the what tokenizer should be used from config somehow. Otherwise, if 
    # getting tokenizer type from config file is ugly then either add argument to argparser, or even add the 
    # converter to config file. Or maybe do something else entirely, I dunno.
    tokenizer = FolkTokenizer()
    converter = ABCTOMidiConverter(tokenizer)

    if not os.path.exists(out_path):
        os.makedirs(out_path,)
    for i, sample in enumerate(samples):
        try:
            converter(sample.cpu(), os.path.join(out_path, f"sample_{i}.mid"))
        except Exception:
            print(f"Invalid format of sample {i}")

