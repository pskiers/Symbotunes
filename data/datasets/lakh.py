import os
import tarfile
from typing import Callable
from mido import MidiFile, MidiTrack, MetaMessage, Message


from .Pipeline.Pipeline import Pipeline
from .base import BaseDataset
from .utils.downloader import Downloader, DownloadError
from .utils.file_utility import FileUtility


class LakhMidiDataset(BaseDataset):
    """
    LakhMIDI Dataset class
    available on: https://colinraffel.com/projects/lmd/#get
    """

    def __init__(
        self,
        root: str = "_data",
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        preload: bool = True,
        download: bool = True,
        replace_if_exists: bool = False,
        **kwargs
    ) -> None:
        super().__init__(root, split, download, replace_if_exists, transform, target_transform, **kwargs)

        self.pipeline = Pipeline(type="midi_path")

        self.data = []  # type: ignore[assignment]
        self.targets = []  # type: ignore[assignment]

        if preload:
            self._load_data()

    def _load_data(self):
        # self._targets = []
        self._data = self.pipeline.process(os.path.join(self.root, "train", "lmd_full"))
        # handle targets

    def __getitem__(self, index: int) -> tuple:
        midi_data = self.data[index]

        # target = self._targets[index]

        if self.transform:
            midi_data = self.transform(midi_data)
        # if self.target_transform:
        #     target = self.target_transform(target)9

        return midi_data  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self.data)

    def remove_empty_bars_from_midi(self, file):
        try:
            midi = MidiFile(file)
        except OSError:  # Corrupted MIDI file
            os.remove(file)
            return
        midi = MidiFile(file)
        new_midi = MidiFile()

        ticks_per_beat = midi.ticks_per_beat

        # Default time signature (4/4)
        time_signature = (4, 4)
        for msg in midi.tracks[0]:
            if msg.type == 'time_signature':
                time_signature = (msg.numerator, msg.denominator)
                break

        # Calculate ticks per bar
        beats_per_bar = time_signature[0]
        ticks_per_bar = beats_per_bar * ticks_per_beat

        for track in midi.tracks:
            new_track = MidiTrack()
            new_midi.tracks.append(new_track)

            current_time = 0
            bar_start_time = 0
            empty_bar = True
            buffer = []

            first_note_on_occured = False

            for msg in track:
                current_time += msg.time

                if current_time >= bar_start_time + ticks_per_bar:
                    bar_start_time += ticks_per_bar
                    if not empty_bar:
                        new_track.extend(buffer)
                    empty_bar = True
                    buffer = []

                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        empty_bar = False
                    if not first_note_on_occured:
                        first_note_on_occured = True
                        msg.time = 1

                if msg.type == 'note_on' and msg.velocity == 0:  # Note-on with velocity 0 means note-off
                    note_off_msg = Message('note_off', note=msg.note, velocity=0, channel=msg.channel, time=msg.time)
                    buffer.append(note_off_msg)
                else:
                    buffer.append(msg)

            if not empty_bar:
                new_track.extend(buffer)

        new_midi.save(file)

    def _split_tracks(self, midi_directory, midi_filename):
        midi_full_path = os.path.join(midi_directory, midi_filename)
        try:
            midi = MidiFile(midi_full_path)
        except OSError:  # Corrupted MIDI file
            return

        channel_tracks = {}
        for i, track in enumerate(midi.tracks):
            for msg in track:
                if msg.type in ['note_on', 'note_off', 'polytouch', 'control_change',
                                'program_change', 'aftertouch', 'pitchwheel']:
                    if msg.channel == 9:  # Channel 10 is reserved for drums (Ch9 for 0-indexed)
                        continue
                    if msg.channel not in channel_tracks:
                        channel_tracks[msg.channel] = MidiTrack()
                        channel_tracks[msg.channel].append(MetaMessage('track_name', name=f'Track {i} Channel {msg.channel}', time=0))
                    channel_tracks[msg.channel].append(msg)
                elif msg.type == 'sysex' or msg.type == 'meta':
                    for channel, ch_track in channel_tracks.items():
                        ch_track.append(msg)

        for channel, ch_track in channel_tracks.items():
            new_midi = MidiFile()
            new_midi.tracks.append(ch_track)
            output_filename = FileUtility.get_filename_with_postfix(midi_directory, midi_filename)
            new_midi.save(os.path.join(midi_directory, output_filename))

    def download(self) -> None:
        # Temporary substitution, so that we don't download 1.7 GB of midi each time.
        # self.url = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
        self.url = "https://drive.google.com/uc?export=download&id=1aV4rNwtb3b8f55bxmoTOmqc0zBPJ1MIp"

        dest_path = os.path.join(
            self.root,
            "train",
        )
        dataset_path = os.path.join(dest_path, "lmd_full")

        if not self.replace_if_exists and os.path.exists(dataset_path):
            print("Dataset directory already exists. Skipping download.")
            return

        os.makedirs(dest_path, exist_ok=True)
        tarball_path = os.path.join(dest_path, "lakh.tar.gz")
        try:
            Downloader.download(self.url, tarball_path)
        except DownloadError as e:
            print(e)

        with tarfile.open(tarball_path, "r:*") as file:
            file.extractall(dest_path)
        os.remove(tarball_path)

        print("Removing drums...")
        file_list = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                file_list.append((root, file))
        print("Splitting tracks by channel...")
        for root, file in file_list:
            self._split_tracks(root, file)
        print("Removing empty bars...")
        for root, _, files in os.walk(dataset_path):
            for file in files:
                full_path = os.path.join(root, file)
                if "_" in file:
                    self.remove_empty_bars_from_midi(full_path)
                else:
                    os.remove(full_path)
