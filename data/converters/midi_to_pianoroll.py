import pypianoroll
import pretty_midi


class MidiToPianoroll:
    """
    Interface for MIDI to pianoroll conversion.
    """

    @staticmethod
    def convert_from_file(midi_path: str):
        return pypianoroll.read(midi_path)

    @staticmethod
    def convert_from_prett_midi(midi: pretty_midi.PrettyMIDI):
        return pypianoroll.from_pretty_midi(midi)

    @staticmethod
    def plot(pianoroll: pypianoroll.Multitrack, save_path: str, number_of_beats: int = -1):
        """
        Visualise the pianoroll with matplotlib. 4 tracks or less are optimal
        Args:
            pianoroll: multitrack acquired either from convert_from_prett_midi or convert_from_file
            save_path: plot save path; leave empty if plot opening in new window is sufficient
            number_of_beats: How many first beats should be included in the plot (-1 if all of them)
                recommended values: [8..32]
        """
        if number_of_beats > 0:
            pianoroll.trim(0, number_of_beats * pianoroll.resolution)
        axs = pianoroll.plot()
        if axs and save_path:
            fig = axs[0].get_figure()
            fig.savefig(save_path)
