import glob
import sys
import os.path
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.timer import start_timer, end_timer_and_print, log  # noqa


class SpeechSplitter:
    """ 
    Utils to split wav files into chunks
    """

    @staticmethod
    def split_on_silence(folder_path, output_dir, min_silence_len=1000, silence_thresh=-40, tag="split_on_silence"):
        start_timer(tag)
        for filepath in glob.glob(folder_path):
            try:
                # split to get parent_dir and filename like ID01_hc_0_0_0.wav
                parent_dir, filenameWithExt = os.path.split(filepath)
                filename, ext = os.path.splitext(filenameWithExt)

                patientID = filename.split('_')[0]
                if "mPower" in filepath:
                    patientID = filename.split('.m4a-')[-1]
                if "Italian" in filepath:
                    leading_path, patientID = os.path.split(parent_dir)

                output_file_path = os.path.join(output_dir, patientID)
                Path(output_file_path).mkdir(parents=True, exist_ok=True)

                sound_file = AudioSegment.from_file(filepath)
                audio_chunks = split_on_silence(sound_file,
                                                # must be silent for at least half a second
                                                min_silence_len=min_silence_len,
                                                # consider it silent if quieter than -16 dBFS
                                                silence_thresh=silence_thresh)

                for chunk_id, chunk in enumerate(audio_chunks):
                    out_file = os.path.join(
                        output_file_path, f"chunk_{chunk_id}.wav")
                    chunk.export(out_file, format="wav")

            except Exception as e:
                print(e)
                print("error while handling file: ", filepath)
        end_timer_and_print(tag)

    @staticmethod
    def split_into_chunks(folder_path, output_dir, chunk_length_ms=3000, tag="split_into_chunks"):
        start_timer(tag)
        for filepath in glob.glob(folder_path):
            try:
                # split to get parent_dir and filename like ID01_hc_0_0_0.wav
                parent_dir, filenameWithExt = os.path.split(filepath)
                filename, ext = os.path.splitext(filenameWithExt)

                patientID = filename.split('_')[0]
                if "mPower" in filepath:
                    patientID = filename.split('.m4a-')[-1]
                if "Italian" in filepath:
                    leading_path, patientID = os.path.split(parent_dir)

                output_file_path = os.path.join(output_dir, patientID)
                Path(output_file_path).mkdir(parents=True, exist_ok=True)

                sound_file = AudioSegment.from_wav(filepath)
                audio_chunks = make_chunks(sound_file, chunk_length_ms)

                for chunk_id, chunk in enumerate(audio_chunks):
                    out_file = os.path.join(
                        output_file_path, f"chunk_{chunk_id}.wav")
                    chunk.export(out_file, format="wav")
            except Exception as e:
                print(e)
                print("error while handling file: ", filepath)
        end_timer_and_print(tag)
