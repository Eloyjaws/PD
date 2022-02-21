import glob
import os.path
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
from feature_extraction import Feature_Extractor


class SpeechSplitter:
    """ 
    Utils to split wav files into chunks
    """

    @staticmethod
    def split_on_silence(folder_path=r"data/dataset/ReadText/HC/*.wav", output_dir=r"data/dataset/MDVR/HC", min_silence_len=1000, silence_thresh=-40):
        for filepath in glob.glob(folder_path):
            try:
                # split to get parent_dir and filename like ID01_hc_0_0_0.wav
                parent_dir, filename = os.path.split(filepath)
                patientID = filename.split('_')[0]

                output_file_path = os.path.join(output_dir, patientID)
                Path(output_file_path).mkdir(parents=True, exist_ok=True)

                sound_file = AudioSegment.from_wav(filepath)
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

    @staticmethod
    def split_into_chunks(folder_path=r"data/dataset/ReadText/HC/*.wav", output_dir=r"data/dataset/MDVR/HC", chunk_length_ms=3000):
        for filepath in glob.glob(folder_path):
            try:
                # split to get parent_dir and filename like ID01_hc_0_0_0.wav
                parent_dir, filename = os.path.split(filepath)
                patientID = filename.split('_')[0]

                output_file_path = os.path.join(output_dir, patientID)
                Path(output_file_path).mkdir(parents=True, exist_ok=True)

                sound_file = AudioSegment.from_wav(filename)
                audio_chunks = make_chunks(sound_file, chunk_length_ms)

                for chunk_id, chunk in enumerate(audio_chunks):
                    out_file = os.path.join(
                        output_file_path, f"chunk_{chunk_id}.wav")
                    chunk.export(out_file, format="wav")
            except Exception as e:
                print(e)
                print("error while handling file: ", filepath)
