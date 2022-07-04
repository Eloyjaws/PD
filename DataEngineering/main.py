import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from DataEngineering.split_speech import SpeechSplitter  # noqa
from DataEngineering.dataset_downloader import download_mPower_dataset, download_MDVR_KCL_dataset, download_italian_dataset  # noqa


def generate_synthetic_datasets():

    min_silence_lengths = [500, 1000, 2000]
    chunk_lengths = [500, 1000, 3000, 5000]

    dataset_paths = {
        # "MDVR_KCL_PD": (r"data/dataset/ReadText/PD/*.wav", r"data/synthetic/MDVR_KCL#/PD"),
        # "MDVR_KCL_HC": (r"data/dataset/ReadText/HC/*.wav", r"data/synthetic/MDVR_KCL#/HC"),
        # "ItalianParkinsonSpeech_EHC": (
        #     r"data/dataset/ItalianParkinsonSpeech/EHC/*.wav",
        #     r"data/synthetic/ItalianParkinsonSpeech#/EHC",
        # ),
        # "ItalianParkinsonSpeech_PD": (
        #     r"data/dataset/ItalianParkinsonSpeech/PD/*.wav",
        #     r"data/synthetic/ItalianParkinsonSpeech#/PD",
        # ),
        "MPower_PD": (
            r"data/dataset/mPower/PD/*/*.wav",
            r"data/synthetic/mPower#/PD",
        ),
        "M_Power_HC": (
            r"data/dataset/mPower/HC/*/*.wav",
            r"data/synthetic/mPower#/HC",
        )
    }

    # TODO: Need to rewrite to extract PD & HC for a dataset in one go - before deleting it

    silence_threshold = -40

    for name, (folder_path, output_dir) in dataset_paths.items():
        print(
            f"\n=====> Splitting {name} dataset on silence lengths of {min_silence_lengths}ms <=====\n")
        for min_silence_length in min_silence_lengths:
            print(f"\n-----{min_silence_length}ms -----\n")
            left_path, right_path = output_dir.split('#')
            output_dir_extended = f"{left_path}_min_silence_{min_silence_length}ms{right_path}"
            SpeechSplitter.split_on_silence(
                folder_path, output_dir_extended, min_silence_length, silence_threshold)

    for name, (folder_path, output_dir) in dataset_paths.items():
        print(
            f"\n=====> Splitting {name} dataset into {chunk_lengths}ms chunks <=====\n")
        for chunk_length_ms in chunk_lengths:
            print(f"\n----- {chunk_length_ms}ms -----\n")
            left_path, right_path = output_dir.split('#')
            output_dir_extended = f"{left_path}_chunk_length_{chunk_length_ms}ms{right_path}"
            SpeechSplitter.split_into_chunks(
                folder_path, output_dir_extended, chunk_length_ms)


if __name__ == "__main__":
    download_MDVR_KCL_dataset()
    download_italian_dataset()
    download_mPower_dataset()
    generate_synthetic_datasets()
