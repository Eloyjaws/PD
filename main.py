from FeatureEngineering.dataset_downloader import download_datasets
from FeatureEngineering.main import Pipeline
from Modelling.main import run_all_experiments


if __name__ == "__main__":
    download_datasets()

    # Note: If empty - will run extraction for all 24 datasets
    VIP_dataset_names = [
        "ItalianParkinsonSpeech",
        "ItalianParkinsonSpeech_split_on_silence_500ms",
        "ItalianParkinsonSpeech_split_on_silence_1000ms",
        "ItalianParkinsonSpeech_split_on_silence_2000ms",
        "ItalianParkinsonSpeech_chunk_500ms",
        "ItalianParkinsonSpeech_chunk_1000ms",
        "ItalianParkinsonSpeech_chunk_3000ms",
        "ItalianParkinsonSpeech_chunk_5000ms",
        "mPower",
        "mPower_split_on_silence_500ms",
        "mPower_split_on_silence_1000ms",
        "mPower_split_on_silence_2000ms",
        "mPower_chunk_500ms",
        "mPower_chunk_1000ms",
        "mPower_chunk_3000ms",
        "mPower_chunk_5000ms",
    ]
    feature_engineer = Pipeline()
    feature_engineer.run(names=VIP_dataset_names)

    # run_all_experiments()
