from FeatureEngineering.dataset_downloader import download_datasets
from FeatureEngineering.main import Pipeline
from Modelling.main import run_experiments

EXPERIMENT_NAME = "Test - v0"

if __name__ == "__main__":
    download_datasets()

    # Note: If empty - will run extraction for all 24 datasets
    VIP_dataset_names = [
        "MDVR_KCL",
        # "MDVR_KCL_min_silence_500ms",
        # "MDVR_KCL_min_silence_1000ms",
        # "MDVR_KCL_min_silence_2000ms",
        # "MDVR_KCL_chunk_500ms",
        # "MDVR_KCL_chunk_1000ms",
        # "MDVR_KCL_chunk_3000ms",
        # "MDVR_KCL_chunk_5000ms",
        # "ItalianParkinsonSpeech",
        # "ItalianParkinsonSpeech_min_silence_500ms",
        # "ItalianParkinsonSpeech_min_silence_1000ms",
        # "ItalianParkinsonSpeech_min_silence_2000ms",
        # "ItalianParkinsonSpeech_chunk_500ms",
        # "ItalianParkinsonSpeech_chunk_1000ms",
        # "ItalianParkinsonSpeech_chunk_3000ms",
        # "ItalianParkinsonSpeech_chunk_5000ms",
        # "mPower",
        # "mPower_min_silence_500ms",
        # "mPower_min_silence_1000ms",
        # "mPower_min_silence_2000ms",
        # "mPower_chunk_500ms",
        # "mPower_chunk_1000ms",
        # "mPower_chunk_3000ms",
        # "mPower_chunk_5000ms",
    ]
    
    # feature_engineer = Pipeline()
    # feature_engineer.run(VIP_dataset_names)

    run_experiments(EXPERIMENT_NAME, VIP_dataset_names)
