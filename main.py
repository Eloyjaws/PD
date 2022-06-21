from DataEngineering.main import download_datasets
from DataEngineering.main import generate_synthetic_datasets
from FeatureExtraction.main import extract_features_from_all_datasets
from Modelling.main import run_all_experiments


if __name__ == "__main__":
    download_datasets()
    generate_synthetic_datasets()
    extract_features_from_all_datasets()
    run_all_experiments()