from FeatureEngineering.dataset_downloader import download_datasets
from FeatureEngineering.main import Pipeline
from Modelling.main import run_all_experiments


if __name__ == "__main__":
    download_datasets()
    
    # Note: If empty - will run extraction for all 24 datasets
    VIP_dataset_names = [

    ]
    # feature_engineer = Pipeline()
    # feature_engineer.run(names=VIP_dataset_names)

    run_all_experiments()
