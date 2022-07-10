import os
import sys
import shutil
import multiprocessing
import logging
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from DataEngineering.split_speech import SpeechSplitter  # noqa
from FeatureEngineering.datasets import datasets  # noqa
from FeatureEngineering.extraction import extract_acoustic_features, extract_acoustic_features_v2, extract_mfcc_features, combine_acoustic_and_mfcc_features  # noqa

from utils.timer import start_timer, end_timer_and_print, log  # noqa


class Tags:
    HC = " for HC"
    PD = " for PD"
    NESTED = "/*/*.wav"


def generate_dataset(dataset_info):
    event_name = f"Generate Dataset - {dataset_info.name}"
    if(dataset_info.create_dataset == False):
        log(f"SKIPPED: {event_name} | REASON: source dataset")
        return
    log(event_name)

    jobs = []

    if dataset_info.method == "silence":
        jobs.append(multiprocessing.Process(
            target = SpeechSplitter.split_on_silence,
            args=(dataset_info.source_HC, dataset_info.sink_HC, dataset_info.duration, -40, event_name + Tags.HC)
        ))
        jobs.append(multiprocessing.Process(
            target = SpeechSplitter.split_on_silence,
            args=(dataset_info.source_PD, dataset_info.sink_PD, dataset_info.duration, -40, event_name + Tags.PD)
        ))

    elif dataset_info.method == "chunk":
        jobs.append(multiprocessing.Process(
            target = SpeechSplitter.split_into_chunks,
            args=(dataset_info.source_HC, dataset_info.sink_HC, dataset_info.duration, event_name + Tags.HC)
        ))
        jobs.append(multiprocessing.Process(
            target = SpeechSplitter.split_into_chunks,
            args=(dataset_info.source_PD, dataset_info.sink_PD, dataset_info.duration, event_name + Tags.PD)
        ))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()


def extract_features_from_dataset(dataset_info):
    event_name = f"Extract features from Dataset - {dataset_info.name}"
    log(event_name)
    
    start_timer(event=event_name)

    manager = multiprocessing.Manager()
    store = manager.dict()
    jobs = []

    # For original datasets, the path to run feature extraction on is the source
    folder_hc = (dataset_info.sink_HC + Tags.NESTED) if dataset_info.create_dataset else dataset_info.source_HC
    folder_pd = (dataset_info.sink_PD + Tags.NESTED) if dataset_info.create_dataset else dataset_info.source_PD

    jobs.append(multiprocessing.Process(
        target = extract_acoustic_features,
        args = (
            folder_hc,
            folder_pd,
            dataset_info,
            store
        )
    ))

    jobs.append(multiprocessing.Process(
        target = extract_acoustic_features_v2,
        args = (
            folder_hc,
            folder_pd,
            dataset_info,
            store
        )
    ))

    jobs.append(multiprocessing.Process(
        target = extract_mfcc_features,
        args = (
            folder_hc,
            folder_pd,
            dataset_info,
            store
        )
    ))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()
    
    df_acoustic_features = store.get('acoustics')
    df_acoustic_features_v2 = store.get('acoustics_v2')
    df_mfcc_features = store.get('mfcc')

    df_all_features = combine_acoustic_and_mfcc_features(df_acoustic_features, df_mfcc_features)
    df_all_features_v2 = combine_acoustic_and_mfcc_features(df_acoustic_features_v2, df_mfcc_features)
    
    output_dir = f"data/extracted_features/{dataset_info.name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_all_features.to_csv(f"{output_dir}/All_features.csv", index=False)
    df_all_features_v2.to_csv(f"{output_dir}/All_features_v2.csv", index=False)
    
    end_timer_and_print(event_name)


def delete_dataset(dataset_info):
    event_name = f"Delete Dataset - {dataset_info.name}"
    if(dataset_info.create_dataset == False):
        log(f"SKIPPED: {event_name} | REASON: source dataset")
        return
    log(event_name)
    start_timer(event=event_name)
    parent_dir, folder_name = os.path.split(dataset_info.sink_HC)
    shutil.rmtree(parent_dir, onerror = lambda fn, filename, err: logging.error(err) )
    end_timer_and_print(event_name)


def process(dataset_info):
    log(f"Running job: {dataset_info.name}")
    try:
        generate_dataset(dataset_info)
        extract_features_from_dataset(dataset_info)
        delete_dataset(dataset_info)
    except Exception as e:
        logging.error(f"Feature engineering failed for {dataset_info.name} \n{e}")


class Pipeline:
    def __init__(self) -> None:
        self.timer = None
        self.threadpool = []


    def run(self, names=[]):
        dataset_info_objects = datasets
        if(len(names) != 0):
            dataset_info_objects = list(
                filter(lambda dataset: dataset.name in names, dataset_info_objects))

        jobs = []

        for dataset_info in dataset_info_objects:
            jobs.append(multiprocessing.Process(
                target = process,
                args = ([dataset_info])
            ))
        
        # Run jobs in twos
        for i in range(0, len(jobs), 2):
            jobs[i].start()
            if(i+1 < len(jobs)):
                jobs[i+1].start()
            
            jobs[i].join()
            if(i+1 < len(jobs)):
                jobs[i+1].join()



if __name__ == "__main__":
    
    # Note: If empty - will run extraction for all 24 datasets
    VIP_dataset_names = [

    ]

    feature_engineer = Pipeline()
    feature_engineer.run(names=VIP_dataset_names)
