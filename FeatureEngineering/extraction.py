import pandas as pd
from FeatureExtraction.feature_extraction import Feature_Extractor   # noqa

from utils.timer import start_timer, end_timer_and_print  # noqa

def extract_acoustic_features(folder_hc, folder_pd, dataset_info, store):
    ###################### Extract the acoustic features in the dataset folder ####################
    event_name = f"Acoustic extraction for {dataset_info.name}"
    start_timer(event_name)
    df_hc = Feature_Extractor.extract_features_from_folder(folder_hc)
    df_pd = Feature_Extractor.extract_features_from_folder(folder_pd)
    df_hc["label"], df_pd["label"] = 0, 1
    store['acoustics'] = pd.concat([df_hc, df_pd])
    end_timer_and_print(event_name)


def extract_acoustic_features_v2(folder_hc, folder_pd, dataset_info, store):
    # for replicating the original ALC research paper
    event_name = f"Acoustic_V2 extraction for {dataset_info.name}"
    start_timer(event_name)
    df_hc = Feature_Extractor.extract_features_from_folder_2(folder_hc)  
    df_pd = Feature_Extractor.extract_features_from_folder_2(folder_pd)
    df_hc["label"], df_pd["label"] = 0, 1
    store['acoustics_v2'] = pd.concat([df_hc, df_pd])
    end_timer_and_print(event_name)


def extract_mfcc_features(folder_hc, folder_pd, dataset_info, store):
    event_name = f"MFCC extraction for {dataset_info.name}"
    start_timer(event_name)
    df_mfcc_hc = Feature_Extractor.extract_mfcc_from_folder(folder_hc)
    df_mfcc_pd = Feature_Extractor.extract_mfcc_from_folder(folder_pd)
    df_mfcc_hc["label"], df_mfcc_pd["label"] = 0, 1
    store['mfcc'] = pd.concat([df_mfcc_hc, df_mfcc_pd])
    end_timer_and_print(event_name)

def combine_acoustic_and_mfcc_features(df_acoustic_features, df_mfcc_features):
    ##################### Combine the both acoustic features and MFCC values ##################
    df_acoustic_features = df_acoustic_features.drop(columns=["label"])
    return pd.merge(df_acoustic_features, df_mfcc_features, on="voiceID", how="inner")