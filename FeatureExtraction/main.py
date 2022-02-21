from feature_extraction import Feature_Extractor
import visualization
import pandas as pd

dataset_paths = {
    "MDVR_KCL": (r"data/dataset/ReadText/HC/*.wav", r"data/dataset/ReadText/PD/*.wav"),
    "ItalianParkinsonSpeech": (
        r"data/dataset/ItalianParkinsonSpeech/EHC/*.wav",
        r"data/dataset/ItalianParkinsonSpeech/PD/*.wav",
    ),
}


def test_one(filename="data/dataset/ReadText/HC/ID00_hc_0_0_0.wav"):
    # To extract features of a file
    features = Feature_Extractor.extract_acoustic_features(
        filename, 75, 100, "Hertz")
    mfcc = Feature_Extractor.extract_mfcc(filename)

    print(
        "Acoustic features \n f0_mean, f0_std_deviation, hnr, jitter_relative, jitter_absolute, jitter_rap, jitter_ppq5, shimmer_relative, shimmer_localDb, shimmer_apq3, shimmer_apq5 \n",
        features,
    )
    print("mfcc \n", mfcc)

    print(len(mfcc), len(features))

    visualization.visualize_sound_sample(filename)


def extract_acoustic_features(folder_hc, folder_pd):
    ######################Extract the acoustic features in the dataset folder####################

    df_hc = Feature_Extractor.extract_features_from_folder(folder_hc)
    df_hc["label"] = 0

    df_pd = Feature_Extractor.extract_features_from_folder(folder_pd)
    df_pd["label"] = 1

    df_acoustic_features = pd.concat([df_hc, df_pd])
    return df_acoustic_features


def extract_acoustic_features_v2(folder_hc, folder_pd):
    df_hc = Feature_Extractor.extract_features_from_folder_2(
        folder_hc
    )  # for replicating the original ALC research paper
    df_hc["label"] = 0
    df_pd = Feature_Extractor.extract_features_from_folder_2(folder_pd)
    df_pd["label"] = 1
    df_acoustics = pd.concat([df_hc, df_pd])
    return df_acoustics


def extract_mfcc_features(folder_hc, folder_pd):
    df_mfcc_hc = Feature_Extractor.extract_mfcc_from_folder(folder_hc)
    df_mfcc_hc["label"] = 0
    df_mfcc_pd = Feature_Extractor.extract_mfcc_from_folder(folder_pd)
    df_mfcc_pd["label"] = 1
    df_mfcc_features = pd.concat([df_mfcc_hc, df_mfcc_pd])
    return df_mfcc_features


def combine_acoustic_and_mfcc_features(df_acoustic_features, df_mfcc_features):
    ##################### Combine the both acoustic features and MFCC in a csv file##################
    df_acoustic_features = df_acoustic_features.drop(columns=["label"])
    df_mfcc_features = df_mfcc_features.drop(columns=["voiceID"])
    df_all_features = pd.concat(
        [df_acoustic_features, df_mfcc_features], axis=1)
    return df_all_features


def main():
    for name, (folder_hc, folder_pd) in dataset_paths.items():
        print(f"\n===== Extracting {name} dataset =====\n")
        acoustic_features = extract_acoustic_features(folder_hc, folder_pd)
        acoustic_features_v2 = extract_acoustic_features_v2(folder_hc, folder_pd)
        mfcc_features = extract_mfcc_features(folder_hc, folder_pd)
        combined_features = combine_acoustic_and_mfcc_features(
            acoustic_features, mfcc_features)
        combined_features_v2 = combine_acoustic_and_mfcc_features(
            acoustic_features_v2, mfcc_features)
        combined_features.to_csv(
            f"data/extracted_features/{name}/All_features.csv", index=False)
        combined_features_v2.to_csv(
            f"data/extracted_features/{name}/All_features_v2.csv", index=False)


if __name__ == "__main__":
    main()
