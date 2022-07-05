import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import glob
import logging
from datetime import datetime

class Feature_Extractor:
    """
    Feature extraction class containing the methods to extract features for each voice sample
    """
    @staticmethod
    def extract_acoustic_features(voice_sample, f0_min, f0_max, unit):
        """
        Extracts the acoustic features such as the jitters and shimmers from the voice sample using functions from Praat software

        Parameters:
        voice_sample : .wav file
            the voice sample we want to extract the features from
        f0_min: integer
            minimum fundamental frequency of the signal. defualt is 75 on Praat software
        f0_max: integer
            maximum fundamental frequency of the signal. default is 500 on Praat software

        """
        try:
            sound = parselmouth.Sound(voice_sample)
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            f0_mean = call(pitch, "Get mean", 0, 0, unit) 
            f0_std_deviation= call(pitch, "Get standard deviation", 0, 0, unit) 
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
            jitter_relative = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_absolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_relative =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_localDb = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
            return f0_mean, f0_std_deviation, hnr, jitter_relative, jitter_absolute, jitter_rap, jitter_ppq5, shimmer_relative, shimmer_localDb, shimmer_apq3, shimmer_apq5
        except:
            logging.error(f"Unable to process this file: {voice_sample}")

    @staticmethod
    def extract_acoustic_features_2(voice_sample, f0_min, f0_max, unit):
        """
        Extracts the acoustic features such as the jitters and shimmers from the voice sample using functions from Praat software

        These fetaures are the features used in the research paper that used mdvr_kcl dataset

        Parameters:
        voice_sample : .wav file
            the voice sample we want to extract the features from
        f0_min: integer
            minimum fundamental frequency of the signal. defualt is 75 on Praat software
        f0_max: integer
            maximum fundamental frequency of the signal. default is 500 on Praat software

        """
        try:
            sound = parselmouth.Sound(voice_sample)
            pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
            f0_mean = call(pitch, "Get mean", 0, 0, unit) 
            f0_max = call(pitch, "Get maximum", 0, 0, unit, "Parabolic") 
            f0_min = call(pitch, "Get minimum", 0, 0, unit, "Parabolic") 
            f0_std_deviation= call(pitch, "Get standard deviation", 0, 0, unit) 
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            pointProcess = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
            jitter_relative = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_absolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ddp = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_relative =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_localDb = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            return f0_mean, f0_max, f0_min, jitter_relative, jitter_absolute, jitter_rap, jitter_ddp, shimmer_relative, shimmer_localDb, shimmer_apq3, shimmer_apq5, shimmer_dda, hnr
        except Exception as e:
            logging.error(f"Unable to process this file: {voice_sample} {e}")
    
    @staticmethod
    def extract_mfcc(voice_sample):
        """
        Extracts the mel frequency ceptral coefficients from the voice sample

        Parameters:
        voice_sample : .wav file
            the voice sample we want to extract the features from
        """
        sound = parselmouth.Sound(voice_sample)
        mfcc_object = sound.to_mfcc(number_of_coefficients=12) #the optimal number of coeefficient used is 12
        mfcc = mfcc_object.to_array()
        mfcc_mean = np.mean(mfcc.T,axis=0)
        return mfcc_mean
    
    @staticmethod
    def extract_mfcc_from_folder(folder_path):
        features = []
        for filename in glob.glob(folder_path):
            try:
                mfcc_per_file = Feature_Extractor.extract_mfcc(filename)
                features.append([filename, mfcc_per_file])
            except Exception as e:
                logging.error(f"error while handling file: {filename} {e}")
        df = pd.DataFrame(features, columns=['voiceID','mfcc'])
        df[['mfcc_feature0','mfcc_feature1','mfcc_feature2', 'mfcc_feature3','mfcc_feature4','mfcc_feature5', 'mfcc_feature6', 'mfcc_feature7','mfcc_feature8', 'mfcc_feature9', 'mfcc_feature10','mfcc_feature11', 'mfcc_feature12']] = pd.DataFrame(df.mfcc.to_list())
        df = df.drop(columns=['mfcc'])
        return df

    @staticmethod
    def extract_features_from_folder(folder_path):
        file_list = []
        mean_F0_list = []
        sd_F0_list = []
        hnr_list = []
        localJitter_list = []
        localabsoluteJitter_list = []
        rapJitter_list = []
        ppq5Jitter_list = []
        localShimmer_list = []
        localdbShimmer_list = []
        apq3Shimmer_list = []
        aqpq5Shimmer_list = []

        for file in glob.glob(folder_path):
            try:
                (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer) = Feature_Extractor.extract_acoustic_features(file, 75, 500, "Hertz") 
                file_list.append(file) # make an ID list
                mean_F0_list.append(meanF0) # make a mean F0 list
                sd_F0_list.append(stdevF0) # make a sd F0 list
                hnr_list.append(hnr)
                localJitter_list.append(localJitter)
                localabsoluteJitter_list.append(localabsoluteJitter)
                rapJitter_list.append(rapJitter)
                ppq5Jitter_list.append(ppq5Jitter)
                localShimmer_list.append(localShimmer)
                localdbShimmer_list.append(localdbShimmer)
                apq3Shimmer_list.append(apq3Shimmer)
                aqpq5Shimmer_list.append(aqpq5Shimmer)
            except:
                logging.error(f"missed: {file}")
        df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, sd_F0_list, hnr_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ppq5Jitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list]), columns=['voiceID','meanF0Hz', 'stdevF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer'])  
        return df

    @staticmethod
    def extract_features_from_folder_2(folder_path): #for the MDVR_KCL dataset (replication)
        file_list = []
        mean_F0_list = []
        max_F0_list = []
        min_F0_list = []
        hnr_list = []
        localJitter_list = []
        localabsoluteJitter_list = []
        rapJitter_list = []
        ddpJitter_list = []
        localShimmer_list = []
        localdbShimmer_list = []
        apq3Shimmer_list = []
        aqpq5Shimmer_list = []
        ddaShimmer_list = []
        curr_time = datetime.now()

        for file in glob.glob(folder_path):
            try:
                (meanF0, maxF0, minF0, localJitter, localabsoluteJitter, rapJitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, ddaShimmer, hnr) = Feature_Extractor.extract_acoustic_features_2(file, 75, 500, "Hertz") 
                file_list.append(file) # make an ID list
                mean_F0_list.append(meanF0) # make a mean F0 list
                max_F0_list.append(maxF0)
                min_F0_list.append(minF0)
                localJitter_list.append(localJitter)
                localabsoluteJitter_list.append(localabsoluteJitter)
                rapJitter_list.append(rapJitter)
                ddpJitter_list.append(ddpJitter)
                localShimmer_list.append(localShimmer)
                localdbShimmer_list.append(localdbShimmer)
                apq3Shimmer_list.append(apq3Shimmer)
                aqpq5Shimmer_list.append(aqpq5Shimmer)
                ddaShimmer_list.append(ddaShimmer)
                hnr_list.append(hnr)
            except Exception as e:
                logging.error(f"Missed {file} {e}")
        df = pd.DataFrame(np.column_stack([file_list, mean_F0_list, max_F0_list,min_F0_list, localJitter_list, localabsoluteJitter_list, rapJitter_list, ddpJitter_list, localShimmer_list, localdbShimmer_list, apq3Shimmer_list, aqpq5Shimmer_list, ddaShimmer_list, hnr_list]), columns=['voiceID','meanF0Hz', 'maxF0Hz', 'minF0Hz', 'localJitter', 'localabsoluteJitter', 'rapJitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'ddaShimmer', 'hnr'])  
        return df

    @staticmethod
    def convert_to_csv(df, filename):
        df.to_csv(filename+".csv", index=False)



