# Robust/Stable Machine Learning Approaches to Predict Parkinson’s Disease using Speech Signals


## Datasets
All datasets (original wav files, synthetic wav files and extracted features in csv format) are stored in the data folder. 

### Original Data
This project so far uses two datasets 

1. MDVR-KCL Dataset 
   - The data/dataset/ReadText/HC folder contains the audio recordings for healthy controls
   - The data/dataset/ReadText/PD folder contains the audio recordings for PD patients
2. Italian Parkinson Speech Dataset
    - The data/dataset/ReadText/HC contains the audio recordings for healthy controls
    - The data/dataset/ReadText/PD contains the audio recordings for PD patients
    
### Synthetic Data
The PyDub library was used to generate more samples/speech files by splitting the existing recordings 
1. Into chunks of lengths [500, 1000, 3000, 5000] milliseconds
2. On Silence, with a minimum silence of lengths [500, 1000, 2000] milliseconds

This means at least 7 new datasets were created for each original dataset - allowing us to experiment across 16 datasets

The synthethic datasets are at data/synthetic

### Extracted Features
Running our Feature Extraction algorithm via parselmouth's pratt, we extract acousti and MFCC features from every wav file for each dataset - storing the results in different csv files

Extracted features: 
1. meanF0Hz
2. maxF0Hz
3. minF0Hz
4. localJitter
5. localabsoluteJitter
6. rapJitter
7. ddpJitter
8. localShimmer
9. localdbShimmer
10. apq3Shimmer
11. apq5Shimmer
12. ddaShimmer
13. hnr
14. mfcc_feature0 - mfcc_feature12

Extracted features for all datasets are stored at data/extracted_features


## Setup
1. Install the project's dependencies
    ```
    pip install -r requirements.txt 
    ```

2. This project uses mlflow to track experiments.
   ```
   pip install mlflow
   ```

3. If you don't intend to rerun experiments and would just like to see the results
   ```
   mlflow ui
   ```

4. The entry point to the program allows you perform all the steps in one go by simply running:
    ```
    python main.py
    ```
    However, if you'd like to run only one step of the process, feel free to run any of the commands seen below:
    - Generate the synthetic datasets
        ```
        python DataEngineering/main.py
        ```
    - Extract the features from these datasets and store the results in csvs
        ```
        python FeatureExtraction/main.py
        ```
    - Run the model training experiments and log the results to mlflow
        ```
        python Modelling/main.py
        ```

## Project Structure

```bash
.
├── DataEngineering
│   ├── main.py
│   └── split_speech.py
├── FeatureExtraction
│   ├── feature_extraction.py
│   ├── main.py
│   └── visualization.py
├── Modelling
│   ├── knn.py
│   ├── lightGBM.py
│   ├── logistic_regression.py
│   ├── main.py
│   ├── random_forests.py
│   ├── svm.py
│   └── utils.py
├── README.md
├── main.py
├── requirements.txt
└── utils
    └── timer.py
```

> *Annotated version
```bash
.
├── DataEngineering
│   ├── main.py (Entry point for Synthetic data generation only)
│   └── split_speech.py (Contains Speech_splitter class - wrapper for PyDub)
├── FeatureExtraction
│   ├── feature_extraction.py (Contains Feature Extractor Class - Wrapper for ParselMouth Praat)
│   ├── main.py (Entry point for Feature Extraction only)
│   └── visualization.py
├── Modelling (Contains classes for models - wrappers around Scikit-Learn and MlFlow)
│   ├── knn.py
│   ├── lightGBM.py
│   ├── logistic_regression.py
│   ├── main.py (Entry point for model training only)
│   ├── random_forests.py
│   ├── svm.py 
│   └── utils.py (MLFlow utils class and Utils class for loading datasets, generating metrics and more)
├── README.md
├── main.py (Entry Point - Run all steps of the experiment)
├── requirements.txt
└── utils
    └── timer.py
```

## Reproducibility of results
For quick verification of results, a `Notebook for reproducibility` folder contains a notebook and some processed datasets 

## More Info
1. To create a new experiment on mlflow
   - Set the experiment via environment variables
        ```
        export MLFLOW_EXPERIMENT_NAME=pd-detection
        ```
   - Run this command in your cli
        ```
        mlflow experiments create --experiment-name pd-detection
        ```
