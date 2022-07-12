import os
import sys
import mlflow
import logging
from collections import namedtuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Modelling.utils import Utils  # noqa
from Modelling.svm import SVM_Model  # noqa
from Modelling.knn import KNN_Model  # noqa
from Modelling.logistic_regression import LR_Model  # noqa
from Modelling.random_forests import RF_Model  # noqa
from Modelling.lightGBM import lightGBM_Model  # noqa
from Modelling.TabNet import TabNet_Model  # noqa
from datasets import datasets # noqa


def run_experiments(experiment_name, dataset_names_to_run = []):
    mlflow.set_experiment(experiment_name)
    model_with_name = namedtuple("model_with_name", ["model_name", "model"])

    dataset_info_objects = datasets
    if(len(dataset_names_to_run) != 0):
        dataset_info_objects = list(
            filter(lambda dataset: dataset.name in dataset_names_to_run, dataset_info_objects))

    # TODO: Fix confusion matrix metrics where NaN values appear
    for dataset_info in dataset_info_objects:
        dataset_name = dataset_info.name
        model_instances = [
            model_with_name._make(["LR", LR_Model()]),
            model_with_name._make(["RF", RF_Model()]),
            model_with_name._make(["KNN", KNN_Model()]),
            model_with_name._make(["SVM", SVM_Model()]),
            model_with_name._make(["lightGBM", lightGBM_Model()]),
            model_with_name._make(["TabNet", TabNet_Model()]),
        ]
        df = Utils.load_data(dataset_name)

        for model_name, model_instance in model_instances:
            run_name = f"Model: {model_name} - Dataset: {dataset_name}"
            try:
                model_instance.mlflow_run(df, K=5, run_name=run_name, verbose=1)
            except Exception as e:
                logging.error(f"FAILED to complete {run_name}")


if __name__ == "__main__":
    run_experiments()
