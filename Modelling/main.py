import os
import sys
from collections import namedtuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Modelling.utils import Utils  # noqa
from Modelling.svm import SVM_Model  # noqa
from Modelling.knn import KNN_Model  # noqa
from Modelling.logistic_regression import LR_Model  # noqa
from Modelling.random_forests import RF_Model  # noqa
from Modelling.lightGBM import lightGBM_Model  # noqa


def run_all_experiments():
    model_with_name = namedtuple("model_with_name", ["model_name", "model"])

    model_instances = [
        model_with_name._make(["SVM", SVM_Model()]),
        model_with_name._make(["KNN", KNN_Model()]),
        model_with_name._make(["LR", LR_Model()]),
        model_with_name._make(["RF", RF_Model()]),
        model_with_name._make(["lightGBM", lightGBM_Model()])
    ]

    for dataset in Utils.get_dataset_names():
        df = Utils.load_data(dataset)
        for model_name, model_instance in model_instances:
            run_name = f"Model: {model_name} - Dataset: {dataset}"
            for k in [4, 12]:
                model_instance.mlflow_run(
                    df, K=k, run_name=run_name, verbose=True)


if __name__ == "__main__":
    run_all_experiments()
