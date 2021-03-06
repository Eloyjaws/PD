import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import mlflow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Modelling.utils import Utils  # noqa
from utils.timer import start_timer, end_timer_and_print, log  # noqa



class LR_Model():
    def __init__(self, params={}):
        """
        Constructor for Logistic Regression
        :param params: all hyperparameter options for the constructor
        """
        self.clf = LogisticRegression(**params)
        self._params = params

        # Default hyperparams

        # Scaler
        self.scaler = MinMaxScaler()

    @classmethod
    def new_instance(clf, params={}):
        return clf(params)

    @property
    def model(self):
        """
        Getter for the model
        :return: return the model
        """
        return self.clf

    @property
    def params(self):
        """
        Getter for the property the model
          :return: return the model params
        """
        return self._params


    def mlflow_run(self, df, K=5, run_name=f"LR Experiment", verbose=True):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param run_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """
        log(run_name)
        start_timer(run_name)
 
        best_accuracy = 0
        tags = {
            "model_class": "LR",
            "dataset_name": run_name.split(" - Dataset: ")[-1],
            }

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags)
            kfold = StratifiedKFold(K, shuffle=True, random_state=None)

            X_kfold = pd.DataFrame(df.iloc[:, :-1].values)
            y_kfold = pd.DataFrame(df.iloc[:, -1].values.ravel())

            k_accuracy_list, k_specificity, k_sensitivity, k_precision, k_f1 = [], [], [], [], []

            for i in range(1, 13):
                row, row_specificity, row_sensitivity, row_precision, row_f1 = [], [], [], [], []

                row.append(i)
                row_specificity.append(i)
                row_sensitivity.append(i)
                row_precision.append(i)
                row_f1.append(i)

                for train, test in kfold.split(X_kfold, y_kfold):
                    Xtrain_kfold = X_kfold.iloc[train, :]
                    Ytrain_kfold = y_kfold.iloc[train, :]
                    Xtest_kfold = X_kfold.iloc[test, :]
                    Ytest_kfold = y_kfold.iloc[test, :]

                    Xtrain_kfold = self.scaler.fit_transform(Xtrain_kfold)
                    Xtest_kfold = self.scaler.transform(Xtest_kfold)

                    model_new = LogisticRegression(max_iter=1000, n_jobs=-1)

                    model_new.fit(Xtrain_kfold, Ytrain_kfold.values.ravel())
                    y_pred_new = model_new.predict(Xtest_kfold)

                    conf_matrix_kfold = confusion_matrix(
                        Ytest_kfold, y_pred_new)

                    (accuracy, sensitivity, specificity, precision,
                     f1_score) = Utils.get_metrics_from_confusion_matrix(conf_matrix_kfold)

                    row.append(accuracy)
                    row_specificity.append(specificity)
                    row_sensitivity.append(sensitivity)
                    row_precision.append(precision)
                    row_f1.append(f1_score)

                    if(accuracy > best_accuracy):
                        best_accuracy = accuracy
                        self.clf = model_new

                # Add average across K folds for run i
                row.append(np.nanmean(row[1:]))
                row_specificity.append(np.nanmean(row_specificity[1:]))
                row_sensitivity.append(np.nanmean(row_sensitivity[1:]))
                row_precision.append(np.nanmean(row_precision[1:]))
                row_f1.append(np.nanmean(row_f1[1:]))

                # collate metrics for run No i
                k_accuracy_list.append(row)
                k_specificity.append(row_specificity)
                k_sensitivity.append(row_sensitivity)
                k_precision.append(row_precision)
                k_f1.append(row_f1)

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.clf, "LR-model")
            mlflow.log_param('K', K)

            avg_accuracy = np.nanmean(np.array(k_accuracy_list)[:, -1])
            avg_specificity = np.nanmean(np.array(k_specificity)[:, -1])
            avg_sensitivity = np.nanmean(np.array(k_sensitivity)[:, -1])
            avg_precision = np.nanmean(np.array(k_precision)[:, -1])
            avg_f1_score = np.nanmean(np.array(k_f1)[:, -1])
            
            # log metrics
            mlflow.log_metric("accuracy", avg_accuracy)
            mlflow.log_metric("specificity", avg_specificity)
            mlflow.log_metric("sensitivity", avg_sensitivity)
            mlflow.log_metric("recall", avg_precision)
            mlflow.log_metric("F1-Score", avg_f1_score)

            if verbose:
                log("LR Kfold Evaluation")
                Utils.print_aggregated_KFold_metric(
                    k_accuracy_list, "accuracy", K)
                Utils.print_aggregated_KFold_metric(
                    k_specificity, "specificity", K)
                Utils.print_aggregated_KFold_metric(
                    k_sensitivity, "sensitivity/recall", K)
                Utils.print_aggregated_KFold_metric(
                    k_precision, "precision", K)
                Utils.print_aggregated_KFold_metric(k_f1, "f1 score", K)

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id
            log("Completed MLflow Run with run_id {} and experiment_id {}".format(
                runID, experimentID))
            end_timer_and_print(run_name)
            return (experimentID, runID)

    def save(self, path="."):
        # Save the model in cloudpickle format
        # set path to location for persistence
        mlflow.sklearn.save_model(
            self.clf, path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
