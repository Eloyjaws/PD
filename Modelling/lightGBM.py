import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import mlflow
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Modelling.utils import Utils  # noqa
from utils.timer import start_timer, end_timer_and_print, log  # noqa


class lightGBM_Model():
    def __init__(self, params={
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 127, 255],
        'n_estimators': [100, 200, 500],
    }):
        """
        Constructor for Light GBM Model
        :param params: all hyperparameter options for the constructor
        """
        self.clf = LGBMClassifier(**params)
        self._params = params

        # Default hyperparams
        self.learning_rate = 0.1
        self.num_leaves = 255
        self.n_estimators = 500

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

    def tune_hyperparameters(self, df, run_name):
        ######## Begin Hyperparameter tuning for Classifier ####################
        event_name = f"Hyperparameter tuning {run_name}"
        log(event_name)
        start_timer(event_name)

        (X_train, X_test, y_train, y_test) = Utils.get_train_test_data(df)

        # Apply Min-Max Scaling
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        learning_rate = self._params.get('learning_rate')
        num_leaves = self._params.get('num_leaves')
        n_estimators = self._params.get('n_estimators')

        hyperparameters = dict(
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
        )

        clf = LGBMClassifier()
        clf = GridSearchCV(clf, hyperparameters, refit=True)

        best_model = clf.fit(X_train, y_train)
        self.learning_rate = best_model.best_params_.get('learning_rate')
        self.num_leaves = best_model.best_params_.get('num_leaves')
        self.n_estimators = best_model.best_params_.get('n_estimators')
        
        log(f"Best Hyperparameters for {run_name} {best_model.best_params_}")
        end_timer_and_print(event_name)
        ######## End Hyperparameter tuning for Classifier ####################

    def mlflow_run(self, df, K=6, run_name=f"Light GBM Experiment", verbose=True):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param run_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """
        log(run_name)
        start_timer(run_name)
        self.tune_hyperparameters(df, run_name)

        best_accuracy = 0

        with mlflow.start_run(run_name=run_name) as run:
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

                total, total_specificity, total_sensitivity, total_precision, total_f1 = 0, 0, 0, 0, 0

                for train, test in kfold.split(X_kfold, y_kfold):
                    Xtrain_kfold = X_kfold.iloc[train, :]
                    Ytrain_kfold = y_kfold.iloc[train, :]
                    Xtest_kfold = X_kfold.iloc[test, :]
                    Ytest_kfold = y_kfold.iloc[test, :]

                    Xtrain_kfold = self.scaler.fit_transform(Xtrain_kfold)
                    Xtest_kfold = self.scaler.transform(Xtest_kfold)

                    model_new = LGBMClassifier(
                        learning_rate=self.learning_rate,
                        num_leaves=self.num_leaves,
                        n_estimators=self.n_estimators
                    )

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

                    total += accuracy
                    total_specificity += specificity
                    total_sensitivity += sensitivity
                    total_precision += precision
                    total_f1 += f1_score

                    if(accuracy > best_accuracy):
                        best_accuracy = accuracy
                        self.clf = model_new

                row.append(total/K)
                row_specificity.append(total_specificity/K)
                row_sensitivity.append(total_sensitivity/K)
                row_precision.append(total_precision/K)
                row_f1.append(total_f1/K)

                k_accuracy_list.append(row)
                k_specificity.append(row_specificity)
                k_sensitivity.append(row_sensitivity)
                k_precision.append(row_precision)
                k_f1.append(row_f1)
            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.clf, "light-GBM-model")
            mlflow.log_param('learning_rate', self.learning_rate)
            mlflow.log_param('num_leaves', self.num_leaves)
            mlflow.log_param('n_estimators', self.n_estimators)
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
                log("Light GBM Kfold Evaluation")
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
