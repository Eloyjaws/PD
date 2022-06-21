import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import mlflow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Modelling.utils import Utils  # noqa


class RF_Model():
    def __init__(self, params={
        'n_estimators': [100, 200],
        'max_features': ['auto'],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 2, 5]
    }):
        """
        Constructor for RF
        :param params: all hyperparameter options for the constructor
        """
        self.clf = RandomForestClassifier(**params, n_jobs=-1)
        self._params = params

        # Default hyperparams
        self.n_estimators = 100
        self.max_features = 'auto'
        self.min_samples_split = 2
        self.min_samples_leaf = 1

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

    def tune_hyperparameters(self, df):
        ######## Begin Hyperparameter tuning for Classifier ####################
        (X_train, X_test, y_train, y_test) = Utils.get_train_test_data(df)

        # Apply Min-Max Scaling
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        n_estimators = self._params.get('n_estimators')
        max_features = self._params.get('max_features')
        min_samples_split = self._params.get('min_samples_split')
        min_samples_leaf = self._params.get('min_samples_leaf')

        hyperparameters = dict(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        clf = RandomForestClassifier()
        clf = GridSearchCV(clf, hyperparameters, refit=True, n_jobs=-1)

        best_model = clf.fit(X_train, y_train)
        self.n_estimators = best_model.best_params_.get('n_estimators')
        self.max_features = best_model.best_params_.get('max_features')
        self.min_samples_split = best_model.best_params_.get(
            'min_samples_split')
        self.min_samples_leaf = best_model.best_params_.get('min_samples_leaf')
        print(best_model.best_params_)
        ######## End Hyperparameter tuning for Classifier ####################

    def mlflow_run(self, df, K=4, run_name=f"SVM Experiment", verbose=True):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param run_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """
        self.tune_hyperparameters(df)

        best_accuracy = 0

        with mlflow.start_run(run_name=run_name) as run:
            kfold = KFold(K, shuffle=True, random_state=None)

            X_kfold = pd.DataFrame(df.iloc[:, :-1].values)
            y_kfold = pd.DataFrame(df.iloc[:, -1].values.ravel())

            k_accuracy_list, k_specificity, k_sensitivity, k_precision, k_f1 = [], [], [], [], []

            for i in range(1, 12):
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

                    model_new = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        n_jobs=-1
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
            mlflow.sklearn.log_model(self.clf, "RF-model")
            mlflow.log_param('n_estimators', self.n_estimators)
            mlflow.log_param('max_features', self.max_features)
            mlflow.log_param('min_samples_split', self.min_samples_split)
            mlflow.log_param('min_samples_leaf', self.min_samples_leaf)
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
                print("RF Kfold Evaluation")
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
            print("Completed MLflow Run with run_id {} and experiment_id {}".format(
                runID, experimentID))
            return (experimentID, runID)

    def save(self, path="."):
        # Save the model in cloudpickle format
        # set path to location for persistence
        mlflow.sklearn.save_model(
            self.clf, path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
