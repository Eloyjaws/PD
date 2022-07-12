import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import confusion_matrix
import mlflow
import multiprocessing as mp

import torch
import pytorch_tabnet

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Modelling.utils import Utils  # noqa
from utils.timer import start_timer, end_timer_and_print, log  # noqa

grid_S = {
    "n_a": [8],                     
    "n_independent": [2],           
    "n_shared": [2, 4],                
    "n_steps": [3, 6, 8],                   
    "gamma": [1.0, 1.3],   
    "device_name": ["cpu"],                
    "verbose": [0]
}

grid_L = {
    "n_a": [8, 16, 24, 32],                         # default = 8       <8-64>
    "n_independent": [1, 2, 3, 4, 5],               # default = 2       <1-5>
    "n_shared": [1, 2, 3, 4, 5],                    # default = 2       <1-5>
    "n_steps": [3, 6, 8, 10],                       # default = 3       <3-10>
    "gamma": [1.0, 1.3, 2.0],                       # default = 1.3     <1.0-2.0>
    "momentum": [0.4, 0.1, 0.05, 0.02, 0.005],      # default = 0.02    <0.01-0.4>
    "lambda_sparse": [0.1, 0.01, 0.001, 0.0001],    # default = 0.001   
    "optimizer_params": [
        {'lr': 0.01}, 
        {'lr': 0.02}, 
        {'lr': 0.001}],
    "verbose": [0]
}

# The last metric in the eval_metric parameter list is used for early stopping.
class F1_Score(Metric):
    def __init__(self):
        self._name = "f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        score = f1_score(y_true, (y_score[:, 1]>0.5)*1)
        return score


class TabNet_Model():
    def __init__(self):
        """
        Constructor for TabNet_Model
        :param params: all hyperparameter options
        """
        self.clf = TabNetClassifier()
        self.model_class = "TabNet"

        # Default hyperparams
        self.hyperparams = {'verbose': 0}
        self.tuned = False

        # globals for multiprocessing
        manager = mp.Manager()

        self.best_accuracy = 0
        self.X_kfold = None
        self.y_kfold = None
        self.k_accuracy_list = manager.list()
        self.k_specificity = manager.list()
        self.k_sensitivity = manager.list()
        self.k_precision = manager.list()
        self.k_f1 = manager.list()



    def tune_hyperparameters(self, df, run_name):
        ######## Begin Hyperparameter tuning for Classifier ####################
        event_name = f"Hyperparameter tuning {run_name}"
        log(event_name)
        start_timer(event_name)
        (X_train, X_test, y_train, y_test) = Utils.get_train_test_data(df)
        
        # (accuracy, F1, AUC)
        best_scores = (0, 0, 0)
        for hyperparams in ParameterGrid(grid_S):
            model = TabNetClassifier(**hyperparams)
            model.fit(
                X_train.values, y_train.values,
                eval_set=[(X_train.values, y_train.values), (X_test.values, y_test.values)],
                eval_name=['train', 'val'],
                eval_metric=['accuracy', F1_Score, 'auc'], 
                patience=100,
                max_epochs=1440,
                batch_size=1024,
            )
        
            y1,  y1_hat = y_train.values, model.predict(X_train.values)
            y2,  y2_hat = y_test.values, model.predict(X_test.values)

            # print(f"TRAIN | Accuracy: {accuracy_score(y1, y1_hat)}      |       F1: {f1_score(y1, y1_hat)}      |       AUC: {roc_auc_score(y1, y1_hat)}")
            # print(f"TEST  | Accuracy: {accuracy_score(y2, y2_hat)}      |       F1: {f1_score(y2, y2_hat)}      |       AUC: {roc_auc_score(y2, y2_hat)}")
            # print(f"Hyperparameters | {hyperparams}")
            
            if(best_scores[1] < f1_score(y2, y2_hat)):
                best_scores = (accuracy_score(y2, y2_hat), f1_score(y2, y2_hat), roc_auc_score(y2, y2_hat))
                self.hyperparams = hyperparams
                self.tuned = True
        
        log(f"Best Hyperparameters for {run_name} {self.hyperparams}")
        log(f"Best Scores || Accuracy: {accuracy_score(y2, y2_hat)}      |       F1: {f1_score(y2, y2_hat)}      |       AUC: {roc_auc_score(y2, y2_hat)}")
        end_timer_and_print(event_name)
        ######## End Hyperparameter tuning for Classifier ####################


    def training_loop(self, run_name, i, K):
        event_name = f"Loop {i} for {run_name}"
        log(event_name)
        start_timer(event_name)

        kfold = StratifiedKFold(K, shuffle=True, random_state=i)
        row, row_specificity, row_sensitivity, row_precision, row_f1 = [], [], [], [], []

        row.append(i)
        row_specificity.append(i)
        row_sensitivity.append(i)
        row_precision.append(i)
        row_f1.append(i)

        for train, test in kfold.split(self.X_kfold, self.y_kfold):
            Xtrain_kfold = self.X_kfold.iloc[train, :]
            Ytrain_kfold = self.y_kfold.iloc[train, :]
            Xtest_kfold = self.X_kfold.iloc[test, :]
            Ytest_kfold = self.y_kfold.iloc[test, :]


            model = TabNetClassifier(**self.hyperparams)
            model.fit(
                Xtrain_kfold.values, Ytrain_kfold.values.ravel(),
                eval_set=[(Xtrain_kfold.values, Ytrain_kfold.values.ravel()), (Xtest_kfold.values, Ytest_kfold.values.ravel())],
                eval_name=['train', 'val'],
                eval_metric=['accuracy', F1_Score, 'auc'], 
                patience=100,
                max_epochs=1440,
                batch_size=1024,
            )

            
            y_hat =  model.predict(Xtest_kfold.values)

            conf_matrix_kfold = confusion_matrix(
                Ytest_kfold, y_hat)    

            (accuracy, sensitivity, specificity, precision,
                f1_score_) = Utils.get_metrics_from_confusion_matrix(conf_matrix_kfold)

            row.append(accuracy)
            row_specificity.append(specificity)
            row_sensitivity.append(sensitivity)
            row_precision.append(precision)
            row_f1.append(f1_score_)

            if(accuracy > self.best_accuracy):
                self.best_accuracy = accuracy
                self.clf = model

        # Add average across K folds for run i
        row.append(np.nanmean(row[1:]))
        row_specificity.append(np.nanmean(row_specificity[1:]))
        row_sensitivity.append(np.nanmean(row_sensitivity[1:]))
        row_precision.append(np.nanmean(row_precision[1:]))
        row_f1.append(np.nanmean(row_f1[1:]))

        # collate metrics for run No i
        self.k_accuracy_list.append(row)
        self.k_specificity.append(row_specificity)
        self.k_sensitivity.append(row_sensitivity)
        self.k_precision.append(row_precision)
        self.k_f1.append(row_f1)

        end_timer_and_print(event_name)


    def mlflow_run(self, df, K=5, run_name=f"TabNet Experiment", verbose=True):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param run_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """
        start_timer(run_name)

        self.tune_hyperparameters(df, run_name)
        # TODO: Fix potential deadlock
        # while self.tuned == False:
        #     hyp_jobs = [
        #         mp.Process(
        #             target=self.tune_hyperparameters,
        #             args=(df, run_name)
        #         )]
        #     hyp_jobs[0].start()
        #     hyp_jobs[0].join()
            
        tags = {
            "model_class": self.model_class,
            "dataset_name": run_name.split(" - Dataset: ")[-1],
            }
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags)
            X_kfold = pd.DataFrame(df.iloc[:, :-1].values)
            y_kfold = pd.DataFrame(df.iloc[:, -1].values.ravel())

            self.X_kfold = X_kfold
            self.y_kfold = y_kfold
            
            # Use multiprocessing to compute each run in parallel 
            # Some attempts fail and segfault - loop until we have 12 entries   
            trials = 1
            ctx = mp.get_context('spawn')
            while (len(self.k_accuracy_list) < 12):
                jobs = []
                for i in range(trials, trials+12):
                    jobs.append(
                        ctx.Process(
                            target=self.training_loop,
                            args=(run_name, i, K)
                        )
                    )
                for job in jobs:
                    job.start()
                for job in jobs:
                    job.join()
                trials += 12
                
            # Log model and params using the MLflow sklearn APIs
            # mlflow.sklearn.log_model(self.clf, "TabNet-Model")
            mlflow.log_param('K', K)

            avg_accuracy = np.nanmean(np.array(self.k_accuracy_list)[:12, -1])
            avg_specificity = np.nanmean(np.array(self.k_specificity)[:12, -1])
            avg_sensitivity = np.nanmean(np.array(self.k_sensitivity)[:12, -1])
            avg_precision = np.nanmean(np.array(self.k_precision)[:12, -1])
            avg_f1_score = np.nanmean(np.array(self.k_f1)[:12, -1])
            # log metrics
            mlflow.log_metric("accuracy", avg_accuracy)
            mlflow.log_metric("specificity", avg_specificity)
            mlflow.log_metric("sensitivity", avg_sensitivity)
            mlflow.log_metric("recall", avg_precision)
            mlflow.log_metric("F1-Score", avg_f1_score)

            if verbose:
                log("TabNet Kfold Evaluation")
                Utils.print_aggregated_KFold_metric(
                    self.k_accuracy_list, "accuracy", K)
                Utils.print_aggregated_KFold_metric(
                    self.k_specificity, "specificity", K)
                Utils.print_aggregated_KFold_metric(
                    self.k_sensitivity, "sensitivity/recall", K)
                Utils.print_aggregated_KFold_metric(
                    self.k_precision, "precision", K)
                Utils.print_aggregated_KFold_metric(self.k_f1, "f1 score", K)

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
        raise Exception("Save method for TabNet not implemented")

# TODO: Log images for interpretability to mlflow 
# mlflow.log_image(PIL_or_np_image, filename)
# Global explainability : feat importance summing to 1
# clf.feature_importances_

# Local explainability and masks
# explain_matrix, masks = clf.explain(X_test)
# fig, axs = plt.subplots(1, 3, figsize=(20,20))

# for i in range(3):
#     axs[i].imshow(masks[i][:50])
#     axs[i].set_title(f"mask {i}")