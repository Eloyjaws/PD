import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import logging


class MLFlowUtils:
    @staticmethod
    def yield_artifacts(run_id, path=None):
        """Yield all artifacts in the specified run"""
        client = mlflow.tracking.MlflowClient()
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                yield from MLFlowUtils.yield_artifacts(run_id, item.path)
            else:
                yield item.path

    @staticmethod
    def fetch_logged_data(run_id):
        """Fetch params, metrics, tags, and artifacts in the specified run"""
        client = mlflow.tracking.MlflowClient()
        data = client.get_run(run_id).data
        # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
        tags = {k: v for k, v in data.tags.items(
        ) if not k.startswith("mlflow.")}
        artifacts = list(MLFlowUtils.yield_artifacts(run_id))
        return {
            "params": data.params,
            "metrics": data.metrics,
            "tags": tags,
            "artifacts": artifacts,
        }


class Utils:
    @staticmethod
    def load_data(dataset_name):
        df = pd.read_csv(
            f"data/extracted_features/{dataset_name}/All_features_v2.csv")
        df = df.dropna()
        df.drop('voiceID', inplace=True, axis=1)
        return df

    @staticmethod
    def get_train_test_data(df):
        # Separate dependent and independent variable
        df_X = df.iloc[:, :-1]
        df_Y = df.iloc[:, -1]
        # Split the dataset into the Training set and Test set
        return train_test_split(df_X, df_Y, test_size=0.3, random_state=0, stratify=df_Y)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Borrowed from the scikit-learn library documentation

        :param y_true: the actual value of y
        :param y_pred: the predicted valuye of y
        :param classes: list of label classes to be predicted
        :param normalize: normalize the data
        :param title: title of the plot for confusion matrix
        :param cmap: color of plot
        :return: returns a tuple of (plt, fig, ax)
        """

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            logging.info("Normalized confusion matrix")
        else:
            logging.info('Confusion matrix, without normalization')
        logging.info(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return (plt, fig, ax)

    @staticmethod
    def get_metrics_from_confusion_matrix(conf_matrix):
        """
        Given a Confusion Matrix, return the classification metrics
        :param conf_matrix
        :return: (accuracy, sensitivity, specificity, precision, f1_score)
        """
        try:

            TN, FP, FN, TP = conf_matrix.ravel()

            accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
            sensitivity = (TP/(TP+FN)) * 100  # recall
            specificity = (TN/(TN + FP)) * 100
            precision = (TP/(TP+FP)) * 100
            f1_score = 2 * ((sensitivity * precision) /
                            (sensitivity + precision))

            return (accuracy, sensitivity, specificity, precision, f1_score)
        except Exception as e:
            logging.info(f"Err: failed to extract metrics from confusion matrix \n{e}")
            return (0, 0, 0, 0, 0)

    @staticmethod
    def print_aggregated_KFold_metric(data, title, K=4):
        columns = ['Loops'] + \
            [f"fold {i}" for i in range(1, K+1)] + [f"mean {title}"]
        data = pd.DataFrame(list(data), columns=columns)
        logging.info(f"\n\n{title} \n{data}")
