import time
# from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
#                              ReduceLROnPlateau)
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.python.keras.callbacks import CSVLogger

from tensorflow.python.keras.models import Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from utils.dataset import Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import os


class BaseAlgos(object):

    def __init__(self, dataset: Dataset, dataset_name):
        super().__init__()
        self.dataset = dataset
        self.pwd = "./"
        # self.pwd = "./../../"
        self.dataset_name = dataset_name
        self.prepair_output_path()
        self.init()

    def init(self):
        raise Exception('Not yet defined')

    def name(self):
        raise Exception("Not yet defined")

    def model_name(self):
        raise Exception("Not yet defined")

    def model_path(self):
        return "results/%s/%s-results/%s-%s" % (self.dataset_name, self.name(), self.name(), self.model_name())

    def history_path(self):
        return "results/%s/%s-results/%s-history.npy" % (self.dataset_name, self.name(), self.name())

    def root_path(self):
        return "results/%s/%s-results/" % (self.dataset_name, self.name())

    def image_path(self, name):
        return "results/%s/%s-results/images/%s.png" % (self.dataset_name, self.name(), name)

    def prepair_output_path(self):
        directory = self.pwd + "results/%s/%s-results/images/" % (self.dataset_name, self.name())

        if not os.path.exists(directory):
            os.makedirs(directory)

    def train(self):
        start_time = time.time()
        self._train()
        end_time = time.time()
        return start_time, end_time, end_time - start_time

    def _train(self) -> float:
        raise Exception("Not yet defined")

    def save_model(self):
        raise Exception("Not yet defined")

    def load_model(self):
        raise Exception("Not yet defined")

    def compile_sequential(self):
        raise Exception("Not yet defined")

    def predict(self):
        start_time = time.time()
        accuracy = self._predict()
        end_time = time.time()
        return start_time, end_time, end_time - start_time, accuracy

    def _predict(self) -> float:
        raise Exception("Not yet defined")


class SequentialMl(BaseAlgos):

    def __init__(self, dataset, dataset_name, epochs=10, batch=64, lr=0.1):
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        super().__init__(dataset, dataset_name)
        self.history = None

    def init(self):
        self.sequential = Sequential()

    def _predict(self) -> float:
        X_test, y_test = self.dataset.load_test()
        loss, accuracy = self.sequential.evaluate(X_test, y_test)
        print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
        return accuracy

    def load_model(self):
        self.sequential = load_model(self.pwd + self.model_path())
        # self.sequential.load_weights(self.pwd + self.model_path())
        # self.compile_sequential()

    def save_history(self, history):
        np.save(self.pwd + self.history_path(), history)

    def load_history(self):
        return np.load(self.pwd + self.history_path(), allow_pickle=True).item()

    def _train(self):
        X_train, y_train, X_test, y_test = self.dataset.load()
        X_train_y0 = X_train[y_train==0]
        y_train_y0 = y_train[y_train==0]

        self.history = self.sequential.fit(X_train_y0, y_train_y0,
                            epochs=self.epochs,
                            batch_size=self.batch,
                            validation_split=0.2,
                            verbose=2,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')]).history
        self.save_model()
        self.save_history(self.history)

    def save_model(self):
        self.sequential.save(self.pwd +self.model_path())

    def callbacks(self):
        checkpointer = callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path(), verbose=1, save_best_only=True, monitor='val_acc', mode='max')
        csv_logger = CSVLogger(self.logger_path(), separator=',', append=False)
        return [checkpointer, csv_logger]

    def checkpoint_path(self):
        return "results/%s/%s-results/checkpoint-{epoch:02d}.hdf5" % (self.dataset_name,self.name())

    def logger_path(self):
        return "results/%s/%s-results/%s-train-analysis.csv" % (self.dataset_name, self.name(), self.name())

    def model_name(self):
        return "model.hdf5"

    def plot_hist(self, history=None, show=False):
        if history == None:
            history = self.history
        plt.plot(history['loss'][:], linewidth=2, label='Train')
        plt.plot(history['val_loss'][:], linewidth=2, label='Valid')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        if show:
            plt.show()
        plt.close()

    def evaluation_metric(self, X, y):
        x_predictions = self.sequential.predict(X)
        mse = np.mean(np.power((X) - (x_predictions), 2), axis=1).reshape(-1)

        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                 'True_class': y.tolist()})
        return error_df

    def predict_y(self, error_df, threshold_fixed):
        pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
        return pred_y

    def precision_recal_curve(self, error_df, show=False):
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class,
                                                                       error_df.Reconstruction_error)
        plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
        plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.savefig(self.pwd + self.image_path("precision_recall_curve"))
        if show:
            plt.show()
        plt.close()
        return threshold_rt

    def ROC(self, error_df, show=False):
        '''
            Receiver operating characteristic curve
        :return:
        '''
        false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
        roc_auc = auc(false_pos_rate, true_pos_rate, )

        plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], linewidth=5)

        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Receiver operating characteristic curve (ROC)')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(self.pwd + self.image_path("ROC"))
        if show:
            plt.show()
        plt.close()

    def reconstruction_error_for_2_class(self, threshold_fixed, error_df, show=False):
        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()

        for name, group in groups:
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="Break" if name == 1 else "Normal")
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(self.pwd + self.image_path("reconstruction_error_for_2_class_" + str(threshold_fixed)))
        if show:
            plt.show()
        plt.close()

    def confusion_matric(self, threshold_fixed, error_df, show=False):
        LABELS = ["Normal", "Anomaly"]
        pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.True_class, pred_y)

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.savefig(self.pwd + self.image_path("confusion_matric_" + str(threshold_fixed)))
        if show:
            plt.show()
        plt.close()

    def accuracy(self, y_true, y_pred, threshold=""):
        correct = np.sum(y_true == y_pred)
        accuracy = float(correct) / y_true.shape[0]
        return accuracy

    def accuracy_to_csv(self, thresholds, accuracies):
        df = pd.DataFrame(list(zip(thresholds, accuracies)), columns=['threshold', 'accuracy'])
        df = df.sort_values(by=['accuracy'], ascending=False)
        path = self.pwd + self.root_path() + "accuracy.csv"
        df.to_csv(path)

    def best_accuracy(self, error_df, threshold_rt):
        accuracies = np.array([])
        save_threshold, save_accuracy = [], []
        for thres in threshold_rt:
            y_pred = self.predict_y(error_df=error_df, threshold_fixed=thres)
            acc = self.accuracy(error_df.True_class.values, y_pred, thres)
            save_accuracy.append(acc)
            save_threshold.append(thres)
            accuracies = np.append(accuracies, acc)

        self.accuracy_to_csv(save_threshold, save_accuracy)

        max_accuracy = np.max(accuracies)
        thres = threshold_rt[np.where(accuracies == max_accuracy)[0][0]]
        return max_accuracy, thres




