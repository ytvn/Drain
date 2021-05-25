from tensorflow.keras import optimizers
from tensorflow.python.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

from utils.algos.base_algos import SequentialMl
from utils.dataset import Dataset, SequenceDataset
from utils.preprocess import AnomalyDetection
import numpy as np
import pandas as pd
import os
import glob
import re

class Sequential(SequentialMl):

    def __init__(self, dataset, dataset_name, epochs=10, batch=64, lr=0.1):
        super().__init__(dataset, dataset_name=dataset_name, epochs=epochs, batch=batch, lr=lr)


    def compile_sequential(self):
        adam = optimizers.Adam(self.lr)
        self.sequential.compile(loss="mse",
                                optimizer=adam, metrics=['accuracy'])


class LSTMAutoencoder(Sequential):

    def __init__(self, dataset: Dataset, dataset_name, epochs=10, batch=64, lr=0.1):
        super().__init__(dataset=dataset, dataset_name=dataset_name, epochs=epochs, batch=batch, lr=lr)

    def init(self):
        super().init()
        timesteps, n_features = self.dataset.input_shape()

        self.sequential.add(
            LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        self.sequential.add(LSTM(16, activation='relu', return_sequences=False))
        self.sequential.add(RepeatVector(timesteps))
        # Decoder
        self.sequential.add(LSTM(16, activation='relu', return_sequences=True))
        self.sequential.add(LSTM(32, activation='relu', return_sequences=True))
        self.sequential.add(TimeDistributed(Dense(n_features)))
        self.compile_sequential()




    def name(self):
        return "LSTM Autoencoder"

def clear_output_file():
    csvs = glob.glob('./../../results/**/*.csv', recursive=True)
    pngs = glob.glob('./../../results/**/*.png', recursive=True)
    for file in csvs + pngs:
        os.remove(file)

if __name__ == '__main__':
    clear_output_file()
    names = []
    full_names = []
    regex = r"Anomalies\\((.*?)\.csv)"
    for name in glob.glob('./../../data/Anomalies/*'):
        result = re.search(regex, name)
        if result:
            names.append(result.group(2))
            full_names.append(result.group(1))

    for i in range(len(full_names)):
        path = ""
        for step in range(3, 5):
            # Load and preprocess data
            preprocess_data = AnomalyDetection(data_path="./../../data/Anomalies/" + full_names[i], seq_size=step, threshold_label=50)
            preprocess_data.preprocess()

            #Create an instant of SequenceDataset which already have some helpfull method. Example: load_test()..
            dataset = SequenceDataset(preprocess_data)

            #Create an instant of model which include method like: train(), predict()..
            model = LSTMAutoencoder(dataset=dataset, dataset_name=names[i]+"/step" + str(step), epochs=20, batch=64, lr=0.1 )
            preprocess_data.plot_data(data=preprocess_data.raw_data, title=names[i], path=model.pwd + model.image_path(names[i]))
            # model.train()
            model.load_model()
            history = model.load_history()
            model.plot_hist(history)
            print("Model predict: \n", model.predict())

            error_df = model.evaluation_metric(dataset.X_test, dataset.y_test)

            model.ROC(error_df=error_df)
            threshold_rt = model.precision_recal_curve(error_df)

            max_accuracy, thres = model.best_accuracy(error_df=error_df, threshold_rt=threshold_rt)

            model.confusion_matric(thres, error_df)
            model.reconstruction_error_for_2_class(thres, error_df)

            d = {"DatasetName": names[i], "Step": step, "Threshold": thres, "Accuracy": max_accuracy}
            df = pd.DataFrame([d])
            path = "./../../results/" + names[i] + "/accuracies.csv"
            if os.path.exists(path):
                df.to_csv(path, mode="a", header=False)
            else:
                df.to_csv(path)
        df = pd.read_csv(path)
        df = df.sort_values(by=['Accuracy'], ascending=False)
        df.to_csv(path)
        break




