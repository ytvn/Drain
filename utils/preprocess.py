from typing import Union
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import math
import matplotlib as mpl

class PreProcess(object):
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path
        self.X: Union[TextFileReader, DataFrame] = None
        self.Y: Union[TextFileReader, DataFrame] = None
        self.C: Union[TextFileReader, DataFrame] = None
        self.T: Union[TextFileReader, DataFrame] = None
        self.sum_of_labels: int = None
        self.test_size: float = 0.2
        self.random_state: int = 123

    def preprocess(self):
        input_data, output_data = self._preprocess()
        # X_train, X_test, y_train, y_test
        self.X, self.T, self.Y, self.C = train_test_split(input_data, output_data, test_size=self.test_size, random_state=self.random_state)

    def processed_data(self):
        return self.X, self.Y, self.T, self.C

    def _preprocess(self) -> (DataFrame, DataFrame):
        raise("Not yet defined")

    def name(self):
        raise ("Not yet defined")

    def scaler(self, data: DataFrame):
        scaler = StandardScaler().fit(data)
        return scaler.transform(data)

    def plot_data(self, data, title, path):
        fig = plt.figure(figsize=(10, 5))
        x = np.array(data['Date'].apply(lambda x: self.handle_time(x)))
        y = np.array(data['Level'])
        ax = fig.add_subplot(111)
        plt.plot(x, y)
        plt.title('Device ID:' + title)
        date_format = mpl.dates.DateFormatter("%m/%d")
        ax.xaxis.set_major_formatter(date_format)
        plt.savefig(path)
        plt.close()

    def handle_time(self, time):
        return datetime.strptime(time, '%d/%m/%Y (%H:%M)')

class AnomalyDetection(PreProcess):
    def __init__(self, data_path: str, seq_size: int, threshold_label: int):
        super().__init__(data_path)
        self.seq_size = seq_size
        self.threshold_label = threshold_label
        self.raw_data = None

    def _preprocess(self) -> (DataFrame, DataFrame):
        full_data: DataFrame = pd.read_csv(self.data_path)
        self.raw_data = full_data.copy()
        full_data = self.drop_unuse_columns(full_data)
        full_data = self.labeling(full_data, self.threshold_label)

        full_data = self.resampling_label(full_data)
        full_data['Level'] = self.scaler(full_data[['Level']])
        input_data, output_data = self.to_senquences(full_data[['Level']], full_data['Label'], self.seq_size)
        return input_data, output_data

    def to_senquences(self, x, y, seq_size=1):
        x_values, y_values = [], []
        for i in range(len(x) - seq_size):
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i + seq_size - 1])
        return np.array(x_values), np.array(y_values)


    def resampling_label(self, df):
        df_class_0, df_class_1 = df[df['Label'] == 0],  df[df['Label'] == 1]

        df.Label.value_counts().plot(kind='bar', title='Count (label) before resampling')
        plt.show()

        count_class_0, count_class_1 = df.Label.value_counts()
        count = math.floor((count_class_0 - count_class_1) / 2) + count_class_1

        df_class_1_over = df_class_1.sample(count, replace=True)

        df_resampled = pd.concat([df_class_1_over, df_class_0], axis=0)
        df_resampled = df_resampled.sort_values(by=['Date'], ascending=True)
        df_resampled.Label.value_counts().plot(kind='bar', title='Count (label) after resampling')
        plt.show()
        return df_resampled

    def labeling(self, df, threshold, start='2000-10-30 8:45:00'):
        df = df.loc[df.Date > start]
        df = df.sort_values(by=['Date'], ascending=True)
        df.reset_index(drop=True, inplace=True)

        df['Label'] = np.where(df.Level > threshold, 1, 0)
        return df

    def drop_unuse_columns(self, df):
        df = df[['Date', 'Level']]
        df = df[df['Level'].notnull()]
        df['Date'] = df.apply(lambda x: self.handle_time(x['Date']), axis=1)
        return df

    def handle_time(self, time):
        return datetime.strptime(time, '%d/%m/%Y (%H:%M)')

if __name__ =="__main__":
    dataset = AnomalyDetection(data_path="./../../data/Anomalies/1D82C4C.csv", seq_size=5, threshold_label=50)
    # dataset.preprocess()
    # print(dataset.X[0], dataset.Y[0])




