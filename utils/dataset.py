from sklearn.preprocessing import StandardScaler

from utils.preprocess import PreProcess, AnomalyDetection
import numpy as np

class Dataset(object):
    def __init__(self, data: PreProcess):
        super().__init__()

        self.data: PreProcess = data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.process_dataset()

    def preprocess(self):
        return self.data.processed_data()

    def process_dataset(self):
        raise("Not yet difined")

    def load(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def load_train(self):
        return self.X_train, self.y_train

    def load_test(self):
        return self.X_test, self.y_test

    def input_shape(self):
        return self.X_train.shape[1], self.X_train.shape[2]

class SequenceDataset(Dataset):
    def __init__(self, data: PreProcess):
        super().__init__(data=data)

    def process_dataset(self):
        X, Y, T, C = self.preprocess()
        self.X_train = np.array(X)
        self.y_train = np.array(Y)
        self.X_test = np.array(T)
        self.y_test = np.array(C)


class ClassicDataset(Dataset):
    def __init__(self, data: PreProcess):
        super().__init__(data=data)

    def process_dataset(self):
        X, Y, T, C = self.preprocess()

        scaler = StandardScaler().fit(X)
        trainX = scaler.transform(X)

        scaler = StandardScaler().fit(T)
        testT = scaler.transform(T)

        self.X_train = np.array(trainX)
        self.y_train = np.array(Y)
        self.X_test = np.array(testT)
        self.y_test = np.array(C)

if __name__ =="__main__":
    preprocess_data = AnomalyDetection(data_path="../data/Anomalies/1D82C4C.csv", seq_size=5, threshold_label=50)
    preprocess_data.preprocess()

    dataset = SequenceDataset(preprocess_data)
    print(dataset.input_shape())



