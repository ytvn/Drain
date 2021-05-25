import glob
import re
from utils.algos.sequential import LSTMAutoencoder
from utils.dataset import SequenceDataset
from utils.preprocess import AnomalyDetection
import pandas as pd
import os

def clear_output_file():
    csvs = glob.glob('./results/**/*.csv', recursive=True)
    pngs = glob.glob('./results/**/*.png', recursive=True)
    for file in csvs + pngs:
        os.remove(file)

if __name__ == '__main__':

    clear_output_file()

    names = []
    full_names = []
    regex = r"Anomalies[\\|\/]((.*?)\.csv)"
    for name in glob.glob('./data/Anomalies/*'):
        result = re.search(regex, name)
        if result:
            names.append(result.group(2))
            full_names.append(result.group(1))
    print(full_names)
    for i in range(len(full_names)):
        path = ""
        df = pd.DataFrame()
        for step in range(0, 77):

            # Load and preprocess data
            preprocess_data = AnomalyDetection(data_path="./data/Anomalies/" + full_names[i], seq_size=step,
                                               threshold_label=50)
            preprocess_data.preprocess()

            # Create an instant of SequenceDataset which already have some helpfull method. Example: load_test()..
            dataset = SequenceDataset(preprocess_data)

            # Create an instant of model which include method like: train(), predict()..
            model = LSTMAutoencoder(dataset=dataset, dataset_name=names[i] + "/step" + str(step), epochs=20, batch=64,
                                    lr=0.1)
            preprocess_data.plot_data(data=preprocess_data.raw_data, title=names[i],
                                      path=model.pwd + model.image_path(names[i]))
            _, _, time_exe = model.train()
            print("Model train in %s seconds" % (str(time_exe)))

            # model.load_model()
            history = model.load_history()
            model.plot_hist(history)
            model.predict()

            error_df: pd.DataFrame = model.evaluation_metric(dataset.X_test, dataset.y_test)
            error_df.dropna(inplace=True)
            print(error_df.shape)
            if error_df.shape[1] < 10:
                continue

            model.ROC(error_df=error_df)
            threshold_rt = model.precision_recal_curve(error_df)

            max_accuracy, thres = model.best_accuracy(error_df=error_df, threshold_rt=threshold_rt)

            model.confusion_matric(thres, error_df)
            model.reconstruction_error_for_2_class(thres, error_df)

            d = {"DatasetName": names[i], "Step": step, "Threshold": thres, "Accuracy": max_accuracy}
            df = pd.DataFrame([d])
            path = "./results/" + names[i] + "/accuracies.csv"
            if os.path.exists(path):
                df.to_csv(path, mode="a", header=False)
            else:
                df.to_csv(path)

        if path != "":
            df = pd.read_csv(path)
            df = df.sort_values(by=['Accuracy'], ascending=False)
            df.to_csv()