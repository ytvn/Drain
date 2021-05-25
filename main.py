
import matplotlib as mpl
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from keras.models import load_model
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow import keras
import math
from pylab import rcParams
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(7)

xls = pd.ExcelFile('data/Data set for Anomaly tagging Kim (1_0).xlsx')

sheet_names = xls.sheet_names
print(sheet_names)

anomalies_data = []
good_data = []
for i in range(1, 9):
  anomalies_data.append(pd.read_excel(xls, sheet_names[i]))
for i in range(10, 14):
  good_data.append(pd.read_excel(xls, sheet_names[i]))
print(len(good_data))
len(anomalies_data)


def handle_time(time):
  return datetime.strptime(time, '%d/%m/%Y (%H:%M)')


plt.figure(figsize=(25, 10))
for i in range(1, len(anomalies_data)+1):
  x = np.array(anomalies_data[i-1]['Date'].apply(lambda x: handle_time(x)))
  y = np.array(anomalies_data[i-1]['Level'])
  ax = plt.subplot(3, 3, i)
  plt.plot(x, y)
  plt.title('Device ID:' + sheet_names[i])
  date_format = mpl.dates.DateFormatter("%m/%d")
  ax.xaxis.set_major_formatter(date_format)


def plot_anomalies():
    for i in range(1, len(anomalies_data)+1):
        fig = plt.figure(figsize=(25, 9))
        x = np.array(anomalies_data[i-1]
                     ['Date'].apply(lambda x: handle_time(x)))
        y = np.array(anomalies_data[i-1]['Level'])
        ax = plt.subplot(111)
        plt.plot(x, y)
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel("Level")
        plt.title('Device ID:' + sheet_names[i])
        date_format = mpl.dates.DateFormatter("%m/%d %H:%M")
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        plt.savefig('Image/Anomalies/DeviceID:' + sheet_names[i])
        plt.show()


plt.figure(figsize=(25, 10))
for i in range(1, len(good_data) + 1):
  x = np.array(good_data[i-1]['Date'].apply(lambda x: handle_time(x)))
  y = np.array(good_data[i-1]['Level'])
  ax = plt.subplot(3, 3, i)
  plt.plot(x, y)
  plt.title('Device ID:' + sheet_names[i])
  date_format = mpl.dates.DateFormatter("%m/%d")
  ax.xaxis.set_major_formatter(date_format)

  # plt.subplot(3,3, i)
  # plt.plot(good_data[i-1]['Level'])
  # plt.title('Device ID:' +sheet_names[i])


def plot_goods():
    for i in range(1, len(good_data)+1):
        fig = plt.figure(figsize=(25, 9))
        x = np.array(good_data[i-1]['Date'].apply(lambda x: handel_time(x)))
        y = np.array(good_data[i-1]['Level'])
        ax = plt.subplot(111)
        plt.plot(x, y)
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel("Level")
        plt.title('Device ID:' + sheet_names[i])
        date_format = mpl.dates.DateFormatter("%m/%d %H:%M")
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        plt.savefig('Image/Good/DeviceID:' + sheet_names[i])
        plt.show()


def handle_time(time):
  return datetime.strptime(time, '%d/%m/%Y (%H:%M)')


def preprocessing_data(df):
  ## Eliminate null value
  df = df[['Date', 'Level']]
#   print("Null statistics\n", df.isnull().sum(axis=0))
  df = df[df['Level'].notnull()]
  ## Reformat datetime
  df['Date'] = df.apply(lambda x: handle_time(x['Date']), axis=1)
  return df


def plot(df):
    plt.figure(figsize=(20, 6))
    x = np.array(df['Date'])
    y = np.array(df['Level'])
    plt.plot(x, y)
    date_format = mpl.dates.DateFormatter("%m/%d")
    ax.xaxis.set_major_formatter(date_format)


def labeling(df, threshold, start='2000-10-30 8:45:00'):
    df = preprocessing_data(df)
    df = df.loc[df.Date > start]
    df = df.sort_values(by=['Date'], ascending=True)
    df.reset_index(drop=True, inplace=True)

    df['label'] = np.where(df.Level > threshold, 1, 0)
    # train_index = math.floor(df.shape[0]*0.8)
    # train, test = df[: train_index], df[train_index:]

    # display(df)
    # plot(df)
    return df


# df = labeling(anomalies_data[1], 50,  '2020-10-30 8:45:00')
print(anomalies_data[1].shape)
df = labeling(anomalies_data[4], 50)

dft = labeling(anomalies_data[4], 50)
dft['Date'] = dft['Date'].astype('datetime64[ns]')
dft['year'] = dft['Date'].dt.year
dft['month'] = dft['Date'].dt.month
dft['day'] = dft['Date'].dt.day

dfc = dft.copy()
dfc = dfc.groupby(['year', 'month', 'day'])['label'].count().reset_index()
dfc.rename(columns={'label': 'count'}, inplace=True)

dfm = dft.merge(dfc, on=['year', 'month', 'day'])
print(dfm['count'].value_counts())
# type(dfm['label'][0])
dfm = dfm.loc[dfm['count'] == 2]
# df = dfm # Important


df_class_0 = df[df['label'] == 0]
df_class_1 = df[df['label'] == 1]

print(df.label.value_counts())
df.label.value_counts().plot(kind='bar', title='Count (label)')
plt.show()
count_class_0, count_class_1 = df.label.value_counts()
count = math.floor((count_class_0 - count_class_1)/2) + count_class_1

df_class_0_under = df_class_0.sample(count)
df_class_1_over = df_class_1.sample(count, replace=True)

df_test_under = pd.concat([df_class_1_over, df_class_0], axis=0)

print('Random under-sampling:')
print(df_test_under.label.value_counts())

df_test_under.label.value_counts().plot(kind='bar', title='Count (label)')

df_test_under = df_test_under.sort_values(by=['Date'], ascending=True)
plot(df_test_under)
df = df_test_under

scaler = StandardScaler()
scaler = scaler.fit(df[['Level']])

df['Level'] = scaler.transform(df[['Level']])
df['Level'].shape

seq_size = 4


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size-1])

    return np.array(x_values), np.array(y_values)


output_X, output_Y = to_sequences(df[['Level']], df['label'], seq_size)

output_X.shape, output_Y.shape

# print(np.where(output_Y==1))
print(pd.DataFrame(np.concatenate(
    output_X[np.where(np.array(output_Y) == 1)[0][0]], axis=0)))


X_train, X_test, y_train, y_test = train_test_split(
    np.array(output_X), np.array(output_Y), test_size=0.2,  random_state=123)
X_train, X_valid, y_train, y_valid = train_test_split(
    np.array(X_train), np.array(y_train), test_size=0.2,  random_state=123)
# X_train, X_test, y_train, y_test = train_test_split(np.array(output_X), np.array(output_Y), test_size=0.2,  shuffle=False)
# X_train, X_valid, y_train, y_valid  = train_test_split(np.array(X_train), np.array(y_train), test_size=0.2,  shuffle=False)
X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_valid.shape

X_train_y0 = X_train[y_train == 0]
y_train_y0 = y_train[y_train == 0]
X_train_y1 = X_train[y_train == 1]
X_valid_y0 = X_valid[y_valid == 0]
y_valid_y0 = y_valid[y_valid == 0]
X_valid_y1 = X_valid[y_valid == 1]
X_train_y0.shape, y_train_y0.shape, X_valid_y0.shape, y_valid_y0.shape, X_test.shape

test_len = df.shape[0] - X_train.shape[0] - X_valid.shape[0]
test_len

timesteps = X_train_y0.shape[1]  # equal to the lookback
n_features = X_train_y0.shape[2]  # 59

epochs = 50
batch = 64
lr = 0.1


lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(
    timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

lstm_autoencoder.summary()

adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                     save_best_only=True,
                     verbose=0)

tb = TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True)

lstm_autoencoder_history = lstm_autoencoder.fit(X_train_y0, y_train_y0,
                                                epochs=epochs,
                                                batch_size=batch,
                                                validation_data=(
                                                    X_valid_y0, y_valid_y0),
                                                verbose=2,
                                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')]).history
# lstm_autoencoder_history = lstm_autoencoder.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

plt.plot(lstm_autoencoder_history['loss'][:], linewidth=2, label='Train')
plt.plot(lstm_autoencoder_history['val_loss'][:], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

X_train.shape

valid_x_predictions = lstm_autoencoder.predict(X_test)
mse = np.mean(np.power((X_test) - (valid_x_predictions), 2),
              axis=1).reshape(-1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_test.tolist()})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(
    error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

test_x_predictions = lstm_autoencoder.predict(X_test)
mse = np.mean(np.power((X_test) - (test_x_predictions), 2), axis=1).reshape(-1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_test.tolist()})
threshold_fixed = 0.8
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[
          1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

LABELS = ["Normal", "Anomaly"]
pred_y = [
    1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS,
            yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

conf_matrix


def acc(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/y_true.shape[0]


print("Accuracy: ", acc(error_df.True_class.values, pred_y))

false_pos_rate, true_pos_rate, thresholds = roc_curve(
    error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate,
         linewidth=5, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# model = load_model('Output/model.h5')
model = lstm_autoencoder
