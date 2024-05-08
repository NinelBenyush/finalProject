import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf


class prepare_LSTM:
    def __init__(self, data):
        self.data = data
        self.seq = 3
        self.num_features = 1

    def split_data_and_seq(self):
        new_data = np.array(self.data['Value'])
        x_data, y_data = [], []

        for i in range(len(new_data) - self.seq+1):
            x_data.append(new_data[i:i + self.seq])
            y_data.append(new_data[i + self.seq-1])

        x_data, y_data = np.array(x_data), np.array(y_data)

        train_size = int(len(self.data) * 0.8)
        x_train, y_train = x_data[:train_size], y_data[:train_size]
        x_test, y_test = x_data[train_size:], y_data[train_size:]

        return x_train, y_train, x_test, y_test

class LSTM:
    def __init__(self):
        self.num_features = 1
        self.seq = 3
        self.model = self.create_lstm()

    def create_lstm(self):
        model = Sequential([
            LSTM(1, activation='relu', input_shape=(self.seq_length, self.num_features)),
            Dense(1)
        ])
        return model



class main:
    data = pd.read_csv("prediction_data.csv")
    print(tf.__version__)

    month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    data['Month'] = data['Month'].map(month_map)
    final_data = data.set_index("Month")

    prepare_to_model = prepare_LSTM(final_data)
    x_train, y_train, x_test, y_test = prepare_to_model.split_data_and_seq()
    lstm_model = LSTM()


if __name__ == "__main__":
    main()
