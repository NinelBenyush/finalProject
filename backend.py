import pandas as pd


class prepare_LSTM():
    def __init__(self, data):
        self.data = data

    def split_data(self):
        train_size = int(len(self.data) * 0.8)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        return train_data, test_data


class main:
    data = pd.read_csv("prediction_data.csv")

    month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    data['Month'] = data['Month'].map(month_map)
    final_data = data.set_index("Month")

    prepare_to_model = prepare_LSTM(final_data)
    x_train, x_test = prepare_to_model.split_data()
    print(x_test.shape)
    print(x_train.shape)


if __name__ == "__main__":
    main()
