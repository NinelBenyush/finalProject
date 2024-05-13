import pandas as pd
import numpy as np
import sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Formatting dates



class prepare_LSTM:
    def __init__(self, data):
        self.data = data
        self.seq = 3
        self.num_features = 1

    def split_data_and_seq(self):
        train_size = int(len(self.data) * 0.8)
        train_data = self.data[:train_size][['code_p', 'Inventory', 'Value']]
        test_data = self.data[train_size:][['code_p', 'Inventory', 'Value']]
        return train_data, test_data

    def select_features(self,train_data,test_data):
        train_data_f = train_data[['Inventory', 'Value']].values
        train_data_f = np.reshape(train_data_f,(-1,2))

        return train_data_f


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, num_epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train_model(self, x_train, y_train):
        for e in range(self.num_epochs):
            output = self.model(x_train.unsqueeze(-1)).squeeze()
            self.optimizer.zero_grad()
            loss = self.criterion(output, y_train)
            loss.backward()
            self.optimizer.step()

    def predict(self, x_test):
        predictions = []
        for input_data in x_test:
            input_data = torch.tensor(input_data).float().unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                prediction = self.model(input_data).squeeze().item()
                predictions.append(prediction)
        return predictions


class main:
    data = pd.read_csv("final_data_prediction.csv")

    month_map_reverse = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                         'July': 7, 'August': 8, 'September': 9, 'October': 10, 'Novermber': 11, 'December': 12}

    data['Month'] = data['Month'].map(month_map_reverse)
    data['Year'] = 2023
    data['Date'] = pd.to_datetime(data['Month'].astype(str) + ' ' + data['Year'].astype(str), format='%m %Y')
    data.set_index('Date', inplace=True)

    data.drop(columns=['Unnamed: 0'], inplace=True)

    start_date = '2023-01-01'
    end_date = '2023-12-01'

    prepare_to_lstm = prepare_LSTM(data)
    train_data, test_data = prepare_to_lstm.split_data_and_seq()
    print(train_data)
    print(test_data)






if __name__ == "__main__":
   main()
