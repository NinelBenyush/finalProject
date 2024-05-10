import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class prepare_LSTM:
    def __init__(self, data):
        self.data = data
        self.seq = 3
        self.num_features = 1

    def split_data_and_seq(self):
        new_data = np.array(self.data['Value'])
        x_data, y_data = [], []

        for i in range(len(new_data) - self.seq + 1):
            x_data.append(new_data[i:i + self.seq])
            y_data.append(new_data[i + self.seq - 1])

        x_data, y_data = np.array(x_data), np.array(y_data)

        train_size = int(len(self.data) * 0.8)
        x_train, y_train = torch.tensor(x_data[:train_size]).float(), torch.tensor(y_data[:train_size]).float()
        x_test, y_test = torch.tensor(x_data[train_size:]).float(), torch.tensor(y_data[train_size:]).float()

        return x_train, y_train, x_test, y_test


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
    data = pd.read_csv("prediction_data.csv")

    month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    data['Month'] = data['Month'].map(month_map)
    data['Year'] = 2023
    data['Date'] = pd.to_datetime(data['Month'] + ' ' + data['Year'].astype(str), format='%B %Y')
    data.set_index('Date', inplace=True)

    prepare_to_model = prepare_LSTM(data)
    x_train, y_train, x_test, y_test = prepare_to_model.split_data_and_seq()

    input_size = 1
    hidden_size = 50
    num_layers = 1
    output_size = 1
    model = LSTM(input_size, hidden_size, num_layers, output_size)

    learning_rate = 0.01
    num_epochs = 100

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = ModelTrainer(model, criterion, optimizer, num_epochs)
    trainer.train_model(x_train, y_train)
    predictions = trainer.predict(x_test)

    # Evaluate and plot the model's performance
    y_test = y_test.numpy()
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error: {rmse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
