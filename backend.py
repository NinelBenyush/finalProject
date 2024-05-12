import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class prepare_LSTM:
    def __init__(self, data):
        self.data = data
        self.seq = 3
        self.num_features = 1

    def split_data_and_seq(self):
        new_data = np.array(self.data[:, -2])  # Access the 'Value' column, which is one before the last column
        inventory_values = self.data[:, 1]
        x_data, y_data = [], []

        for i in range(len(new_data) - self.seq + 1):
            x_data.append(new_data[i:i + self.seq])
            y_data.append(new_data[i + self.seq - 1])

        for i in range(self.seq - 1, len(new_data)):
            y_data[i - self.seq + 1] = inventory_values[i]

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
    data = pd.read_csv("final_data_prediction.csv")

    month_map_reverse = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                         'July': 7, 'August': 8, 'September': 9, 'October': 10, 'Novermber': 11, 'December': 12}

    data['Month'] = data['Month'].map(month_map_reverse)
    data['Year'] = 2023
    data['Date'] = pd.to_datetime(data['Month'].astype(str) + ' ' + data['Year'].astype(str), format='%m %Y')
    data.set_index('Date', inplace=True)

    data.drop(columns=['Unnamed: 0'], inplace=True)
    print(data)

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    prepare_to_model = prepare_LSTM(data_normalized)
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

    sum_predictions = []
    for i in range(0, len(predictions), 12):
        sum_predictions.append(sum(predictions[i:i+12]))

    # Normalize the sum of predictions
    max_sum_prediction = max(sum_predictions)
    min_sum_prediction = min(sum_predictions)
    sum_predictions_normalized = []
    for sum_pred in sum_predictions:
        normalized_value = (sum_pred - min_sum_prediction) / (max_sum_prediction - min_sum_prediction)
        sum_predictions_normalized.append(normalized_value)

    # Compare the sum of every 12 predictions with the inventory values
    for i in range(0, len(predictions), 12):
        sum_prediction = sum(predictions[i:i+12])  # Calculate sum for this interval
        sum_inventory = y_test[i].item()           # Get inventory sum for this interval

    # Evaluate and plot the model's performance
    y_test = y_test.numpy()
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error: {rmse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(range(0, len(predictions), 12), sum_predictions_normalized, label='Sum of Predictions (Normalized)')
    plt.plot(range(0, len(predictions), 12), y_test[::12], label='Inventory')
    plt.xlabel('Time (in 12-month intervals)')
    plt.ylabel('Sum (Normalized)')
    plt.title('Comparison of Sum of Predictions and Inventory')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
