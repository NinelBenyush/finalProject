import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt


class prepare_LSTM:
    def __init__(self, data):
        self.seq =3
        self.data = data
        self.num_features = 1
        self.scaler = None

    def split_data(self):
        train_size = int(len(self.data) * 0.8) + 3
        train_data = self.data[:train_size][['Value']]
        test_data = self.data[train_size:][['Value']]
        dataset_train = np.reshape(train_data, (-1, 3))  # reshape to get only numeric values
        dataset_test = np.reshape(test_data, (-1, 3))
        # here we can also add another columns if we will need to

        return dataset_train, dataset_test

    def data_normalization(self, train_data, test_data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_f = self.scaler.fit_transform(train_data)
        scaled_test_f = self.scaler.transform(test_data)

        return scaled_train_f, scaled_test_f

    def inverse_transform(self, scaled_data):
        return self.scaler.inverse_transform(scaled_data)

    def divide_to_seq(self, test, train):
        x_test, y_test, x_train, y_train = [], [], [], []
        for i in range(len(test) - self.seq):
            x_test.append(test[i:i+self.seq])
        x_test= np.array(x_test)
        x_test = torch.tensor(x_test, dtype=torch.float32)

        for i in range(len(train) - self.seq):
            x_train.append(train[i:i+self.seq])
        x_train= np.array(x_train)
        x_train = torch.tensor(x_train, dtype=torch.float32)

        for i in range(len(test) - self.seq):
            y_test.append(test[i + self.seq])  # Only append the last value of the sequence
        for i in range(len(train) - self.seq):
            y_train.append(train[i + self.seq])  # Only append the last value of the sequence
        y_test, y_train = np.array(y_test), np.array(y_train)
        y_test = torch.tensor(y_test, dtype=torch.float32)  # shape: (num_samples,)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        return x_test, y_test, x_train, y_train



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(-1)
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out.squeeze(-1)


class Training:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device=torch.device('cpu')):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_hist = []
        self.test_hist = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0.0

            # Training
            self.model.train()
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                loss = self.loss_fn(predictions, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Calculate average training loss
            average_loss = total_loss / len(self.train_loader)
            self.train_hist.append(average_loss)

            # Validation on test data
            self.model.eval()
            with torch.no_grad():
                total_test_loss = 0.0

                for batch_X_test, batch_y_test in self.test_loader:
                    batch_X_test, batch_y_test = batch_X_test.to(self.device), batch_y_test.to(self.device)
                    predictions_test = self.model(batch_X_test)
                    test_loss = self.loss_fn(predictions_test, batch_y_test)

                    total_test_loss += test_loss.item()

                # Calculate average test loss
                average_test_loss = total_test_loss / len(self.test_loader)
                self.test_hist.append(average_test_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}')

            return self.train_hist, self.test_hist



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
    indices = list(range(len(data)))

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
    train_data, test_data = prepare_to_lstm.split_data()

    final_train_data, final_test_data = prepare_to_lstm.data_normalization(train_data, test_data)

    # Inverse to real values
    # inv_train_data = prepare_to_lstm.inverse_transform(final_train_data)
    # inv_test_data = prepare_to_lstm.inverse_transform(final_test_data)

    x_test, y_test, x_train, y_train = prepare_to_lstm.divide_to_seq(final_test_data, final_train_data)

    input_size = 3
    num_layers = 2
    hidden_size = 64
    output_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM(input_size, hidden_size, num_layers).to(device)

    loss_function = torch.nn.MSELoss(reduction="mean")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # print(model)

    batch_size = 12

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = Training(model, train_loader, test_loader, loss_function, optimizer, device)
    train_hist, test_hist = trainer.train(num_epochs=50)

    num_epochs = 50

    x = np.linspace(1, num_epochs, num_epochs)
    plt.plot(x, train_hist, scalex=True, label="Training loss")
    plt.plot(x, test_hist, label="Test loss")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
