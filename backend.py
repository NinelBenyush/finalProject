import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class prepare_LSTM:
    def __init__(self, data):
        self.data = data
        self.seq = 3
        self.num_features = 1

    def split_data_and_seq(self):
        new_data = np.array(self.data['Value'])
        x_data, y_data = [], []

        for product_id, group_data in self.data.groupby('code_p'):
            product_values = group_data['Value'].values
            for i in range(len(product_values) - self.seq):
                x_data.append(product_values[i:i + self.seq])
                y_data.append(product_values[i + self.seq])

        x_data, y_data = np.array(x_data), np.array(y_data)

        train_size = int(len(self.data) * 0.8)
        x_train, y_train = torch.tensor(x_data[:train_size]).float(), torch.tensor(y_data[:train_size]).float()
        x_test, y_test = torch.tensor(x_data[train_size:]).float(), torch.tensor(y_data[train_size:]).float()

        return x_train, y_train, x_test, y_test

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class main:
    data = pd.read_csv("prediction_data.csv")

    month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    data['Month'] = data['Month'].map(month_map)
    final_data = data.set_index("Month")

    prepare_to_model = prepare_LSTM(final_data)
    x_train, y_train, x_test, y_test = prepare_to_model.split_data_and_seq()

    model = LSTM(input_size=1, hidden_size=1, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train.unsqueeze(2))
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')





if __name__ == "__main__":
    main()



