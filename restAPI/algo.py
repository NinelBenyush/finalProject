import pandas as pd
import os
FINAL_RESULTS_DIRECTORY="C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/results"
FINAL_RESULTS_PHOTOS ="C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/resultsPng"
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.dates as mdates


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def send_for_alg(filepath):
    df = pd.read_csv(filepath)
    #print(filepath)
    #df.set_index('Date', inplace=True)
    
    df.columns = df.columns.str.strip()
    print("Columns in the DataFrame:", df.columns.tolist())
    final_data = df[["Inventory", "Value", "code"]]
    #print(final_data)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(final_data[['Inventory', 'Value']])
    final_data[['Inventory', 'Value']] = scaled_values
    #print("Scaled data:\n", final_data)

    train_size = int(len(final_data) * 0.8)
    train_data = final_data[:train_size]
    test_data = final_data[train_size:]

    inverse_scaled_values = scaler.inverse_transform(final_data[['Inventory', 'Value']])
    final_data[['Inventory', 'Value']] = inverse_scaled_values
    #print(final_data)


def create_lstm_model(input_size, hidden_size, num_layers):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)
            self.softplus = nn.Softplus()

        def forward(self, x):
            hidden = torch.zeros(num_layers, x.size(0), hidden_size)
            cell = torch.zeros(num_layers, x.size(0), hidden_size)
            out, _ = self.lstm(x,(hidden,cell))
            out = self.linear(out[:, -1, :])
            out = self.softplus(out)
            return out

    return LSTMModel(input_size, hidden_size, num_layers)


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y) 
    return np.array(xs), np.array(ys).reshape(-1, 1)  


def prepare(final_data):
    sequence_length = 3
    X_train, y_train, X_test, y_test = [], [], [], []

# Split data and create sequences for each product
    for code_p, group in final_data.groupby('code'):
       group_values = group['Value'].values
    
    #  Determine split index
       train_size = int(len(group_values) * 0.8)
    
    # Training data
       group_train = group_values[:train_size]
       X_tr, y_tr = create_sequences(group_train, sequence_length)
       X_train.append(X_tr)
       y_train.append(y_tr)
    
    # Testing data
       group_test = group_values[train_size - sequence_length:]  
       X_te, y_te = create_sequences(group_test, sequence_length)
       X_test.append(X_te)
       y_test.append(y_te)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    X_train = torch.FloatTensor(X_train).unsqueeze(2) 
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test).unsqueeze(2)  
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test


   #we will send the df to the alg

    #file_name = filepath.split('\\')[-1]
    #onlyName = file_name.split("_")[0]
    #file_path = os.path.join(FINAL_RESULTS_DIRECTORY, onlyName + '.csv')
    #print(file_path)
    #df.to_csv(file_path) """


