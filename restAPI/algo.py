import pandas as pd
import os
FINAL_RESULTS_DIRECTORY="C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/results"
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
    print(final_data)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(final_data[['Inventory', 'Value']])
    final_data[['Inventory', 'Value']] = scaled_values
    print("Scaled data:\n", final_data)

    train_size = int(len(final_data) * 0.8)
    train_data = final_data[:train_size]
    test_data = final_data[train_size:]

    inverse_scaled_values = scaler.inverse_transform(final_data[['Inventory', 'Value']])
    final_data[['Inventory', 'Value']] = inverse_scaled_values
    print(final_data)


   

   #we will send the df to the alg

    #file_name = filepath.split('\\')[-1]
    #onlyName = file_name.split("_")[0]
    #file_path = os.path.join(FINAL_RESULTS_DIRECTORY, onlyName + '.csv')
    #print(file_path)
    #df.to_csv(file_path) """


