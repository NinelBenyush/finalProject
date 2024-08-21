#clean the excel file and prepare for the LSTM model
import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 1
num_layers = 3
hidden_size = 80
dropout = 0.2

counter_file = "count_for_data.txt"

def get_num():
    with open(counter_file, 'r') as f:
        count = int(f.read().strip())
    count += 1

    with open(counter_file, 'w') as f:
        f.write(str(count))

    return count

def work_on_file(filePath):
    onlyTheName = os.path.splitext(os.path.basename(filePath))[0]
    file_num = get_num()
    file_name = f"{onlyTheName}_{file_num}.csv"
    df = pd.read_excel(filePath)
    location = os.path.join("./DataForPrediction", file_name)
    df.to_csv(location)
    res_file_path, final_df = clean(df, onlyTheName, file_num)
    return res_file_path

def clean(file, onlyTheName, file_num):
    relevantColumns = file[['code', 'color', 'Value', 'Inventory', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]
    
    hot_encoding = pd.get_dummies(relevantColumns[['color']])
    rColumns = pd.concat([relevantColumns, hot_encoding], axis=1).drop(['color'], axis=1)

    if rColumns.isnull().values.any():
        rColumns = rColumns.fillna(0)
    rColumns = rColumns.apply(lambda x: np.where(x < 0, 0, x))

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    rColumns[months] = rColumns[months].apply(pd.to_numeric, errors='coerce')

    subset_rColumns = rColumns[['code', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]

    temp_df = pd.melt(subset_rColumns, id_vars=['code'], var_name="Month", value_name='Value')
    temp_df = temp_df.sort_values(by=['code', 'Month'])

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    temp_df['Month numeric'] = temp_df['Month'].map(month_map)

    temp_df = temp_df.sort_values(by=['code', 'Month numeric']).drop('Month numeric', axis=1)
    
    # Remove duplicates in temp_df if any
    temp_df = temp_df.drop_duplicates(subset=['code', 'Month'])

    df_rest_columns = rColumns.drop(columns=months).drop_duplicates(subset=['code'])

    merged_df = pd.merge(df_rest_columns, temp_df, on='code')

    name_to_change = {'Value_x': 'Inventory Value', 'Value_y': "Value"}
    merged_df.rename(columns={'Value ': 'Value'}, inplace=True)
    merged_df.rename(columns=name_to_change, inplace=True)

    merged_df['Month'] = merged_df['Month'].map(month_map)
    merged_df['Year'] = 2023
    merged_df['Date'] = pd.to_datetime(merged_df['Month'].astype(str) + ' ' + merged_df['Year'].astype(str), format='%m %Y')
    merged_df.set_index('Date', inplace=True)

    file_name = f"{onlyTheName}_{file_num}.csv"
    new_path = os.path.join("./results", file_name)

    merged_df = merged_df[['code', 'Value']]

    random_values = np.random.uniform(3, 5, size=merged_df.shape[0])

    random_values = np.round(random_values).astype(int)

    random_signs = np.random.choice([-1, 1], size=merged_df.shape[0])


    modified_values = np.abs(merged_df['Value'] + random_signs * random_values)

    modified_values[modified_values == 0] = 1

    merged_df['Value'] = modified_values



    merged_df.to_csv(new_path)

    send_the_result(new_path, merged_df)
    return new_path, merged_df

def send_the_result(path, df):
    return path, df
