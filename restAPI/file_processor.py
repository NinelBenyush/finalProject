#clean the excel file and prepare for the LSTM model
import datetime
import os
import numpy as np
import pandas as pd
import torch
from trainAndPredict import makeModel, prepare_data, predict_by_product


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
    seq_length = 4

    # Rename columns
    file.rename(columns={'Minimun stock': 'Minimum stock'}, inplace=True)
    
    # Verify column names after renaming
    print("Columns after renaming:", file.columns)

    # Extract relevant columns
    relevantColumns = file[['code', 'color', 'Value', 'Inventory', 'Minimum stock', 'purchase_r', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]
    print("Relevant columns:", relevantColumns.columns)

    hot_encoding = pd.get_dummies(relevantColumns[['color']])
    rColumns = pd.concat([relevantColumns, hot_encoding], axis=1).drop(['color'], axis=1)

    if rColumns.isnull().values.any():
        rColumns = rColumns.fillna(0)
    rColumns = rColumns.apply(lambda x: np.where(x < 0, 0, x))

    print("rColumns columns:", rColumns.columns)

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    rColumns[months] = rColumns[months].apply(pd.to_numeric, errors='coerce')

    subset_rColumns = rColumns[['code', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']]
    print("Subset rColumns columns:", subset_rColumns.columns)

    temp_df = pd.melt(subset_rColumns, id_vars=['code'], var_name="Month", value_name='Value')
    temp_df = temp_df.sort_values(by=['code', 'Month'])
    print("temp_df columns:", temp_df.columns)

    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    temp_df['Month numeric'] = temp_df['Month'].map(month_map)

    temp_df = temp_df.sort_values(by=['code', 'Month numeric']).drop('Month numeric', axis=1)
    temp_df = temp_df.drop_duplicates(subset=['code', 'Month'])
    print("After dropping duplicates:", temp_df.columns)

    df_rest_columns = rColumns.drop(columns=months).drop_duplicates(subset=['code'])
    print("df_rest_columns columns:", df_rest_columns.columns)

    merged_df = pd.merge(df_rest_columns, temp_df, on='code')
    print("merged_df columns:", merged_df.columns)

    name_to_change = {'Value_x': 'Inventory Value', 'Value_y': "Value"}
    merged_df.rename(columns={'Value ': 'Value'}, inplace=True)
    merged_df.rename(columns=name_to_change, inplace=True)

    merged_df['Month'] = merged_df['Month'].map(month_map)
    merged_df['Year'] = 2023
    merged_df['Date'] = pd.to_datetime(merged_df['Month'].astype(str) + ' ' + merged_df['Year'].astype(str), format='%m %Y')
    merged_df.set_index('Date', inplace=True)
    merged_df.columns = merged_df.columns.str.strip()  


    print("Final merged_df ", merged_df)

    input_size = 3  
    num_layers = 2 
    hidden_size = 64
    model_load_path = r"C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/bilstm_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = makeModel(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval()

    merged_df, scaler_value, scaler_min_stock, scaler_purchase_r = prepare_data(merged_df)

# Make predictions
    seq_length = 4
    predictions = predict_by_product(model, merged_df, seq_length)
    for code in predictions:
      predictions[code] = scaler_value.inverse_transform(np.array(predictions[code]).reshape(-1, 1)).flatten()
      print(predictions[code])


    df = pd.DataFrame(dict([(k, pd.Series(v[:12])) for k, v in predictions.items()]))
    df = df.round().astype(int)

    date_range = pd.date_range(start='2024-01-01', periods=12, freq='MS')
    formatted_dates = date_range.strftime('%d-%m-%Y')

    df.index = formatted_dates

    print(df)


    # Save the merged DataFrame (optional)
    file_name = f"{onlyTheName}_{file_num}.csv"
    new_path = os.path.join("./results", file_name)
    return new_path, merged_df

def send_the_result(path, df):
    return path, df
