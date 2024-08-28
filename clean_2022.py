import numpy as np
import pandas as pd

data = pd.read_csv("data_2022.csv")
data.columns = data.columns.str.strip()

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

data[months] = data[months].apply(pd.to_numeric, errors='coerce')

additional_columns = ['Minimum stock', 'purchase_r']

subset_data = data[['code'] + months + additional_columns]

temp_df = pd.melt(subset_data, id_vars=['code'] + additional_columns, var_name="Month", value_name='Value')

temp_df['Month'] = pd.Categorical(temp_df['Month'], categories=months, ordered=True)

temp_df = temp_df.sort_values(by=['code', 'Month'])

month_map_reverse = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
temp_df['Month'] = temp_df['Month'].map(month_map_reverse)

temp_df['Year'] = 2022
temp_df['Date'] = pd.to_datetime(temp_df['Month'].astype(str) + ' ' + temp_df['Year'].astype(str), format='%m %Y')

temp_df.set_index('Date', inplace=True)

print(temp_df)
columns_to_fix = ['Value', 'purchase_r', 'Minimum stock']

for column in columns_to_fix:
    temp_df[column] = pd.to_numeric(temp_df[column], errors='coerce')
    temp_df[column] = temp_df[column].fillna(0)
    temp_df[column] = temp_df[column].apply(lambda x: max(x, 0))
print(temp_df)

temp_df.to_csv("data_for_pred2022.csv")
