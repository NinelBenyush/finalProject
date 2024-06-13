import os
import numpy as np
import pandas as pd

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

    #file_name = f"data{file_num}.csv"

    file_name = f"{onlyTheName}_{file_num}.csv"


    df = pd.read_excel(filePath)
    location = os.path.join("./DataForPrediction", file_name)
    df.to_csv(location)
    clean(df, onlyTheName, file_num)


def clean(file, onlyTheName, file_num):
    relevantColumns = file[['code','color','Value','Inventory','January','February','March','April','May','June','July','August','September','October','November','December']]
    print(relevantColumns)
   # file_name = f"{onlyTheName}_{file_num}.csv"
   # new_path = os.path.join("./DataForPredictionAfterCleaning", file_name)
   # relevantColumns.to_csv(new_path)
   # print(f"Cleaned data saved to {new_path}")
    hot_encoding = pd.get_dummies(relevantColumns[['color']])
    rColumns = pd.concat([relevantColumns, hot_encoding], axis=1)
    rColumns = rColumns.drop(['color'], axis=1)
    #print(rColumns)

    if rColumns.isnull().values.any():
        rColumns = rColumns.fillna(0)
        print("NaN values found and replaced with 0.")
    else:
        print("No NaN values found.")

    rColumns = rColumns.apply(lambda x: np.where(x < 0, 0, x))

    print(rColumns)

    file_name = f"{onlyTheName}_{file_num}.csv"
    new_path = os.path.join("./DataForPredictionAfterCleaning", file_name)
    rColumns.to_csv(new_path)
    print(f"Cleaned data saved to {new_path}")

    months = ['January','February','March','April','May','June','July','August','September','October','November','December']
    rColumns[months] = rColumns[months].apply(pd.to_numeric, errors='coerce')
    
    #rest_columns = rColumns[['code','color_B','color_W', 'Value','Inventory']]
    #selected_columns = rColumns[['code', 'January','February','March','April','May','June','July','August','September','October','November','December']]
    subset_rColumns = rColumns[['code', 'January','February','March','April','May','June','July','August','September','October','November','December']]

    temp_df = pd.melt(subset_rColumns, id_vars=['code'], var_name="Month", value_name='Value')
    temp_df = temp_df.sort_values(by=['code', 'Month'])

    month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    #temp_df['Month numeric'] = temp_df.map(lambda x: month_map.get(x, None))
    temp_df['Month numeric'] = temp_df['Month'].map(month_map)
    
    #for month in month_map:
     #   if month in rColumns.columns:
     #       rColumns[f'{month} numeric'] = rColumns[month].map(lambda x: month_map.get(month, None))

    # Print the updated DataFrame
    #print(rColumns)

    temp_df = temp_df.sort_values(by=['code', 'Month numeric'])
    temp_df = temp_df.drop('Month numeric', axis=1)
    selected = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    df_rest_columns = rColumns.drop(columns=selected)

    merged_df = pd.merge(df_rest_columns, temp_df, on='code' )
    print(merged_df)

    name_to_change = {'Value_x':'Inventory Value', 'Value_y': "Value "}
    merged_df.rename(columns=name_to_change,inplace=True)

    print(merged_df)



  