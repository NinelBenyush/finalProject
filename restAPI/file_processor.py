import os
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
    encoded_df = pd.concat([relevantColumns, hot_encoding], axis=1)
    encoded_df = encoded_df.drop(['color'], axis=1)
    print(encoded_df)


  