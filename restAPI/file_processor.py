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
    clean(df)


def clean(file):
    relevantColumns = file[['code','color','Value','Inventory','January','February','March','April','May','June','July','August','September','October','November','December']]
    print(relevantColumns)
  