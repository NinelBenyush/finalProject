import pandas as pd
import os
FINAL_RESULTS_DIRECTORY="C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/results"

def send_for_alg(filepath):
    df = pd.read_csv(filepath)
    #print(filepath)
   # print(f'ready to send to the alg {df}')

   #we will send the df to the alg

    file_name = filepath.split('\\')[-1]
    onlyName = file_name.split("_")[0]
    #print(f"only the name {onlyName}")
    file_path = os.path.join(FINAL_RESULTS_DIRECTORY, onlyName + '.csv')
    print(file_path)

    df.to_csv(file_path)


