import os
import pandas as pd

def work_on_file(filePath):
    print(f"name of the file: {filePath}")

    df = pd.read_excel(filePath)
    df.to_csv("data.csv")
