import pandas as pd

import preprocessing.main as preprocessing

# Read the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\arpit\Downloads\sample_data.csv")

df = preprocessing.preprocessor(df, "text")

print(df.head())
