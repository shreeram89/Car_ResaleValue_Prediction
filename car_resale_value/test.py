import pandas as pd
df = pd.read_csv(r"D:\Machine Leraning\TNS\capstone_project\manufacturing_data.csv")
# print(df.head())
print(df["Shifts"].unique())