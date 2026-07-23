import numpy as np
import pandas as pd

df = pd.read_csv(r"DATASETS\atlantic.csv\atlantic.csv")

print(df.describe())
print("Mean:",df['Time'].mean())
print("Median:",df['Time'].median())
print("Mode:",df['Time'].mode())