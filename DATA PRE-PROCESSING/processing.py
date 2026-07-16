import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"D:\ML LAB\DATASETS\atlantic.csv\atlantic.csv")
print(df.head())
print(df.isnull().sum()) #This dataset trickly shows NaN as -999 which is also a missing data we need to find those values
sneaky_nulls = (df == -999).sum()
print(sneaky_nulls)
#Here we need to filter or remove innocent columns
sneaky_nulls = sneaky_nulls[sneaky_nulls > 0]
print(sneaky_nulls)
# Flexing with a colorful plot
plt.figure(figsize=(12, 6))
sns.barplot(x=sneaky_nulls.index, y=sneaky_nulls.values, palette='husl')

# Making it look put-together
plt.title('Exposing the -999 Fake Nulls', fontsize=16, fontweight='bold')
plt.ylabel('Count of -999', fontsize=12)
plt.xlabel('Suspect Columns', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Now we replace the -999 values with NaN to make it easier to handle missing data
df.replace(-999, np.nan, inplace=True)

data = df['Minimum Pressure'].dropna()
# 1. Find the 25th percentile (Q1) and 75th percentile (Q3)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)

# 1. Find the 25th percentile (Q1) and 75th percentile (Q3)
'''
data.quantile(0.25) walks 25% of the way down that line and grabs that exact number. This is Q1 (Quartile 1).
data.quantile(0.75) walks 75% of the way down that line and grabs that number. This is Q3 (Quartile 3).
'''
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
# 2. Calculate the Interquartile Range (IQR) - the middle 50% of your data
IQR = Q3 - Q1
'''
It represents the middle 50% of your dataset. It’s the "VIP section" where the most normal, average, reliable data lives. By strictly looking at the middle 50%, you completely ignore the extreme weirdos at the very bottom and the very top.
'''

# 3. Set the boundaries (1.5x IQR is the industry standard for outlier detection)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

'''
This is a legendary stats rule created by a mathematician named John Tukey back in the 70s. He basically figured out how to mathematically define "too weird."
'''

'''
We take the size of your normal VIP section (the IQR).

We multiply it by 1.5 to create a safe "buffer zone."

lower_bound = Q1 - 1.5 * IQR: We build a fence below the 25% mark.

upper_bound = Q3 + 1.5 * IQR: We build a fence above the 75% mark.
'''
# 4. Filter the data (Keep only the stuff inside the boundaries)
clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
print(f"Original data size: {len(data)}")
print(f"Cleaned data size: {len(clean_data)}")
print(f"Cleaned data head: {clean_data.head()}")

print(df['Minimum Pressure'].isnull().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns

# 2. The "Pattern" Method (Linear Interpolation)
# This calculates the exact increasing/decreasing intervals you talked about
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

# 3. The "Neighbor" Method (Forward/Backward Fill)
# We use this as a backup just in case the very FIRST or LAST row was a NaN 
# and interpolation didn't have two points to draw a line between
df[numeric_cols] = df[numeric_cols].ffill().bfill()

print("Remaining missing values:", df['Minimum Pressure'].isnull().sum())
print(df.isnull().sum())