import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"DATASETS\archive\Life Expectancy Data.csv")

print(df.size)
print(df.describe())
print(df.isnull().sum())

# print((df == -999).sum())
#Data cleaning using mean, median and mode
df['Life expectancy'] = df['Life expectancy '].fillna(df['Life expectancy '].mean(), inplace=True)
df['Adult Mortality'] = df['Adult Mortality'].fillna(df['Adult Mortality'].mean(), inplace=True)
df['Alcohol'] = df['Alcohol'].fillna(df['Alcohol'].mean(), inplace=True)
df['Hepatitis B'] = df['Hepatitis B'].fillna(df['Hepatitis B'].mean(), inplace=True)
df['BMI'] = df['BMI'].fillna(df['BMI'].mean(), inplace=True)
df['Polio'] = df['Polio'].fillna(df['Polio'].mean(), inplace=True)
df['Total expenditure'] = df['Total expenditure'].fillna(df['Total expenditure'].mean(), inplace=True)
df['Diphtheria'] = df['Diphtheria '].fillna(df['Diphtheria '].mean(), inplace=True)
df['GDP'] = df['GDP'].fillna(df['GDP'].mean(), inplace=True)
df['Population'] = df['Population'].fillna(df['Population'].mean(), inplace=True)
df['thinness 1-19 years'] = df['thinness  1-19 years'].fillna(df['thinness  1-19 years'].mean(), inplace=True)
df['thinness 5-9 years'] = df['thinness 5-9 years'].fillna(df['thinness 5-9 years'].mean(), inplace=True)
df['Income composition of resources'] = df['Income composition of resources'].fillna(df['Income composition of resources'].mean(), inplace=True)
#Data cleaning done, now we can check for null values
df = df.fillna(df.mean(numeric_only=True))
print(df.isnull().sum())
# df = df.fillna(df.mean(numeric_only=True))
from sklearn.model_selection import train_test_split
df['Status'] = df['Status'].map({'Developing': 1, 'Developed': 0})
X =  df.drop(columns=['Life expectancy', 'Country']) 
y = df['Life expectancy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
x = X.iloc[0].values
print("X:\n",x)
val = model.predict([x])
print(val)
