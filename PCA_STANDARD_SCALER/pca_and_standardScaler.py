import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\ML LAB\PCA_STANDARD_SCALER\spotify_sample.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())

#StandardScaling start
df.drop(columns=['track_id','artist','album','release_date','genre'], inplace=True) #removing identifers
print(df.select_dtypes(include='number'))
print(df.describe())
numeric_cols = df.select_dtypes(include='number').columns
print("Numeric columns:", numeric_cols)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("Verifying the scaling process:")
print(df.describe())
#StandardScaling end

#PCA start

#check co-relation
corr = df.corr(numeric_only=True)
print(corr)
sns.heatmap(corr, annot=True)
#plt.show()
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df[numeric_cols])
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
# print("Cumulative Explained Variance Ratio:")
# print(np.cumsum(pca.explained_variance_ratio_))
#import numpy as np

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

print("Cumulative Variance:")
print(cumulative_variance)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+1),
         cumulative_variance,
         marker='o')

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Choosing n_components")
plt.grid(True)

plt.show()
pca = PCA(n_components=5)

X_pca = pca.fit_transform(df[numeric_cols])
print(X_pca.shape)