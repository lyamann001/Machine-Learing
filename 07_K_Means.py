
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/Cust_Segmentation.csv')
print(df.head())

df = df.drop('Address', axis=1)

from sklearn.preprocessing import StandardScaler
X = df.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)


from sklearn.cluster import KMeans
k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(X)

labels = k_means.labels_
print(labels)

df['Clus_km'] = labels
print(df.head())


df.groupby('Clus_km').mean()

plt.scatter(X[:, 0], X[:, 3], c=labels.astype(np.float64), alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(8, 6))
# plt.clf()
# ax = Axes3D(fig, rect=(0, 0, .95, 1), elev=48, azim=134, auto_add_to_figure=False)
#
# plt.cla()
# ax.set_xlabel('Education')
# ax.set_ylabel('Age')
# ax.set_zlabel('Income')
#
# ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float64))
# ax.imshow()
