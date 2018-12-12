import pandas
import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D 

df = pandas.read_csv("credit_default_corrected_train.csv")
kmeans_df = df.drop(['credit_default','education','status','sex'], axis=1)

scaler = MinMaxScaler()
kmeans_df = scaler.fit_transform(kmeans_df.values)

kmeanses = list()

for i in range(0,10):
    kmeanses.append(KMeans(n_clusters=6, random_state=0).fit(kmeans_df))

best_kmeans = kmeanses.pop(9)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.scatter(xs, ys, zs, c=c, marker=m)

plt.show()

