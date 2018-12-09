import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

k_max = 20 + 1
df = pandas.read_csv("credit_default_corrected_train.csv")
dist = numpy.empty(k_max, dtype=float);
kmeans_df = df.drop(['age','credit_default','education','status','sex'], axis=1)

scaler = MinMaxScaler()
kmeans_df = scaler.fit_transform(kmeans_df.values)

for k in range(2,k_max):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(kmeans_df)
    dist[k] = kmeans.inertia_

axes = plt.gca()
axes.set_xlim([2,k_max])
axes.set_ylim([0.5 * 10**14, 3 * 10**14])

plt.xticks(numpy.arange(2,k_max,1))

plt.xlabel("K");
plt.ylabel("SSE")

plt.plot(dist)
plt.show()

