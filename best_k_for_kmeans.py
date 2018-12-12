import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

k_max = 50 + 1
df = pandas.read_csv("credit_default_corrected_train.csv")
dist = numpy.empty(k_max, dtype=float);
kmeans_df = df.drop(['credit_default','education','status','sex'], axis=1)

scaler = MinMaxScaler()
kmeans_df = scaler.fit_transform(kmeans_df.values)

for k in range(2,k_max):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(kmeans_df)
    dist[k] = kmeans.inertia_

axes = plt.gca()
axes.set_xlim([2,k_max])
axes.set_ylim([400, 1500])

plt.xticks(numpy.arange(0,k_max,10))

plt.xlabel("K");
plt.ylabel("SSE")

plt.plot(dist)
plt.show()

#best_k = 6
