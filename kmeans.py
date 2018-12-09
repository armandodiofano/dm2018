import pandas
import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

best_k = 6;
df = pandas.read_csv("corrected_tr.csv")
dist = numpy.empty(k_max, dtype=float);
kmeans_df = df.drop(['age','credit_default','education','status','sex'], axis=1)
kmeans = KMeans(n_clusters=best_k, random_state=0).fit(kmeans_df)
dist[k] = kmeans.inertia_

axes = plt.gca()
axes.set_xlim([2,k_max])
axes.set_ylim([0.5 * 10**14, 3 * 10**14])

plt.xticks(numpy.arange(2,k_max,1))

plt.xlabel("K");
plt.ylabel("SSE")

plt.plot(dist)
plt.show()
