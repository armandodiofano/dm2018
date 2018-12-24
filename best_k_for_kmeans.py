import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pandas.read_csv("credit_default_corrected_train.csv")

X = df[df.credit_default == 1].drop(['credit_default','education','status','sex'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X.values)

sse_list = list()

max_k = 20
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, max_iter=100)
    kmeans.fit(X)

    sse_list.append(kmeans.inertia_)
    print silhouette_score(X, kmeans.labels_)

plt.plot(range(2, len(sse_list) + 2), sse_list)

plt.ylabel('SSE')
plt.xlabel('K')
plt.xticks(numpy.arange(0,max_k+1,1))
plt.tick_params(axis='both', which='major')

plt.show()

#best_k = 6
