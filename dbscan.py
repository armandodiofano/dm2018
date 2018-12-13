import pandas
import numpy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

df = pandas.read_csv("credit_default_corrected_train.csv")

X = df.drop(['credit_default','education','status','sex', 'limit'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X.values)

dist = pdist(X, 'euclidean')
dist = squareform(dist)


k = 280
'''
kth_distances = list()
for d in dist:
    index_kth_distance = numpy.argsort(d)[k]
    kth_distances.append(d[index_kth_distance])

plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
plt.ylabel('dist from %sth neighbor' % k)
plt.xlabel('sorted distances')

plt.show()
'''
for i in range(50,400):
    dbscan = DBSCAN(eps=0.32, min_samples=i)
    dbscan.fit(X)

    print("%s" %i, numpy.unique(dbscan.labels_, return_counts=True))
#print "silhoutte_score:", silhouette_score(kmeans_df, kmeans.labels_)


