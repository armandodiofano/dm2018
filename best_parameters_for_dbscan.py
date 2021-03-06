import pandas
import numpy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

df = pandas.read_csv("credit_default_corrected_train.csv")

X = df.drop(['credit_default','education','status','sex', 'age','limit', "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep", "ba-apr", 'ba-may','ba-jun', "ba-jul", "ba-aug", "ba-sep"], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X.values)

dist = pdist(X, 'euclidean')
dist = squareform(dist)

k = 55

kth_distances = list()
for d in dist:
    index_kth_distance = numpy.argsort(d)[k]
    kth_distances.append(d[index_kth_distance])

plt.plot(range(0, len(kth_distances)), sorted(kth_distances))
plt.ylabel('dist from %sth neighbor' % k)
plt.xlabel('sorted distances')

plt.show()
'''

for i in range(100,1000, 20):
    for e in numpy.arange(0.17, 0.60, 0.02):
        dbscan = DBSCAN(eps=e, min_samples=i)
        dbscan.fit(X)
        print(i, e , numpy.unique(dbscan.labels_, return_counts=True))
        if len(numpy.unique(dbscan.labels_)) <= 2:
            print("BREAK")
            break
'''
#print "silhoutte_score:", silhouette_score(X, dbscan.labels_)


