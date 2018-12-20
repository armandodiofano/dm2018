import pandas
import numpy
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

def select_cluster(x, c):
    cluster_map = pandas.DataFrame()
    cluster_map['index'] = x.index.values
    cluster_map['cluster'] = dbscan.labels_
    return x.iloc[cluster_map[cluster_map.cluster == c].index]

def substract_cluster(x, c):
    cluster_map = pandas.DataFrame()
    cluster_map['index'] = x.index.values
    cluster_map['cluster'] = dbscan.labels_
    return x.iloc[cluster_map[cluster_map.cluster != c].index]

df = pandas.read_csv("credit_default_corrected_train.csv")

X = df.drop(['credit_default','education','status','sex', 'age', 'limit', "ps-apr","ps-may", "ps-jun", "pa-apr", "pa-may", "pa-jun", "ba-apr", 'ba-may','ba-jun'], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X.values)

dist = pdist(X, 'cosine')
dist = squareform(dist)

dbscan = DBSCAN(eps = 0.06, min_samples = 670, metric='precomputed')
dbscan.fit(dist)
'''
for i in range(0, len(dbscan.labels_[1:])):
    cluster = select_cluster(df, i)
    rows = float(len(cluster.index))
    print "------------------"
    print "cluster", i
    print "% credit default:", len(cluster[cluster.credit_default == 1].index)/rows
    print "mean age:", cluster.age.mean()
    print "mode education:", cluster.education.mode()[0]
    print "mode status:", cluster.status.mode()[0]
    print "mode sex:", cluster.sex.mode()[0]
'''
print(numpy.unique(dbscan.labels_, return_counts=True))

#print "silhoutte_score:", silhouette_score(X, dbscan.labels_)


