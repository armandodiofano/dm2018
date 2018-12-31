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

X = df.drop(['credit_default','education','status', 'sex', 'age','limit', "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep", "ba-apr", 'ba-may','ba-jun', "ba-jul", "ba-aug", "ba-sep"], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X.values)

dist = pdist(X, 'cityblock')
dist = squareform(dist)

dbscan = DBSCAN(eps = 0.20, min_samples = 55, metric='precomputed')
dbscan.fit(dist)

'''
X = select_cluster(df, 0)
X = df.drop(['credit_default','education','status', 'sex', 'age','limit', "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep", "ba-apr", 'ba-may','ba-jun', "ba-jul", "ba-aug", "ba-sep"], axis=1)
dist = pdist(X, 'cityblock')
dist = squareform(dist)

dbscan = DBSCAN(eps = 0.30, min_samples = 350, metric='precomputed')
dbscan.fit(dist)
'''


#0.30 cityblock 350
#0.18 euclidean 350
#0.18 cityblock 700
for i in range(0, len(numpy.unique(dbscan.labels_)[1:])):
    cluster = select_cluster(df, i)
    rows = float(len(cluster.index))
    print "------------------"
    print "cluster", i
    print "% credit default:", len(cluster[cluster.credit_default == 1].index)/rows
    print "mean limit:", cluster.limit.mean(), cluster.limit.std()
    print "mean age:", cluster.age.mean(), cluster.age.std()
    print "mean age:", cluster.age.mean(), cluster.age.std()
    print "mode education:", cluster.education.mode()[0]
    print "mode status:", cluster.status.mode()[0]
    print "mode sex:", cluster.sex.mode()[0]
    sta = cluster[['ps-sep', 'ps-aug', 'ps-jul', 'ps-jun', 'ps-may']]
    pay = cluster[['pa-sep', 'pa-aug', 'pa-jul', 'pa-jun', 'pa-may']]
    bay = cluster[['ba-sep', 'ba-aug', 'ba-jul', 'ba-jun', 'ba-may']]
    print sta.mean(), sta.std()
    print pay.mean(), pay.std()
    print bay.mean(), bay.std()

print(numpy.unique(dbscan.labels_, return_counts=True))

print "silhoutte_score:", silhouette_score(X, dbscan.labels_)


