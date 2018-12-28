from scipy.stats import mode
from sklearn.metrics import silhouette_score
import pandas
import numpy as nd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

def select_cluster(x, c):
    cluster_map = pandas.DataFrame()
    cluster_map['index'] = x.index.values
    cluster_map['cluster'] = hier.labels_
    return x.iloc[cluster_map[cluster_map.cluster == c].index]

df = pandas.read_csv("credit_default_corrected_train.csv")
X = df.drop(['credit_default','education','status', 'sex', 'age'], axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X.values)

hier = AgglomerativeClustering(affinity='euclidean', linkage='complete', n_clusters=2) 
hier = hier.fit(X)

for i in range(0, len(nd.unique(hier.labels_))):
    cluster = select_cluster(df, i)
    rows = float(len(cluster.index))
    print ("------------------")
    print ("cluster", i)
    print("element:", len(cluster))
    print ("% credit default:", len(cluster[cluster.credit_default == 1].index)/rows)

cf = select_cluster(df,0)
X = cf.drop(['credit_default','education','status', 'sex', 'age'], axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X.values)

hier = AgglomerativeClustering(affinity='euclidean', linkage='complete', n_clusters=3) #funzione che fa clustering con label
hier = hier.fit(X)

for j in range(0, len(nd.unique(hier.labels_))):
    cluster = select_cluster(cf, j)
    rows = float(len(cluster.index))
    print ("------------------")
    print ("subcluster[0]", j)
    print("element:", len(cluster))
    print ("% credit default:", len(cluster[cluster.credit_default == 1].index)/rows)