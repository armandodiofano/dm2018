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
X = df.drop(['credit_default','education','status','sex', 'age','limit', "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep", "ba-apr", 'ba-may','ba-jun', "ba-jul", "ba-aug", "ba-sep"], axis=1)  
scaler = MinMaxScaler() 
X = scaler.fit_transform(X.values)  

distanza="euclidean" #euclidean, manhattan, ... 
link_mode="ward" #complete, single, average, ward... 
threshold=15.2 #taglio dendogramma per colorazione 
num_cluster=4 #numero di cluster first clusterign
cluster_to_sub=0 #indice cluster da riclusterizzare
num_sub_cluster=2 #numero di cluster subclusterign
 
#DENDOGRAM 
data_dist = pdist(X, metric=distanza) 
data_link = linkage(data_dist, method=link_mode, metric=distanza)  
res = dendrogram(data_link, color_threshold=threshold, truncate_mode='lastp') #aggiungere truncate_mode='lastp' per troncare a foglie 
plt.show() 

#FIRST CLUSTERING
hier = AgglomerativeClustering(affinity=distanza, linkage=link_mode, n_clusters=num_cluster)  
hier = hier.fit(X) 
 
for i in range(0, len(nd.unique(hier.labels_))): 
    cluster = select_cluster(df, i) 
    rows = float(len(cluster.index)) 
    print ("------------------") 
    print ("cluster", i) 
    print("element:", len(cluster))
    print ("% credit default:", len(cluster[cluster.credit_default == 1].index)/rows)
    print "mean limit:", cluster.limit.mean(), cluster.limit.std()
    print "mean age:", cluster.age.mean(), cluster.age.std()
    print "mode education:", cluster.education.mode()[0]
    print "mode status:", cluster.status.mode()[0]
    print "mode sex:", cluster.sex.mode()[0]
    sta = cluster[['ps-sep', 'ps-aug']]
    pay = cluster[['pa-sep', 'pa-aug']]
    bay = cluster[['ba-sep', 'ba-aug']]
    print sta.mean(), sta.std()
    print pay.mean(), pay.std()
    print bay.mean(), bay.std()

'''
#SUB CLUSTERING of cluster_to_sub
cf = select_cluster(df,cluster_to_sub) 
X = cf.drop(['credit_default','education','status', 'sex', 'age'], axis=1) 

hier = AgglomerativeClustering(affinity=distanza, linkage=link_mode, n_clusters=num_sub_cluster) #funzione che fa clustering con label 
hier = hier.fit(X) 
for j in range(0, len(nd.unique(hier.labels_))): 
    cluster = select_cluster(cf, j) 
    rows = float(len(cluster.index))
    print ("------------------") 
    print ("subcluster[0]", j) 
    print("element:", len(cluster)) 
    print ("% credit default:", len(cluster[cluster.credit_default == 1].index)/rows)
'''
