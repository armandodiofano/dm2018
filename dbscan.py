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
for i in range(400,800, 25):
    for e in numpy.arange(0.27, 0.37, 0.005):
        dbscan = DBSCAN(eps=e, min_samples=i)
        dbscan.fit(X)
        print(i, e , numpy.unique(dbscan.labels_, return_counts=True))
        
#print "silhoutte_score:", silhouette_score(kmeans_df, kmeans.labels_)


