import pandas
import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.metrics import silhouette_score

def select_cluster(x, c):
    cluster_map = pandas.DataFrame()
    cluster_map['index'] = x.index.values
    cluster_map['cluster'] = kmeans.labels_
    return x.iloc[cluster_map[cluster_map.cluster == c].index]

def substract_cluster(x, c):
    cluster_map = pandas.DataFrame()
    cluster_map['index'] = x.index.values
    cluster_map['cluster'] = kmeans.labels_
    return x.iloc[cluster_map[cluster_map.cluster != c].index]

number_of_clusters = 5;

df = pandas.read_csv("credit_default_corrected_train.csv")
kmeans_df = df.drop(['credit_default','education','status','sex', 'limit'], axis=1)

scaler = MinMaxScaler()
kmeans_df = scaler.fit_transform(kmeans_df.values)

kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(kmeans_df)
print "silhoutte_score:", silhouette_score(kmeans_df, kmeans.labels_)

for i in range(0, number_of_clusters):
    cluster = select_cluster(df, i)
    rows = float(len(cluster.index))
    print "------------------"
    print "cluster", i
    print "% credit default:", len(cluster[cluster.credit_default == 1].index)/rows
    print "mean age:", cluster.age.mean()
    print "mode education:", cluster.education.mode()[0]
    print "mode status:", cluster.status.mode()[0]
    print "mode sex:", cluster.sex.mode()[0]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = substract_cluster(df, 2)

xdim = 'ba-jun'
ydim = 'pa-jul'
zdim = 'ps-jun'

xs = X[xdim]
ys = X[ydim]
zs = X[zdim]

ax.set_xlabel(xdim)
ax.set_ylabel(ydim)
ax.set_zlabel(zdim)

ax.scatter(xs, ys, zs, c=numpy.where(kmeans.labels_ != 2))


'''
plt.scatter(df['ba-jun'], df['pa-jul'], c=kmeans.labels_)
plt.xlabel("ba")
plt.ylabel("pa")
'''
plt.show()

