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

number_of_clusters = 4;

df = pandas.read_csv("credit_default_corrected_train.csv")
dio_bambu = df.drop(['credit_default','education','status','sex', 'limit'], axis=1)
statuses = df[['ps-sep', 'ps-aug', 'ps-jul', 'ps-jun', 'ps-may', 'ps-apr']]
payments = df[['pa-sep', 'pa-aug', 'pa-jul', 'pa-jun', 'pa-may', 'pa-apr']]
billings = df[['ba-sep', 'ba-aug', 'ba-jul', 'ba-jun', 'ba-may', 'ba-apr']]

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

centers = scaler.inverse_transform(kmeans.cluster_centers_)

#Stampo i centroidi di ogni cluster relativi ai payment statuses
#si evince che il cluster 2 (quello con piu' % di credit default)
#ha le coordinate dei centroidi (di cui sopra) molto piu' alte dei restanti
'''
for i in range(0, len(centers)):
    plt.plot(centers[i, 1:7], marker='o', label='cluster %s' % i)
plt.tick_params(axis='both', which='major')
plt.xticks(range(0,len(statuses.columns)), statuses.columns)
plt.legend()
'''

#stampo i centroidi per i billing amounts
'''
for i in range(0, len(centers)):
    plt.plot(centers[i,7:13], marker='o', label='cluster %s' % i)
plt.tick_params(axis='both', which='major')
plt.xticks(range(0,len(billings.columns)), billings.columns)
plt.legend()
'''

#stampo i centroidi per i payments 
'''
for i in range(0, len(centers)):
    plt.plot(centers[i,13:19], marker='o', label='cluster %s' % i)
plt.tick_params(axis='both', which='major')
plt.xticks(range(0,len(payments.columns)), payments.columns)
plt.legend()
'''


'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = df
xdim = 'ba-aug'
ydim = 'pa-sep'
zdim = 'ps-aug'

xs = X[xdim]
ys = X[ydim]
zs = X[zdim]

ax.set_xlabel(xdim)
ax.set_ylabel(ydim)
ax.set_zlabel(zdim)

ax.scatter(xs, ys, zs, c=kmeans.labels_)
'''

plt.show()

