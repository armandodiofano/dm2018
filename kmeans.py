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
color_labels = ['b', 'y', 'r', 'g']
label_names= ['Senza rischio', 'Piccoli pagatori', 'Ritardatari', 'Grandi pagatori']
df = pandas.read_csv("credit_default_corrected_train.csv")

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
    print label_names[i]
    print cluster.shape[0]
    print "% credit default:", len(cluster[cluster.credit_default == 1].index)/rows
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
centers = scaler.inverse_transform(kmeans.cluster_centers_)

'''
#Stampo i centroidi di ogni cluster relativi ai payment statuses
#si evince che il cluster 2 (quello con piu' % di credit default)
#ha le coordinate dei centroidi (di cui sopra) molto piu' alte dei restanti
fig, (ax1,ax2,ax3) = plt.subplots(3, 1)

for i in range(0, len(centers)):
    ax1.plot(centers[i, 1:7], color=color_labels[i], marker='o', label=label_names[i])
ax1.tick_params(axis='both', which='major')
plt.sca(ax1)
plt.xticks(range(0,len(statuses.columns)), statuses.columns)
ax1.legend()


#stampo i centroidi per i billing amounts

for i in range(0, len(centers)):
    ax2.plot(centers[i,7:13], color=color_labels[i], marker='o', label=label_names[i])
ax2.tick_params(axis='both', which='major')
plt.sca(ax2)
plt.xticks(range(0,len(billings.columns)), billings.columns)
ax2.legend()

#stampo i centroidi per i payments 

for i in range(0, len(centers)):
    ax3.plot(centers[i,13:19], marker='o', color=color_labels[i], label=label_names[i])
ax3.tick_params(axis='both', which='major')
plt.sca(ax3)
plt.xticks(range(0,len(payments.columns)), payments.columns)
ax3.legend()
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


diolupo = pandas.DataFrame()
diolupo["a"] = kmeans.labels_
diolupo["a"] = diolupo["a"].replace({0:'b',1:'y', 2:'r', 3:'g'})
print diolupo["a"]
ax.scatter(xs, ys, zs, c=diolupo["a"].values)

plt.show()

