import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from fim import apriori
import time
import math

df = pd.read_csv("credit_default_corrected_train.csv", skipinitialspace=True, sep=',')
bins=math.ceil(math.log2(10000)+1)
statuses = df[['ps-sep', 'ps-aug', 'ps-jul', 'ps-jun', 'ps-may', 'ps-apr']]
to_cut=['limit','age','pa-sep', 'pa-aug', 'pa-jul', 'pa-jun', 'pa-may', 
'pa-apr','ba-sep', 'ba-aug', 'ba-jul', 'ba-jun', 'ba-may', 'ba-apr']
for i in to_cut:
    df[i] = pd.cut(df[i].astype(int), bins, precision=1, right=False)
    df[i] = i+" "+df[i].astype(str)
for i in statuses:
    df[i] = i+" "+df[i].astype(str)
df["credit_default"]="default "+df["credit_default"].astype(str)
print(df.head())

baskets = df.values.tolist()

#FREQUENT ITEMSET
percent=[20,30,40,50,60,70,80,90]
for i in percent:
    itemsets = apriori(baskets, supp=i, zmin=2, target='m')  #target='m' maximal  ='c' closed  ='a' all
    print(str(i)+"%:" + str(len(itemsets)))

    f = open(str(i)+"support-maximal["+str(len(itemsets))+"].txt", "w") 
    for j in range(0,len(itemsets)):
        f.write(""+ str(i) + str(itemsets[j]))
        f.write("\n\n")
    f.close()

#Ass RULES
for i in percent:
    for k in percent:
        rules = apriori(baskets, supp=i, zmin=2, target='r', conf=k, report='ascl') #report a:number of transaction, s:supporto, c:confidence, l:lift
        print(str(i)+"%:" + str(len(rules)))

        f = open("rules "+str(i)+"support-"+str(k)+"confidence["+str(len(rules))+"].txt", "w") 
        for j in range(0,len(rules)):
            f.write(""+ str(i) + str(rules[j]))
            f.write("\n\n")
        f.close()
