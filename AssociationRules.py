import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from fim import apriori

import math

df = pd.read_csv("credit_default_corrected_train.csv", skipinitialspace=True, sep=',')
bins=math.ceil(math.log2(10000)+1)
statuses = df[['ps-sep', 'ps-aug', 'ps-jul', 'ps-jun', 'ps-may', 'ps-apr']]
to_cut=['limit','age','pa-sep', 'pa-aug', 'pa-jul', 'pa-jun', 'pa-may', 
'pa-apr','ba-sep', 'ba-aug', 'ba-jul', 'ba-jun', 'ba-may', 'ba-apr']
for i in to_cut:
    df[i] = pd.cut(df[i].astype(int), bins, precision=0, right=False)
    df[i] = i+": "+df[i].astype(str)
for i in statuses:
    df[i] = i+": "+df[i].astype(str)
df["credit_default"]="default: "+df["credit_default"].astype(str)
print(df.head())

baskets = df.values.tolist()
print("START")
itemsets = apriori(baskets, supp=10, zmin=2, target='a')  #supporto 10%
print('Number of itemsets:', len(itemsets))
rules = apriori(baskets, supp=10, zmin=2, target='r', conf=80, report='ascl') #supporto 10%  confidenza 80%
print('Number of rules:', len(rules))
