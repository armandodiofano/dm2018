import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv("credit_default_train.csv")

#education, status, sex, age

df["education"] = df["education"].fillna(df["education"].mode()[0])
df["status"] = df["status"].fillna(df["status"].mode()[0])
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

age_mean = int(df[df["age"] != -1]["age"].mean())
df["age"] = df["age"].replace(to_replace=-1, value=age_mean)

df["credit_default"] = df["credit_default"].replace({"no": 0, "yes": 1})
'''
ps = ['ps-sep', 'ps-aug', 'ps-jul', 'ps-jun', 'ps-may', 'ps-apr']
pa = ['pa-sep', 'pa-aug', 'pa-jul', 'pa-jun', 'pa-may', 'pa-apr']
ba = ['ba-sep', 'ba-aug', 'ba-jul', 'ba-jun', 'ba-may', 'ba-apr']

df['ps'] = df[ps].sum(axis=1)
df['pa'] = df[pa].sum(axis=1)
df['ba'] = df[ba].sum(axis=1)

df = df.drop(ps, axis=1)
df = df.drop(pa, axis=1)
df = df.drop(ba, axis=1)
'''
df.to_csv("credit_default_corrected_train.csv", index=False)
