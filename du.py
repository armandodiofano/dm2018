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

df.to_csv("credit_default_corrected_train.csv", index=False)
