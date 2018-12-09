import pandas

df = pandas.read_csv("credit_default_train.csv")

#print("Do missing values exist in our dataset?")
#df.isnull().any() #check if some columns have missing values

#education, status, sex, age

df["education"] = df["education"].fillna(df["education"].mode()[0])
df["status"] = df["status"].fillna(df["status"].mode()[0])
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

age_mean = int(df[df["age"] != -1]["age"].mean())
df["age"] = df["age"].replace(to_replace=-1, value=age_mean)

#data transformations

df["credit_default"] = df["credit_default"].replace({"no": 0, "yes": 1})
df["education"] = df["education"].replace({"graduate school": 0, "high school": 1, "university": 2, "others": 3})
df["status"] = df["status"].replace({"married": 0, "single": 1, "others": 2})
df["sex"] = df["sex"].replace({"female": 1, "male": 2})

df.to_csv("credit_default_corrected_train.csv", index=False)

