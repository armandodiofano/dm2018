import pandas
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

df = pandas.read_csv("credit_default_corrected_train.csv")

def logic(row):
    if row[1] == -2:
        return "female";
    if row[1] == -1:
        return "female";
    if row[2] == -2:
        return "female";
    if row[2] == -1:
        return "female";
    if row[3] == -2:
        return "female";
    if row[5] == -2:
        return "female";
    if row[4] == -2:
        return "female";
    if row[0] == 0:
        return "female";
    if row[0] == -1:
        return "female";
    if row[3] == -1:
        return "female";
    if row[5] == -1:
        return "female";
    if row[4] == -1:
        return "female";
    return "male"
'''
['ps-aug [-1, 0)', 'ba-jun [-15910.0, 6717.7)', 'ba-sep [-9802.0, 15869.8)', 'ps-jun [-2, -1)', 'ba-may [15421.2, 38372.3)', 'ps-aug [-2, -1)', 'ps-apr [-2, -1)', 'ps-sep [0, 1)', 'ps-may [-1, 0)', 'ba-jul [-15910.0, 7900.0)', 'ps-apr [-1, 0)', 'ba-apr [17105.2, 41826.9)', 'ps-may [-2, -1)', 'ba-may [-7529.9, 15421.2)', 'ba-jul [7900.0, 31710.0)', 'ps-jun [-1, 0)', 'limit [10000.0, 59090.9)', 'ps-jul [-2, -1)', 'ba-jun [6717.7, 29345.5)', 'ba-aug [20552.2, 50661.9)', 'ps-sep [-1, 0)', 'ba-sep [15869.8, 41541.6)', 'ps-jul [-1, 0)'])
'''


X = df.drop('sex',axis=1).values
Y = df['sex'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.25, 
                                                    random_state=100)

Ypred=[logic(row) for row in X_test]
print (accuracy_score(y_test,Ypred))

import pandas
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

df = pandas.read_csv("credit_default_corrected_train.csv")
'''
def logic(row):
    if np.count_nonzero(row == -1) >= 2:
        return 0
    return 1

df=df[['ps-sep', 'ps-aug', 'ps-jul', 'ps-jun', 'ps-may', 'ps-apr', 'credit_default']]

X = df.drop('credit_default',axis=1).values
Y = df['credit_default'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.25, 
                                                    random_state=100, 
                                                    stratify=Y)

Ypred=[logic(row) for row in X_test]

print (accuracy_score(y_test,Ypred))
print (recall_score(y_test,Ypred))
print (precision_score(y_test,Ypred))
print (f1_score(y_test,Ypred))
