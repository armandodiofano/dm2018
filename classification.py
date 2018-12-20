import pandas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn import tree
from IPython.display import Image, display
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

df = pandas.read_csv("credit_default_corrected_train.csv")

df = df[['ps-sep','ps-aug','ps-jul', 'ps-jun', 'ps-may', 'ps-apr', 'credit_default']]
attributes = [col for col in df.columns if col != 'credit_default']
X = df[attributes].values
y = df['credit_default']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=100, 
                                                    stratify=y)

clf = DecisionTreeClassifier(criterion='gini')

clf = DecisionTreeClassifier(criterion='gini', min_samples_split=150, min_samples_leaf=20, presort=True)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                    hidden_layer_sizes=(10, 10), random_state=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

scores = cross_val_score(clf, X, y, cv=10)
print('Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

test = pandas.read_csv("credit_default_test.csv")
test = test[['ps-sep','ps-aug','ps-jul', 'ps-jun', 'ps-may' ,'ps-apr']]

y_pred = clf.predict(test)
test = test.drop(['ps-sep','ps-aug','ps-jul', 'ps-jun', 'ps-may' ,'ps-apr'], axis=1)
test['credit_default'] = y_pred
test['credit_default'] = test['credit_default'].replace({0: 'no', 1: 'yes'})

test.to_csv('results.csv')

