#!/usr/bin/python
"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl",
                             "r"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    features, labels,
    test_size=0.3,
    random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

from sklearn.metrics import *
print(confusion_matrix(y_test, clf.predict(X_test)))
