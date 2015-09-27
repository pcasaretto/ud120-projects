#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

for i, pair in enumerate(features_train):
    features_train[i].append(pair[0]**2 + pair[1]**2)

for i, pair in enumerate(features_test):
    features_test[i].append(pair[0]**2 + pair[1]**2)

print features_train[0]

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
print "svc", clf.score(features_test, labels_test)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
print "naive bayes", clf.score(features_test, labels_test)

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print "tree", clf.score(features_test, labels_test)

from sklearn import ensemble
from time import time

clf = ensemble.AdaBoostClassifier()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

print "accuracy: ", clf.score(features_test, labels_test)
print "estimators: ", len(clf.estimators_)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
