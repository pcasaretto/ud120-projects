#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary']  # You will need to use more features

features_list += ['bonus', 'exercised_stock_options', 'total_stock_value']
features_list += ['salary', 'deferral_payments', 'total_payments',
                  'loan_advances', 'bonus', 'restricted_stock_deferred',
                  'deferred_income', 'total_stock_value', 'expenses',
                  'exercised_stock_options', 'other', 'long_term_incentive',
                  'restricted_stock', 'director_fees']
features_list += ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))

### Task 2: Remove outliers
del data_dict['TOTAL']

### Task 3: Create new feature(s)
for name in data_dict:

    try:
        data_dict[name]['from_poi_ratio'] = data_dict[name][
            'from_poi_to_this_person'
        ] / data_dict[name]['to_messages']
    except:
        data_dict[name]['from_poi_ratio'] = 'NaN'
    try:
        data_dict[name]['to_poi_ratio'] = data_dict[name][
            'from_this_person_to_poi'
        ] / data_dict[name]['from_messages']
    except:
        data_dict[name]['to_poi_ratio'] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


def train_optimal_classifier(clf, X, y, params, scale=False, folds=1000):
    pipeline = 0

    combined_features = FeatureUnion([("pca", PCA()),
                                      ("univ_select", SelectKBest())])

    if scale:
        pipeline = Pipeline([("minmax", MinMaxScaler()),
                             ("features", combined_features), ("clf", clf)])
    else:
        pipeline = Pipeline([("minmax", MinMaxScaler()),
                             ("features", combined_features), ("clf", clf)])

    param_grid = dict(features__pca__n_components=[0, 1, 5, 10, 20],
                      features__univ_select__k=list(range(0, len(X[0]))))

    for k, v in params.iteritems():
        param_grid["clf__" + k] = v

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cross_validation.StratifiedShuffleSplit(y, folds),
        verbose=1,
        scoring='f1',
        error_score=0,
        refit=True,
        )
    grid_search.fit(X, y)
    return (grid_search.best_estimator_, grid_search.best_score_)


classifiers = []

# AdaBoost
classifiers.append((AdaBoostClassifier(), {"n_estimators": [20, 25, 30, 40, 50, 100]}))

# SVC
classifiers.append((SVC(), {"C": [0.1, 1, 10], "kernel": ['rbf', 'linear']}))

# Random Forest
classifiers.append( (RandomForestClassifier(), {  "n_estimators":[2, 3, 5], "criterion": ('gini', 'entropy') }) )

# KNN
classifiers.append( (KNeighborsClassifier(), {"n_neighbors":[2, 5], "p":[2,3]}) )

# Logistic Regression
params = {  "C":[0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20],
                "tol":[10**-1, 10**-5, 10**-10],
                "class_weight":['auto']
                }
classifiers.append( (LogisticRegression(), params) )

# LDA
classifiers.append( (LDA(), {"n_components":[0, 1, 2, 5, 10]}) )


trained = []
for v in classifiers:
    clf = train_optimal_classifier(v[0], features, labels, v[1],
                                   scale=True,
                                   folds=1000)
    trained.append(clf)

best = max(trained, key=lambda r: r[1])

print 'Best classifier and score'
print best
print 'Steps'
print best[0].steps

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(best[0], my_dataset, features_list)
