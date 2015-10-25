#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

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
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

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
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

combined_features = FeatureUnion([("pca", PCA(n_components=20)), ("univ_select", SelectKBest(k=1))])
clf = LogisticRegression(C=100000, class_weight='auto', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', penalty='l2', random_state=None,
          solver='liblinear', tol=1e-05, verbose=0 )
clf = Pipeline([("minmax", MinMaxScaler()), ("features", combined_features), ("clf", clf)])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
