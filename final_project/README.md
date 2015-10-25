# Data Analyst Nanodegree Final Project by Paulo Casaretto

## Directory structure

Along with project resouces, and expected python scritps there are three more notable files in this directory.

* Machine Learning Final Project.ipynb : iPython notebook used for exploration
* exploration.py : python script used for evaluating and testing different classifiers
* sources.txt : where I note the resources used for this project

## Answers for the proposed questions

https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/pub?embedded=true

The goal of this project is to build an algorithm capable of identifying possible fraudsters from Enron's financial and email dataset.

### Data Exploration

This dataset has 146 data points and is very unbalanced with only 18 data points marked as POIs and 128 as not.
Most of it's 21 features are numeric and there is a lot of missing data (e.g. the feature restricted_stock_deferred has almost no information).

### Outlier investigation

Analyzing salary data, I found a clear outlier that was the sum of all other data points. A clear data entry mistake. This data point was excluded from all posterior analysis.


### Create new features

Looking at the mail exchanged features I had an insight. The idea was to extract what percentage of total messages each person exchanged with a POI. The hipothesis is that a higher ratio of messages exchanged with POIs might indicate that you are a POI yourself. 

The results were interesting. The ratio between messages sent to POIs and total messages sent had a higher correlation coefficient than the original features.

### Intelligently select features

To intelligently select features, I used a feature union between PCA and SelectKBest. I then used grid search to select both the optimal number of components for the PCA and also the K original features to use.

### Properly scale features

Features were always scaled using a MinMaxScaler.

### Pick and tune an algorithm & Validate and Evaluate

A multitude of different classifiers and parameters were tested for this dataset.
Grid search was used to select an optimal classifier. The choice of best classifier was made using a stratified shuffle split with scoring set to f1.
Accuracy would be a very poor choice for validation since the datase is very unbalanced. If I had chosen it I could have ended up with a classifier that classified all data points as non-POIs. I chose to use the f1 score to evaluate and choose the best algorithm.

List of evaluated algorithms:
* AdaBoost
* SVC
* Random Forest
* K nearest neighbors
* Logistic Regression Classifier
* LDA

Logistic Regression showed the best results having an accuracy, precision and recall of 0.82, 0.39, 0.61 respectively.
The interpretation of these is highly dependent on the application of the project.
I consider having recall being greater than precision a positive aspect of the results as I imagine this project being used in assisting finding possible fraudsters that might have slipped under the radar. In order to bring these to trial, there would be a thorough investigation. A higher false positive ratio here in therefore not damaging to anyone.
If this was not the case, for example, if the algorithm was being used to generate a public list of possible suspects, we would need to be more careful.


