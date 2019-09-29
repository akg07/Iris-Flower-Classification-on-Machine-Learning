import pandas as pd
iris = pd.read_csv('iris.csv')
iris.head()

iris = pd.read_csv('iris.csv', na_values = ['NA'])

iris.describe()

import matplotlib.pyplot as plt
import seaborn as sb

sb.pairplot(iris.dropna(), hue = 'class')

iris.loc[iris['class'] == 'versicolor', 'class'] == 'Iris-versicolor'
iris.loc[iris['class'] == 'Iris-setossa', 'class'] == 'Iris-setosa'

iris['class'].unique()

iris = iris.loc[(iris['class'] != 'Iris-setosa') | (iris['sepal_width_cm'] >= 2.5)]
iris.loc[iris['class'] == 'Iris-setosa', 'sepal_width_cm'].hist()

iris.loc[(iris['class'] == 'Iris-versicolor') & (iris['sepal_length_cm'] < 1.0)]

iris.loc[iris['sepal_length_cm'].isnull() | iris['sepal_width_cm'].isnull() | iris['petal_length_cm'].isnull() | iris['petal_width_cm'].isnull()]

# there is no missing values in the dataset if there is any misisng value then we will tackel with this code
"""avg = iris.loc[iris['class'] == 'class_name', 'Missing_feild_name'].mean()

iris.loc[(iris['class'] == 'class_name') & (iris['missing_feild_name'].isnull()), 'missing_feild_name'] = avg

iris.loc[(iris['class'] == 'class_name') & (iris['missing_feild_name'] == avg)]"""

# for saving the clean data into new dataset   iris ----> iris_clean
"""iris.to_csv('iris_clean.csv', index = false)
iris_clean = pd.read_csv('iris_clean.csv')"""

sb.pairplot(iris, hue = 'class')

sb.pairplot(iris)

plt.figure(figsize = (10, 10))

for col, coloum in enumerate(iris):
    if (coloum == 'class'):
        continue
    plt.subplot(2, 2, col+1)
    sb.violinplot(x = 'class', y = coloum, data = iris)
    

X = iris.iloc[:, :-1].values
y = iris.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state = 1)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)

acc = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    classifier_acc = classifier.score(X_test, y_test)
    acc.append(classifier_acc)

plt.hist(acc)

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(classifier, X, y, cv = 10)

import numpy as np
plt.hist(cvs)
plt.title('average score = {}'.format(np.mean(cvs)))

# Apply grid search method
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
dct = DecisionTreeClassifier()
parameter_grid = {'max_depth': [1,2,3,4,5], 'max_features': [1,2,3,4]}
cross_Validation = StratifiedKFold(n_splits = 10)
g_search = GridSearchCV(dct, param_grid = parameter_grid, cv = cross_Validation)
g_search.fit(X,y)
print('best score: {}'.format(g_search.best_score_))
print('best parameters: {}'.format(g_search.best_params_))

# Apply visualization
g_visualization = g_search.cv_results_['mean_test_score']
g_visualization.shape = (5,4)
sb.heatmap(g_visualization, cmap = 'Blues', annot = True)
plt.xticks(np.arange(5)+0.5, g_search.param_grid['max_features'])
plt.yticks(np.arange(4)+0.5, g_search.param_grid['max_depth'])
plt.xlabel('max feature')
plt.ylabel('max depth')

dct = DecisionTreeClassifier()
parameter_grid = {'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'],
                  'max_depth': [1,2,3,4,5], 'max_features': [1,2,3,4]}
cross_Validation = StratifiedKFold(n_splits = 10)
g_search = GridSearchCV(dct, param_grid = parameter_grid, cv = cross_Validation)
g_search.fit(X,y)
print('best score: {}'.format(g_search.best_score_))
print('best parameters: {}'.format(g_search.best_params_))

dct = g_search.best_estimator_
dct

#create some visuales for demo classifier
dt_scores = cross_val_score(dct, X, y, cv = 10)
sb.boxplot(dt_scores)
sb.stripplot(dt_scores, jitter = True, color = 'black')

# Apply random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
p_grid = {'n_estimators': [10, 25, 50, 100], 'criterion': ['gini', 'entropy'], 'max_features': [1,2,3,4]}
cross_validation = StratifiedKFold(n_splits = 10)
g_search = GridSearchCV(rfc, param_grid = p_grid, cv = cross_validation)
g_search.fit(X, y)
print('best score: {}'.format(g_search.best_score_))
print('best parameter: {}'.format(g_search.best_params_))
g_search.best_estimator_

rfc =  g_search.best_estimator_
rf_df = pd.DataFrame({'accuracy': cross_val_score(rfc, X, y, cv = 10), 'classifier': ['Random Forest']*10})
dt_df = pd.DataFrame({'accuracy': cross_val_score(dct, X, y, cv =10), 'classifier': ['Decision tree']*10})
both_df = rf_df.append(dt_df)
sb.boxplot(x = 'classifier', y ='accuracy', data = both_df )
sb.stripplot(x = 'classifier', y = 'accuracy', data = both_df, jitter = True, color = 'black')
'''   |
-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------
'''
import pandas as pd
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# We can jump directly to working with the clean data because we saved our cleaned data set
iris = pd.read_csv('iris.csv')

# Testing our data: Our analysis will stop here if any of these assertions are wrong

# We know that we should only have three classes
assert len(iris['class'].unique()) == 3

# We know that sepal lengths for 'Iris-versicolor' should never be below 2.5 cm
assert iris.loc[iris['class'] == 'Iris-versicolor', 'sepal_length_cm'].min() >= 2.5

# We know that our data set should have no missing measurements
assert len(iris.loc[(iris['sepal_length_cm'].isnull()) |
                               (iris['sepal_width_cm'].isnull()) |
                               (iris['petal_length_cm'].isnull()) |
                               (iris['petal_width_cm'].isnull())]) == 0

X = iris[['sepal_length_cm', 'sepal_width_cm',
                             'petal_length_cm', 'petal_width_cm']].values

y = iris['class'].values

# This is the classifier that came out of Grid Search
random_forest_classifier = RandomForestClassifier(criterion='gini', max_features=3, n_estimators=50)

# All that's left to do now is plot the cross-validation scores
rf_classifier_scores = cross_val_score(random_forest_classifier, X, y, cv=10)
sb.boxplot(rf_classifier_scores)
sb.stripplot(rf_classifier_scores, jitter=True, color='black')

# ...and show some of the predictions from the classifier
(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(X, y, test_size=0.25)

random_forest_classifier.fit(X_train, y_train)

for input_features, prediction, actual in zip(X_train[:10],
                                              random_forest_classifier.predict(X_test[:10]),
                                              y_test[:10]):
    print('{}\t-->\t{}\t(Actual: {})'.format(input_features, prediction, actual))