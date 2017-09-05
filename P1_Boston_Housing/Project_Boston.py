# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 00:09:49 2017

@author: whuang67
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import visuals as vs
data = pd.read_csv("housing.csv")
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)


minimum_price = np.min(prices)
maximum_price = np.max(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)
prices.describe()
np.percentile(prices, 25)

np.corrcoef(features.RM, prices)
np.corrcoef(features.LSTAT, prices)
np.corrcoef(features.PTRATIO, prices)

def performance_metric(y_true, y_predict):
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    return score

score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        features, prices, test_size =.2, random_state = 0)

vs.ModelLearning(features, prices)
vs.ModelComplexity(X_train, y_train)

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
for train, test in cv_sets.split(X_train, y_train):
    print 'train', train, 'test', test


def fit_model(X, y):

    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 1)

    regressor = DecisionTreeRegressor()
    params = {'max_depth': range(1, 11)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, cv = cv_sets, scoring = scoring_fnc)
    grid = grid.fit(X, y)
    return grid.best_estimator_
reg = fit_model(X_train, y_train)
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)
    
for  b in price.describe().values:
    print  b
c = price.describe()