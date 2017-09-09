# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:04:35 2017

@author: whuang67
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Load datasets
os.chdir("C:/users/whuang67/downloads/ml_master/Capstone")
bank_data = pd.read_csv("bank-full.csv", sep = ";")
print("Dataset has {} observations and {} variables in total.".format(
        bank_data.shape[0], bank_data.shape[1]))

#### Categorical and numerical variables
cat_var = ["job", "marital", "education", "default", "housing", "loan",
           "contact", "poutcome"]
cont_var = ["age", "balance", "duration", "campaign", "pdays", "previous"]

#### Removed variables
removed_var = ["day", "month"]


#### Frequency table of variable y
print(bank_data["y"].value_counts())
print(bank_data["y"].value_counts(normalize = True))

#### Scatter matrix plot
pd.plotting.scatter_matrix(bank_data[cont_var], alpha = 0.3,
                           figsize = (14, 8), diagonal = "kde")
# plt.suptitle("Figure 1: Scatter Matrix Plot of Continuous Variables", y = 0.02)
plt.show()
############ Outliers #######################
np.argmax(bank_data["previous"])
bank_data = bank_data.drop(29182)
################################################

#### Scatter matrix plot without outliers
pd.plotting.scatter_matrix(bank_data[cont_var], alpha = 0.3,
                           figsize = (14, 8), diagonal = "kde")
# plt.suptitle("Figure 2: Scatter Matrix Plot of Continuous Variables (without outlier)", y = 0.02)
plt.show()

#### Barplot of categorical variables
def barplot(var):
    val = bank_data[var].value_counts()
    x = val.index
    y = val.values
    pos = range(len(val.values))
    plt.bar(pos, y)
    if var == "job":
        plt.xticks(pos, x, rotation = 25)
    else:
        plt.xticks(pos, x)
    plt.title("Barplot of "+var)


plt.figure(figsize = (14, 16))
for count in range(1, 9):
    plt.subplot(int("4"+"2"+str(count)))
    var = cat_var[8-count]
    barplot(var)
# plt.suptitle("Figure 3: Barplots of Categorical Variables", y = 0.08)
plt.show()


bank_data1 = bank_data.copy()

####################### from preprocessing import
### One hot
all_var = cont_var+cat_var
all_var.remove("poutcome")
X = pd.get_dummies(bank_data1[all_var], drop_first = True)
y = pd.get_dummies(bank_data1[["y"]], drop_first = True)
columns = X.columns

### Min-max Scaler
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)

### Variable Importance 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "gini",
                             random_state = 1).fit(X, y)

importance = pd.Series(clf.feature_importances_, index = columns).sort_values()
pos = range(len(importance))

plt.figure(figsize = (14, 8))
plt.barh(pos, importance.values)
plt.yticks(pos, importance.index)
# plt.suptitle("Figure 4: Variable Importance", y = 0.08)
plt.show()

X = pd.get_dummies(
        bank_data[["duration", "balance", "age", "pdays", "housing", "campaign"]],
        drop_first = True)
X = MinMaxScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify = y, test_size = 0.2, random_state = 1)

y_train["y_yes"].value_counts(normalize = False)
y_test["y_yes"].value_counts(normalize = False)

##### Benchmark ######
from sklearn.metrics import precision_score, recall_score
max_age = max(bank_data["age"])
def age_classifier(x):
    if x > 65/max_age:
        output = 1
    else:
        output = 0
    return output
vfunc = np.vectorize(age_classifier)
# Train
y_pred_benchmark = vfunc(X_train[:, 2])
print((y_pred_benchmark == y_train["y_yes"].values).mean())
print(precision_score(y_train["y_yes"].values,
                      y_pred_benchmark))
print(recall_score(y_train["y_yes"].values,
                      y_pred_benchmark))
# Test
y_pred_benchmark_test = vfunc(X_test[:, 2])
print((y_pred_benchmark_test == y_test["y_yes"].values).mean())
print(precision_score(y_test["y_yes"].values,
                      y_pred_benchmark_test))
print(recall_score(y_test["y_yes"].values,
                   y_pred_benchmark_test))

##### Logistic Regression ######
from sklearn.linear_model import LogisticRegression
# Train
clf_LR = LogisticRegression().fit(X_train, y_train["y_yes"])
print((clf_LR.predict(X_train) == y_train["y_yes"].values).mean())
print(precision_score(y_train["y_yes"].values,
                      clf_LR.predict(X_train)))
print(recall_score(y_train["y_yes"].values,
                   clf_LR.predict(X_train)))

# Test
print((clf_LR.predict(X_test) == y_test["y_yes"].values).mean())
print(precision_score(y_test["y_yes"].values,
                      clf_LR.predict(X_test)))
print(recall_score(y_test["y_yes"].values,
                   clf_LR.predict(X_test)))

##### KNN ######
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
parameters = {"n_neighbors": [5, 7, 9, 11, 13, 15, 17]}
clf_KNN = GridSearchCV(KNeighborsClassifier(),
                       parameters,
                       cv = 5).fit(X_train, y_train["y_yes"])

clf_KNN_best = clf_KNN.best_estimator_
# Train
print((clf_KNN_best.predict(X_train) == y_train["y_yes"].values).mean())
print(precision_score(y_train["y_yes"].values,
                      clf_KNN_best.predict(X_train)))
print(recall_score(y_train["y_yes"].values,
                   clf_KNN_best.predict(X_train)))
# Test
print((clf_KNN_best.predict(X_test) == y_test["y_yes"].values).mean())
print(precision_score(y_test["y_yes"].values,
                      clf_KNN_best.predict(X_test)))
print(recall_score(y_test["y_yes"].values,
                   clf_KNN_best.predict(X_test)))

##### Random Forest 50 trees ######
from sklearn.ensemble import RandomForestClassifier

parameters = {"n_estimators": list(range(50, 350, 50)),
              "max_features": [1, 2, 3]} # sqrt(total feature number)
clf_RF = GridSearchCV(RandomForestClassifier(random_state = 1,
                                             verbose = 1),
                      parameters).fit(X_train, y_train["y_yes"])
clf_RF_best = clf_RF.best_estimator_
# Train
print((clf_RF_best.predict(X_train) == y_train["y_yes"].values).mean())
print(precision_score(y_train["y_yes"].values,
                      clf_RF_best.predict(X_train)))
print(recall_score(y_train["y_yes"].values,
                   clf_RF_best.predict(X_train)))
# Test
print((clf_RF_best.predict(X_test) == y_test["y_yes"].values).mean())
print(precision_score(y_test["y_yes"].values,
                      clf_RF_best.predict(X_test)))
print(recall_score(y_test["y_yes"].values,
                   clf_RF_best.predict(X_test)))

##### MLP Neural Network ######
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
model = Sequential()
model.add(Dense(50, activation = "relu", input_dim = X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(30, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation = "sigmoid"))

model.summary()


model.compile(loss = "binary_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

from keras.callbacks import ModelCheckpoint  
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)
y_train_MLP = np_utils.to_categorical(y_train)
model.fit(X_train, y_train_MLP,
          epochs = 20, callbacks = [checkpointer], verbose = 2)

print(model.evaluate(X_train, y_train_MLP, verbose = 0)[-1])
y_pred = model.predict(X_train)
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(y_train_MLP)):
    if y_train_MLP[i][1] == 1 and y_pred[i][0] <= y_pred[i][1]:
        TP += 1
    elif y_train_MLP[i][0] == 1 and y_pred[i][0] > y_pred[i][1]:
        TN += 1
    elif y_train_MLP[i][1] == 1 and y_pred[i][0] > y_pred[i][1]:
        FN += 1
    elif y_train_MLP[i][0] == 1 and y_pred[i][0] <= y_pred[i][1]:
        FP += 1
    else:
        print("It shouldn't be printed!!!")

print("Recall:", TP/(TP+FN))
print("Precision:", TP/(TP+FP))
print((TP+TN)/(TP+FP+TN+FN))

### Test
y_test_MLP = np_utils.to_categorical(y_test)


print(model.evaluate(X_test, y_test_MLP, verbose = 0)[-1])
y_pred = model.predict(X_test)
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(y_test_MLP)):
    if y_test_MLP[i][1] == 1 and y_pred[i][0] <= y_pred[i][1]:
        TP += 1
    elif y_test_MLP[i][0] == 1 and y_pred[i][0] > y_pred[i][1]:
        TN += 1
    elif y_test_MLP[i][1] == 1 and y_pred[i][0] > y_pred[i][1]:
        FN += 1
    elif y_test_MLP[i][0] == 1 and y_pred[i][0] <= y_pred[i][1]:
        FP += 1
    else:
        print("It shouldn't be printed!!!")

print("Recall:", TP/(TP+FN))
print("Precision:", TP/(TP+FP))
print((TP+TN)/(TP+FP+TN+FN))




##### MLP Neural Network Larger learning rate ######
from keras import optimizers

model1 = Sequential()
model1.add(Dense(50, activation = "relu", input_dim = X_train.shape[1]))
model1.add(Dense(30, activation = "relu"))
model1.add(Dropout(0.2))
model1.add(Dense(10, activation = "relu"))
model1.add(Dropout(0.2))
model1.add(Dense(2, activation = "sigmoid"))

model1.summary()
model1.compile(loss = "binary_crossentropy",
               optimizer = "rmsprop",
               metrics = ["accuracy"])
opt = optimizers.RMSprop(lr = 0.05)

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best2.hdf5', 
                               verbose=1, save_best_only=True)
model1.fit(X_train, y_train_MLP,
           epochs = 20, callbacks = [checkpointer], verbose = 2)
## Train 
print(model1.evaluate(X_train, y_train_MLP, verbose = 0)[-1])

y_pred = model1.predict(X_train)
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(y_train_MLP)):
    if y_train_MLP[i][1] == 1 and y_pred[i][0] <= y_pred[i][1]:
        TP += 1
    elif y_train_MLP[i][0] == 1 and y_pred[i][0] > y_pred[i][1]:
        TN += 1
    elif y_train_MLP[i][1] == 1 and y_pred[i][0] > y_pred[i][1]:
        FN += 1
    elif y_train_MLP[i][0] == 1 and y_pred[i][0] <= y_pred[i][1]:
        FP += 1
    else:
        print("It shouldn't be printed!!!")

print("Recall:", TP/(TP+FN))
print("Precision:", TP/(TP+FP))
print((TP+TN)/(TP+FP+TN+FN))

## Test
print(model1.evaluate(X_test, y_test_MLP, verbose = 0)[-1])

y_pred = model1.predict(X_test)
TP, TN, FP, FN = 0, 0, 0, 0
for i in range(len(y_test_MLP)):
    if y_test_MLP[i][1] == 1 and y_pred[i][0] <= y_pred[i][1]:
        TP += 1
    elif y_test_MLP[i][0] == 1 and y_pred[i][0] > y_pred[i][1]:
        TN += 1
    elif y_test_MLP[i][1] == 1 and y_pred[i][0] > y_pred[i][1]:
        FN += 1
    elif y_test_MLP[i][0] == 1 and y_pred[i][0] <= y_pred[i][1]:
        FP += 1
    else:
        print("It shouldn't be printed!!!")

print("Recall:", TP/(TP+FN))
print("Precision:", TP/(TP+FP))
print((TP+TN)/(TP+FP+TN+FN))