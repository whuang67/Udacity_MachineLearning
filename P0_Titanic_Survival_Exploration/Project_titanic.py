# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 18:46:32 2017

@author: whuang67
"""


import pandas as pd
# import visuals as vs

data = pd.read_csv("titanic_data.csv")
outcomes = data['Survived']
dat = data.drop('Survived', axis = 1)




def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    


def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger.Sex == 'male':
            predictions.append(0)
        else:
            predictions.append(1)
    
    # Return our predictions
    return pd.Series(predictions)


# Make the predictions
predictions = predictions_1(dat)
print accuracy_score(data['Survived'], predictions)


def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger.Sex == 'female':
            predictions.append(1)
        elif passenger.Age < 10:
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(dat)
print accuracy_score(data['Survived'], predictions)


def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if (passenger.Sex == 'female') & (passenger.Pclass != 3):
            predictions.append(1)
        elif passenger.Age < 8:
            predictions.append(1)
        else:
            predictions.append(0)

    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(dat)
print accuracy_score(data['Survived'], predictions)