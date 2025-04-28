#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:38:36 2025

@author: bradsommer
"""

## this code is adapted from Dr. G's logistic regression example code 
## found in Module 5 Exploration 2

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

filepath = '/Users/bradsommer/Documents/School/OSU/CS 432/Module 5/Discussion/custom_dataset.csv'

MyDataSet=pd.read_csv(filepath)

print("My Data Set:\n",MyDataSet)

### Train the model

## Split training and testing data (will need to remove Planet_ID as well)

# make sure training and testing data is balanced using stratify
TrainingData, TestingData = train_test_split(MyDataSet, test_size=.4, 
                                            stratify=MyDataSet["Habitable"],
                                            random_state=42)
print("Training Data:\n", TrainingData)
print("Testing Data:\n", TestingData)

## Save training labels and planet IDs
TrainingLabels = TrainingData["Habitable"]
TrainingPlanetIDs = TrainingData["Planet_ID"]
## Drop training labels and planet IDs
TrainingData = TrainingData.drop(["Habitable", "Planet_ID"], axis=1)

# Print to make sure everything is correct
print("The Training Labels are:\n", TrainingLabels)
print("The Training Data is:\n", TrainingData)

## Save testing labels and planet IDs
TestingLabels = TestingData["Habitable"]
TestingPlanetIDs = TestingData["Planet_ID"]
## Drop testing labels and planet IDs
TestingData = TestingData.drop(["Habitable", "Planet_ID"], axis=1)

# print to make sure everything is correct
print("The Testing labels are:\n", TestingLabels)
print("The Testing Data is:\n", TestingData)


#### Perform Logistic Regression ####

## instantiate
MyLR = LogisticRegression()

## Perform logistic regression on training data and training labels
My_LR_Model = MyLR.fit(TrainingData, TrainingLabels)

### Use model to predict test data ###

MyModelPredictions = My_LR_Model.predict(TestingData)
print("My Model Predictions:\n", MyModelPredictions)

## compare to the actual labels
print("Actual test data labels:\n", TestingLabels)

## create standard confusion matrix to compare actual and predicted labels
MyCM = confusion_matrix(TestingLabels, MyModelPredictions)
print(MyCM)

# use Seaborn to create a nice looking confusion matrix visualization
sns.heatmap(MyCM, annot=True, cmap='Greens')

#### view properties of the model ####
## view accuracy score
print("Training Data Accuracy score: ", My_LR_Model.score(TrainingData, TrainingLabels))
print("Testing Data Accuracy score: ", My_LR_Model.score(TestingData, TestingLabels))

## lets look at the prediction probabilities for the testing data
print("Prediction Probabilities:\n", My_LR_Model.predict_proba(TestingData))

## lets print the coefficients (weights) of the model
print("Model coefficients: ", My_LR_Model.coef_)
# we can also print the intercept (b)
print("Model intercept: ", My_LR_Model.intercept_)








