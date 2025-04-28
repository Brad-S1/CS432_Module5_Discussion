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

TrainingData, TestingData = train_test_split(MyDataSet, test_size=.4)
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





