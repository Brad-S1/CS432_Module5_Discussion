#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:38:36 2025

@author: bradsommer
"""

# This file copied from module5_discussion_logistic_regression.py and modified 
# to perform linear regression instead.

## this code is adapted from Dr. G's logistic regression example code 
## found in Module 5 Exploration 2
## this code is adapted from Dr. G's linear regression example code 
## found in Module 4 Exploration 2

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

filepath = '/Users/bradsommer/Documents/School/OSU/CS 432/Module 5/Discussion/custom_dataset_linear_regression.csv'

MyDataSet=pd.read_csv(filepath)

print("My Data Set:\n",MyDataSet)

## Place all independent variables in to X and dependent variable into Y
Y = MyDataSet[["Habitability_Score"]]
X = MyDataSet.drop(["Habitability_Score", "Planet_ID"], axis=1)
print("dependent variable Y: \n", Y)
print("independent variable X: \n", X)

# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

### Train the model

## Split training and testing data (will need to remove Planet_ID as well)

# Print to make sure everything is correct
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.4)
print("Independent Variable Training Data:\n", x_train)
print("Dependent Variable Training Data:\n", y_train)
print("Independent Variable Testing Data:\n", x_test)
print("Dependent Variable Testing Data:\n", y_test)


#### Perform Linear Regression ####

## instantiate
MyLR = LinearRegression()

## Fit the model to X and Y of the Training data
My_LR_Model = MyLR.fit(x_train, y_train)

### Use model to predict test data ###
MyModelPredictions = My_LR_Model.predict(x_test)
print("My Model Predictions:\n", MyModelPredictions)

## compare to the actual labels
print("Actual test data labels:\n", y_test)

# ## create standard confusion matrix to compare actual and predicted labels
# MyCM = confusion_matrix(y_test, MyModelPredictions)
# print(MyCM)

# # use Seaborn to create a nice looking confusion matrix visualization
# sns.heatmap(MyCM, annot=True, cmap='Reds')

# Create a scatter plot to visualize predictions vs actual values
# Get predictions for training data too
train_predictions = My_LR_Model.predict(x_train)

# Create a new figure
plt.figure(figsize=(10, 7))

# Plot training data
plt.scatter(y_train, train_predictions, color='blue', alpha=0.7, label='Training data')

# Plot test data
plt.scatter(y_test, MyModelPredictions, color='red', alpha=0.7, label='Testing data')

# Calculate a single trend line based on ONLY the training data
z = np.polyfit(y_train.values.flatten(), train_predictions.flatten(), 1)
p = np.poly1d(z)

# Create line points
x_line = np.linspace(min(y_train.values.flatten()), max(y_train.values.flatten()), 100)
plt.plot(x_line, p(x_line), 'g-', label='Model trend line (training only)')

plt.xlabel('Actual Habitability Score')
plt.ylabel('Predicted Habitability Score')
plt.title('Linear Regression Model: Training and Testing Performance')
plt.legend()
plt.grid(True)
plt.show()

#### view properties of the model ####
## view accuracy score
print("Training Data Accuracy score: ", My_LR_Model.score(x_train, y_train))
print("Testing Data Accuracy score: ", My_LR_Model.score(x_test, y_test))

# evaluate model
print("mean_squared_error: ", mean_squared_error(y_test, MyModelPredictions))
print("mean_absolute_error: ", mean_absolute_error(y_test, MyModelPredictions))

## lets print the coefficients (weights) of the model
print("Model coefficients: ", My_LR_Model.coef_)
# we can also print the intercept (b)
print("Model intercept: ", My_LR_Model.intercept_)





