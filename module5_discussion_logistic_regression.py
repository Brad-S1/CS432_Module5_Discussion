#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:38:36 2025

@author: bradsommer
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

filepath = '/Users/bradsommer/Documents/School/OSU/CS 432/Module 5/Discussion/module5_discussion_logical_regression.py'

MyDataSet=pd.read_csv(filepath)

print(MyDataSet)


