#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys

%matplotlib inline

# some comments

#1.
df = pd.read_csv("./../train.csv")
df.describe (include = 'all')

for column in df.columns:
    if df[column].dtype == type (object):
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform (df[column])

y = df.target

#%%

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split (df, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# fit a model
lm = linear_model.LogisticRegression()
model = lm.fit (X_train, y_train)
predictions = lm.predict (X_test)

predictions[0:5]

## The line / model
plt.scatter (y_test, predictions)
plt.xlabel ("True Values")
plt.ylabel ("Predictions")
