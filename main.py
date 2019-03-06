#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import sys

%matplotlib inline

#1.
df = pd.read_csv("./train/train.csv")
df.describe ()
