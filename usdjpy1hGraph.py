# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 01:36:30 2019

@author: Paolo per avere i grafici plottati su una finestra andare alle preferenze, grafica, automatico
"""

import pandas as pd
import mglearn
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt



#read the file with data
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\PythonScripts_NEW\\usdjpy1h_ml_y.csv',sep=";")

#divide data and outcome
X=df.loc[1:100 , "stdvfast":"return%slow"]
y=df.loc[1:100 , "y"]

#normalize data 0-1
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)


#plot data to see clusters
grr = pd.plotting.scatter_matrix(X, c=y, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
