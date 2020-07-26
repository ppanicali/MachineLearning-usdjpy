# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 03:56:01 2019

@author: Paolo
"""

# -*- coding: utf-8 -*-
"""
NNet
"""

import pandas as pd
#import mglearn
from sklearn import preprocessing
import numpy as np


#import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense


from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


#read the file with data
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\PythonScripts_NEW\\usdjpy1h_ml_y.csv',sep=";")

#divide data and outcome
X=df.loc[1:1000 , "stochfast":"return%slow"]
y=df.loc[1:1000 , "y"]

#normalize data 0-1
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)




#plot data to see clusters
#grr = pd.plotting.scatter_matrix(X, c=y, figsize=(15, 15), marker='o',
#hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


#X_train, X_test, y_train, y_test = train_test_split(X, dummy_y,test_size=0.3)



encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(14, input_dim=7, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=10, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

