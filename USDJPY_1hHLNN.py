# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 04:43:07 2019

@author: Paolo
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing



#iIMPORT DATA
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\PythonScripts_NEW\\usdjpy1h_ml_y.csv',sep=";")

#DIVIDE DATA AND OUTCOME
X=df.loc[1:5000 , "stochfast":"return%slow"]
y=df.loc[1:5000 , "y"]

#NORMALIZE FROM 0 TO 1
x = X.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)


#ENCODE CATEGORIES 0 1 AND 2 TO BIT ENCODE FOR THE SOFTMAX FUNCTION OF THE NEURAL NET
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
#------------------------------------------------------------------



X_train, X_test, y_train, y_test = train_test_split(X, dummy_y,test_size=0.3)


#undersampling ora modifico i dati per avere numero uguale di classi dei tre tipi riducendo quella maggioritaria 0
#print("prima")
#print(sorted(Counter(y_train).items()))

rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(X_train, y_train)

#print("dopo")
#print(sorted(Counter(y_train).items()))
##---------------------------------------------------------------


input_dim = 9

model = Sequential()
model.add(Dense(36, input_dim = input_dim , activation = 'relu'))
model.add(Dense(18, activation = 'relu'))
model.add(Dense(9, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

history=model.fit(X_train, y_train, epochs = 200, batch_size = 10, class_weight = 'auto', validation_data=(X_test,y_test))


scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(15, acc, 'bo', label='Training acc')
plt.plot(15, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()














