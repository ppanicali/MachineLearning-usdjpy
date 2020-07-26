# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 04:43:07 2019

epoch 18 best 0.69 su test

[(0, 6325), (1, 3666)]
9991/9991 [==============================] - 1s 104us/step

acc: 67.39%
[[3853 2472]
 [ 786 2880]]
              precision    recall  f1-score   support

           0       0.83      0.61      0.70      6325
           1       0.54      0.79      0.64      3666

   micro avg       0.67      0.67      0.67      9991
   macro avg       0.68      0.70      0.67      9991
weighted avg       0.72      0.67      0.68      9991

@author: Paolo
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential 
from keras.layers import Dense 
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#IMPORT DATA
#df = pd.read_csv('C:\\Users\\Paolo\\Documents\\PythonScript_HighLowNN\\usdjpy20k_1h_hl80pts.csv',sep=";")
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\PythonScript_HighLowNN\\allfeatures.csv',sep=";")

#DIVIDE DATA AND OUTCOME
y=df.loc[2:20000 , "y"]

#converte il dataframe in incrementi percentuali dopo avere estratto gli outcome e li passa a X
df2=df.pct_change( fill_method='bfill')*1000
X=df2.loc[2:20000 , "High":"stdvslow"]

#normalizza gli incrementi percentuali, si puo provare anche altra formula
X = preprocessing.normalize(X)

#divide e fa shuffle di train e test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

#undersampling della classe maggioritaria solo nei blocchi train
rus = RandomUnderSampler(random_state=0)
X_train, y_train = rus.fit_resample(X_train, y_train)
#X_test, y_test = rus.fit_resample(X_train, y_train)

#bilanciamento delle classi nei tensori di training e di test
print(sorted(Counter(y_train).items()))
print(sorted(Counter(y_test).items()))
##---------------------------------------------------------------

#crea modello rete neurale per la predizione dei dati high e low e outcome di +90 pt rispoetto
#la chiusura della 4rta barra invece della rottura del massimo delle 4 barre successime, ha un 
#bilanciamento di circa il 42% di positivi

input_dim = 8
model = Sequential()
model.add(Dense(24, input_dim = input_dim , activation = 'sigmoid'))
model.add(Dense(24, activation = 'sigmoid'))
model.add(Dense(12, activation = 'sigmoid'))
model.add(Dense(6, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

####invece di accuracy precision????

history=model.fit(X_train, y_train, epochs = 15, batch_size = 10, class_weight = 'auto', validation_data=(X_test,y_test))

print(sorted(Counter(y_test).items()))

scores = model.evaluate(X_test, y_test)
y_pred=model.predict_classes(X_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test, y_pred)
print(confusion)

print(classification_report(y_test,y_pred))










