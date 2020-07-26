# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:54:07 2019

@author: Paolo
"""
##import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
##import mglearn
from sklearn.svm import SVC
from collections import Counter


#IMPORT DATA
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\PythonScript_HighLowNN\\7Close.csv',sep=";",decimal=",")

y=df.loc[0:39900 , "y"]

#valori X
X=df.loc[0:39900 , "Close":"Close7"]
X=X-0.5

print(sorted(Counter(X).items()))
print(sorted(Counter(y).items()))




#
#
###from sklearn.model_selection import train_test_split##
#
###importa file con dati per training e test
#df = pd.read_csv('C:\\Users\\Paolo\\Documents\\Python Scripts\\ANN26mayR.csv',sep=";")
#y = df.y
#X = df.drop('y', axis=1)
#
###effettua lo split dei dati
#
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
#
#
#classifier = KNeighborsClassifier(n_neighbors=5)  
#classifier.fit(X_train, y_train) 

X_train = X[1:34500]
y_train = y[1:34500]

#X_test = X[34501:39982]
#y_test = y[34501:39982]

X_test = X[34500:39982]
y_test = y[34500:39982]

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(initial_data['diagnosis'],label="Sum")

plt.show()




###carica algoritmo kn, assegna punti di controllo e training##
#knn = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',
#           metric_params=None, n_jobs=None, n_neighbors=10, p=12,
#           weights='distance')
#knn.fit(X_train,y_train)

##carica algoritmo vector classification##
knn=SVC(kernel="linear",C=10,random_state=1000, class_weight='balanced')
knn.fit(X_train,y_train)

##predizione applicata alle features della X di test##
y_pred=knn.predict(X_test)
print(y_pred)
print(y_test)

##valore medio di accuracy##
print("score:{:.2f}".format(np.mean(y_pred == y_test)))
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test, y_pred)
print(confusion)

print(classification_report(y_test,y_pred))


##predizione on the fly a modello gia creato##
#X_new=np.array([[0.113,-0.108,0.108,-0.119]])
#prediction=knn.predict(X_new)
#print(prediction)


##carica il file di test finale con i nuovi valori e lo suddivide features e output
df2 = pd.read_csv('testLast.csv')
y2 = df2.y
X2 = df2.drop('y', axis=1)

##predizione sul file di prova con ultimi valori
prediction2=knn.predict(X2)
print(prediction2)
print(y2)
print("score:{:.2f}".format(np.mean(prediction2 == y2)))
print(classification_report(y2,prediction2))

import pandas as pd
import matplotlib.pyplot as plt
import mglearn
pd.plotting.scatter_matrix(X_train, c=y_train,figsize=(15,15), marker='o',hist_kwds={'bins':20}, s=60,alpha=8,cmap=mglearn.cm3)

handles = [plt.plot([],[],color=plt.cm.brg(i/2.), ls="", marker=".", \
                    markersize=np.sqrt(10))[0] for i in range(3)]
labels=["Label A", "Label B", "Label C"]
plt.legend(handles, labels, loc=(1.02,0))





