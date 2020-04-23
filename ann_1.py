# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:47:05 2020

@author: HSU6KOR
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:26:46 2020

@author: HSU6KOR
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
cat_col = X[:,1]
onehotencoder = OneHotEncoder()
one_hot = onehotencoder.fit_transform(cat_col.reshape(-1,1)).toarray()
X = np.delete(X, 1, axis =1)
X = np.concatenate((one_hot, X), axis =1)
X = X[:, 1:] # Inorder to dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##building ANN
#import keras library
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()
#adding the input layer and first hidden layer
#units = number of nodes in hidden layer inputlayer+outputlayer/2
classifier.add(Dense(units = 6, kernel_initializer ='uniform', activation ='relu',input_dim = 11))

#adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer ='uniform', activation ='relu'))
#adding output layer
classifier.add(Dense(units = 1, kernel_initializer ='uniform', activation ='sigmoid'))

#compiling the ANN applying SGD(stochastic gradient descent) on the network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# this does if y_pred >0.5 is 1 else 0
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)