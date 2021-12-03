#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:56:57 2021

@author: ciaranmcdonnell
"""

"""Neural network classification of mammogram images as benign or not."""


import pandas as pd  
import numpy as np
from sklearn import preprocessing


# --------------- 1. Reading and massaging data ---------------
# First thing to do is simply read in the data
# Can also clean up the data in the same step. Documentation of pd.read_csv has lots of info on inputs

m_masses = pd.read_csv('mammographic_masses.data.txt',na_values=['?'],names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
m_masses.head(10)
m_masses.describe()


# Next to further check the data, are there any obvious ones to leave out (be careful),
# will there be any bias?
# A good idea is now to look at all of the data that has a NaN in it.

m_masses.loc[(m_masses['age'].isnull()) | (m_masses['shape'].isnull()) | (m_masses['margin'].isnull())| (m_masses['density'].isnull())]
#m_masses.loc[(m_masses['severity'].isnull())]

#Now it doesn't look like there's any particular bias in the missing data so let's naively remove it for now.
Full_masses_data = m_masses.dropna()
Full_masses_data.describe()           


#WE'VE DONE THE PANDAS PART! THE DATA IS READY FOR NUMPY ARRAYS


# --------------- 2. Swapping panda DF for numpy arrays ---------------
#Features into its own table, then into just an array of values
Feature_Data = Full_masses_data[['age','shape','margin','density']].values
Feature_Data

# Severity into its own table, then into just an array of values
Classifier_Data = Full_masses_data[['severity']].values
Classifier_Data 

# Let's save the names of the features in case we want to call them for showing data
names = ['age','shape','margin','density']


# --------------- 3.  Normalisation ---------------
scaler = preprocessing.StandardScaler()
Scaled_Features = scaler.fit_transform(Feature_Data)
Scaled_Features


# NOW WE'RE READY TO TRAIN AND TEST THE DATA!!!

# --------------- 4.  Splitting train and test data ---------------
from sklearn.model_selection import train_test_split

# Use the random seed funtion so that the same test and train sets are randomly chosen. Unsure if both this and random_state 
# in the train test split are needed.
np.random.seed(0)

(train_features,test_features,train_class,
  test_class) = train_test_split(Scaled_Features, Classifier_Data, train_size=0.8, random_state =1)

# --------------- 5. Ok now let's train a neural network, play around with optimizers, nodes, layers, activation etc. ---------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam

#1) - Build the model layers
model = Sequential()
model.add(Dense(100, input_dim = 4,kernel_initializer='normal', activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(50,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(20,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation = 'sigmoid'))


# --------------- 6. Compile the model ---------------
model.compile(loss='binary_crossentropy', optimizer='RMSprop',metrics=['accuracy'])


# --------------- 7. Try out the model ---------------
model.fit(train_features,train_class,batch_size = 100, epochs=100,verbose=2,validation_data = (test_features,test_class))


# --------------- 8. Add cross-validation to the train-test method ---------------
def Mammogram_classifier():
    
    """ Neural network complier as a function for cross-validation """
    
    model = Sequential()
    
    model.add(Dense(100, input_dim = 4,kernel_initializer='normal', activation = 'relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(50,activation = 'relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(20,activation = 'relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1,activation = 'sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='RMSprop',metrics=['accuracy'])
    return model


#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

Estimator = KerasClassifier(model=Mammogram_classifier,epochs =100,verbose=2)
cv_scores = cross_val_score(Estimator,Scaled_Features,Classifier_Data, cv = 10)
cv_scores.mean()

