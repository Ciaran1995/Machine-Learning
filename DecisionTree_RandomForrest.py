#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:05:45 2021

@author: ciaranmcdonnell
"""

""" Use a decision tree and random forrest approach to classifying mammogram images. """

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



# --------------- 5. Decision Tree ---------------

import matplotlib.pyplot as plt
from sklearn import tree

classifier = tree.DecisionTreeClassifier()
clf = classifier.fit(train_features,train_class)
#tree.plot_tree(clf)

clf.score(test_features,test_class)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(clf,test_features,test_class,cv=10)
print(cv_scores)

cv_scores.mean()

## Struggling to get this plotting to work.
# from IPython.display import Image 
# import pydotplus
# dot_data = tree.export_graphviz(clf,feature_names = names)
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())


# --------------- 6. Random Forrest ---------------

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 100, random_state=1)
RF_fit= RF.fit(Scaled_Features,np.ravel(Classifier_Data))

RF_cv_score = cross_val_score(RF_fit,Scaled_Features,np.ravel(Classifier_Data),cv = 100)
RF_cv_score.mean()
#fit(RF,Scaled_Features,Classifier_Data, CV=10)

