#!/usr/bin/python
#Numpy
import numpy as np
from numpy import sys
np.set_printoptions(threshold=sys.maxsize) #otherwise the print output is suppressed

#Scipy
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import stats

#Dimensionality Reduction and Machine Learning
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
import sys
import os

#Matplotlib
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt   
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

#CSV (for writing data to file)
import csv

### MODIFY THESE TO SPECIFY THE TRAINING EXPERIMENT TO RUN ###
train_file = 'lightfuels_train3.csv'
test_file = 'lightfuels_test3.csv'
accuracy_threshold = -1 #print results for models with accuracies above this level

# Read in the training data:
train_data = pd.read_csv(train_file)
# Define y (actual class) and corresponding x (spectral data)
y_train = train_data.loc[:,'best1'].values
x_train = train_data.drop(['ID', 'Name', 'OldAlgorithmClassification', 'best1'],axis=1)


# Read in the test data:
test_data = pd.read_csv(test_file)
y_test = test_data.loc[:,'best1'].values
x_test = test_data.drop(['ID', 'Name', 'OldAlgorithmClassification', 'best1'],axis=1)
test_IDs = test_data.loc[:,'ID'].values

# Define PCA parameters
PCA = PCA(n_components=10, svd_solver='randomized', whiten=True)

# Fit training data
X_train = PCA.fit_transform(x_train)
#print('Explained variance:', PCA.explained_variance_ratio_)

# Run PCA on test data
X_test = PCA.transform(x_test)

#Iterate through Random Forest models
for n in (10, 100, 500, 1000, 2000): #number of trees in the forest (ensemble)
    for m in (20, 50, 100, 150): #number of samples used in each tree
        #Define classifier
        clf = RandomForestClassifier(n_estimators = n, max_samples = m, random_state = 0)
        # Run the classifier for ALL PCs to predict the Test data
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > accuracy_threshold:
            print("# Trees: ", n, "# Samples per Tree:", m, "Accuracy:", accuracy)

