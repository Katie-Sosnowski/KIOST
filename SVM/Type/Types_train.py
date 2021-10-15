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
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn import svm
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
train_file = 'lightfuels_train2.csv'
test_file = 'lightfuels_test2.csv'
accuracy_threshold = 0.85 #print results for models with accuracies above this level

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
print('Explained variance:', PCA.explained_variance_ratio_)

# Run PCA on test data
X_test = PCA.transform(x_test)

#Iterate through SVM models
highest_accuracy = [0,0,0,0]
for gam in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
    for c in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        for ker in ('poly', 'linear', 'rbf'):
            #Define SVM parameters
            clf = svm.SVC(kernel=ker, gamma=gam, C=c)
            # Run the classifier for ALL PCs to predict the Test data
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > accuracy_threshold:
                print(ker, gam, c, accuracy)

