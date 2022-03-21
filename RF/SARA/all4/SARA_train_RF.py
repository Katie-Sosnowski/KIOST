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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#Matplotlib
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt   
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

#Data reading/writing
import csv
import pandas as pd
import sys
import os

###MODIFY THESE TO SPECIFY TRAINING EXPERIMENT TO RUN###
train_file = 'SARA_train_rep1.csv'
labels = 'levelResin'
accuracy_threshold = -1

# Read in the data:
data = pd.read_csv(train_file)

# Define y (actual class) and corresponding x (spectral data)
y = data.loc[:,labels].values
#UNCOMMENT FOR "SARA" FILES
x = data.drop(['ID', 'Name', '%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin', 'levelResin', '%Asphaltene', 'levelAsphaltene'], axis=1)
#UNCOMMENT FOR "SA" FILES
#x = data.drop(['ID', 'Name', '%Saturate', 'levelSat_4classQuartiles', 'levelSat_2classMedian', 'levelSat_3classRange',
               #'%Asphaltene', 'levelAsp_4classQuartiles', 'levelAsp_3classQuartiles', 'levelAsp_3classRange1530', 'levelAsp_3classRange1020'],axis=1)


# Run random splitting, PCA, and all SVM parameter combinations 10 times
for i in range(10):
    # Split into test vs train data (70% train, 30% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, stratify = y, random_state=None)
    # Define PCA parameters
    pca = PCA(n_components=8, svd_solver='randomized', whiten=True)
    # Fit training data
    X_train = pca.fit_transform(x_train)
    # Run PCA on test data
    X_test = pca.transform(x_test)
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
