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
from sklearn import svm
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
train_file = 'SA_train_rep1.csv'
labels = 'levelSat_3classRange'

# Read in the data:
data = pd.read_csv(train_file)

# Define y (actual class) and corresponding x (spectral data)
y = data.loc[:,labels].values
x = data.drop(['ID', 'Name', '%Saturate', 'levelSat_4classQuartiles', 'levelSat_2classMedian', 'levelSat_3classRange',
               '%Asphaltene', 'levelAsp_4classQuartiles', 'levelAsp_3classQuartiles', 'levelAsp_3classRange1530', 'levelAsp_3classRange1020'],axis=1)

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
    # Iterate through SVM models
    highest_accuracy = 0
    highest_accuracy_list = [['ex','ex','ex',0,[],[]]]
    for gam in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        for c in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
            for ker in ('poly', 'linear', 'rbf'):
                #Define SVM parameters
                clf = svm.SVC(kernel=ker, gamma=gam, C=c)
                # Run the classifier for ALL PCs to predict the test data
                y_pred = clf.fit(X_train, y_train).predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                # Keep a record of the accuracies - this will eliminate
                # some but not all lower accuracies from ever being appended
                if accuracy >= highest_accuracy:
                    highest_accuracy_list.append([ker,gam,c,accuracy, y_test, y_pred])
                    # Reset highest accuracy each time a higher accuracy is found
                    highest_accuracy = accuracy
    # Only print the results that give the highest accuracy (could be tied between multiple)
    print('Run %d: \n' %i,
          #[confusion_matrix(entry[4], entry[5]) for entry in highest_accuracy_list if entry[3]==highest_accuracy],
          [entry[0:4] for entry in highest_accuracy_list if entry[3]==highest_accuracy])


