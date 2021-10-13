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
from sklearn import svm
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score
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

#SVM parameters for Step 1: light fuel vs. lubricant vs. heavy fuel
ker1='rbf'
gam1=0.01
c1=1
#SVM parameters for Step 2: MGO vs. Bunker A
ker2='rbf'
gam2=0.01
c2=10

# Read in the training datasets:
train_1 = pd.read_csv('Types_train.csv')
train_2 = pd.read_csv('Types_lightfuels_train.csv')

# Define y (actual class) and corresponding x (spectral data)
y_train1 = train_1.loc[:,'best1'].values
x_train1 = train_1.drop(['ID', 'Name', 'best1'],axis=1)
y_train2 = train_2.loc[:,'best1'].values
x_train2 = train_2.drop(['ID', 'Name', 'best1'],axis=1)

# Read in the "test" (validation) data:
test_data = pd.read_csv('Types_validation.csv')
y_test = test_data.loc[:,'best1'].values
x_test = test_data.drop(['ID', 'Name', 'best1'],axis=1)
test_IDs = test_data.loc[:,'ID'].values

### PCA/SVM FOR STEP 1: light fuel vs. lubricant vs. heavy fuel ###
# Define PCA parameters
PCA1 = PCA(n_components=10, svd_solver='randomized', whiten=True)
# Fit training data
X_train1 = PCA1.fit_transform(x_train1)
# Run PCA on test data
X_test1 = PCA1.transform(x_test)
# Define SVM parameters

# Create classifier
clf1 = svm.SVC(kernel=ker1, gamma=gam1, C=c1)
# Generate predicted class (y=label 1,2,or3; X=PCs) 
y_pred = clf1.fit(X_train1, y_train1).predict(X_test1)
# Generate array for distances to classification boundaries
y_distance = clf1.decision_function(X_test1)

#y_pred_final_array = np.zeros(len(y_pred))

#Make a second level classification if sample is identified as light fuel in Step 1
for count, y_val in enumerate(y_pred):
    if y_val == 1:
        ### PCA/SVM FOR STEP 2: MGO vs. Bunker A ###
        # Define PCA parameters
        PCA2 = PCA(n_components=10, svd_solver='randomized', whiten=True)
        # Fit training data
        X_train2 = PCA2.fit_transform(x_train2)
        # Run PCA on test data
        X_test2 = PCA2.transform(np.reshape(x_test.iloc[count, :].values, (1, -1)))
        
        # Create classifier
        clf2 = svm.SVC(kernel=ker2, gamma=gam2, C=c2)
        # Generate predicted class (y=label 1 or 4; X=PCs) 
        y_pred_final = clf2.fit(X_train2, y_train2).predict(X_test2)[0]
        # Generate distances to classification boundary
        y_distance_2 = clf2.decision_function(X_test2)

    else:
        y_pred_final = y_val
        y_distance_2 = 'n/a'
    #y_pred_final_array[count] = y_pred_final     

#Print a list of the ID'd samples and their classification results
    print("ID:", test_IDs[count], "Actual:", y_test[count], "Prediction:", y_pred_final,
          "Distance to Boundaries 1:", y_distance[count], "Distance to Boundaries 2:", y_distance_2)

# Compute overall accuracy
#accuracy = accuracy_score(y_test, y_pred_final_array)
#print("Overall accuracy:", accuracy)


