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

# Read in the training data:
os.chdir('/Users/jtsosnowski/Desktop/UA/YoonLab/CrudeOil/MATT/AugustDeadline')
train_data = pd.read_csv('train_Andrew_2.csv')
# Define y (actual class) and corresponding x (spectral data)
y = train_data.loc[:,'best1'].values
x = train_data.drop(['reference','New Class','Location/type','type','best1'],axis=1)

#Define PCA parameters
PCA = PCA(n_components=10, svd_solver='randomized', whiten=True)
#Fit training data
X = PCA.fit_transform(x)

#Iterate through SVM models
for gam in (0.01, 0.1, 1, 10, 100):
    for c in (0.01, 0.1, 1, 10, 100):
        for ker in ('poly', 'linear', 'rbf'):
            correct=0
            incorrect=0
            accuracy=0

            #Iterate through rows (each row is a sample)
            for row in range(0,43): #NOTE: Change this depending on size of dataset
                #Define the test data:
                os.chdir('/Users/jtsosnowski/Desktop/UA/YoonLab/CrudeOil/MATT/AugustDeadline')
                testx = pd.read_csv('test_Andrew_2.csv', encoding='latin-1').iloc[row,5:1208]
                actual_class = pd.read_csv('test_Andrew_2.csv', encoding='latin-1').iloc[row,4]
                #Reformat the test data
                testxarray = np.asarray(testx)
                testxarray = testxarray.reshape(1, -1)
                
                #Run PCA on test data
                X_test = PCA.transform(testxarray)
                
                #Define SVM parameters
                clf = svm.SVC(kernel=ker, gamma=gam, C=c)
                #Generate predicted class (y=label 0,1,2,or3; X=PCs) 
                y_pred = clf.fit(X, y).predict(X_test)
                if actual_class==y_pred[0]:
                    correct+=1
                else:
                    incorrect+=1
            accuracy=correct/(correct+incorrect)
            print('kernel= ', ker, 'gamma= ', gam, 'C= ', c, 'accuracy= ', accuracy)


