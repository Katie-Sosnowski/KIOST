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

# Read in the data:
os.chdir('/Users/jtsosnowski/Desktop/UA/YoonLab/CrudeOil/KATIE/Nov2020Deadline/SARA')
data = pd.read_csv('OTA2_SARA_double_sliced.csv')
# Define y (actual class) and corresponding x (spectral data)
y = data.loc[:,'levelSaturate'].values
x = data.drop(['ID', 'Name', 'OldAlgorithmClassification', '%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin', 'levelResin', '%Asphaltene', 'levelAsphaltene'],axis=1)
#Split into test vs train data (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=None)
#Define PCA parameters
PCA = PCA(n_components=10, svd_solver='randomized', whiten=True)
#Fit training data
X_train = PCA.fit_transform(x_train)
#Run PCA on test data
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
            if accuracy > highest_accuracy[3]:
                highest_accuracy=[ker,gam,c,accuracy]
print(highest_accuracy)


