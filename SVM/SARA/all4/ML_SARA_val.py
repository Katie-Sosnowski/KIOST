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

# Read in the training data:
train_data = pd.read_csv('SARA_train.csv')
# Define y (actual class) and corresponding x (spectral data)
y_train = train_data.loc[:,'levelAsphaltene'].values
x_train = train_data.drop(['ID', 'Name','%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin', 'levelResin', '%Asphaltene', 'levelAsphaltene'],axis=1)


# Read in the test data:
test_data = pd.read_csv('SARA_validation.csv')
y_test = test_data.loc[:,'levelAsphaltene'].values
x_test = test_data.drop(['ID', 'Name','%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin', 'levelResin', '%Asphaltene', 'levelAsphaltene', 'type'],axis=1)
test_IDs = test_data.loc[:,'ID'].values

# Define PCA parameters
PCA = PCA(n_components=10, svd_solver='randomized', whiten=True)
# Fit training data
X_train = PCA.fit_transform(x_train)
# Run PCA on test data
X_test = PCA.transform(x_test)

# Define SVM parameters
ker='rbf'
gam=0.1
c=1
# Create classifier
clf = svm.SVC(kernel=ker, gamma=gam, C=c)
# Generate predicted class (y=label 0,1,2,or3; X=PCs) 
y_pred = clf.fit(X_train, y_train).predict(X_test)
for count, sample in enumerate(y_test):
    if y_test[count] != y_pred[count]:
        print("ID:", test_IDs[count], "Actual class:", y_test[count], "Prediction:", y_pred[count])

# Compute confusion matrix
#plt.figure()
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)
plot_confusion_matrix(clf, X_test, y_test, values_format='d', display_labels=['Very Low \n (0%)', 'Low \n (0.01-0.79%)', 'Medium \n (0.8-9.88%)', 'High \n (9.89% or more)'])
#plot_confusion_matrix(clf, X_test, y_test, normalize='true', display_labels=['Crude', 'Light Fuel', 'Lubricant', 'Heavy Fuel'])
plt.show()

