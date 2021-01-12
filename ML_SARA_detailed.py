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
os.chdir('/Users/jtsosnowski/Desktop/UA/YoonLab/CrudeOil/KATIE/Nov2020Deadline/SARA')
train_data = pd.read_csv('OTA2_SARA_double_sliced.csv')
# Define y (actual class) and corresponding x (spectral data)
y = train_data.loc[:,'levelSaturate'].values
x = train_data.drop(['ID', 'Name', 'OldAlgorithmClassification', '%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin', 'levelResin', '%Asphaltene', 'levelAsphaltene'],axis=1)
#Split into test vs train data (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=None)
#Define PCA parameters
PCA = PCA(n_components=10, svd_solver='randomized', whiten=True)
#Fit training data
X_train = PCA.fit_transform(x_train)
#Run PCA on test data
X_test = PCA.transform(x_test)
#Create pc1 and pc2 vectors for entire dataset for plotting
pc1 = np.hstack((X_train[:,0], X_test[:,0]))
pc2 = np.hstack((X_train[:,1], X_test[:,1]))
#Define SVM parameters
ker='rbf'
gam=0.1
c=100
true_class_color='green'
true_class_list=[]
#Define SVM parameters
clf = svm.SVC(kernel=ker, gamma=gam, C=c)
#Generate predicted class (y=label 0,1,2,or3; X=PCs) 
y_pred = clf.fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
#cm = confusion_matrix(y_test, y_pred)#Confusion Matrix for the raw test data
np.set_printoptions(precision=3)
#Non-normalized confusion matrix
plot_confusion_matrix(clf, X_test, y_test, values_format='d',
                      display_labels=['Very Low \n (less than 35.6%)', 'Low \n (35.6-62.5%)', 'Medium \n (62.5-71.3%)', 'High \n (more than 71.3%)'])
plt.show()

###Plot stuff that I'm no longer using
##Iterate through rows (each row is a sample)
##for row in range(len(x)): 
##    #Color code data similar to Matt's SVM plot
##    if (y[row]==3): 
##        true_class_color='lightskyblue' #high
##    elif (y[row]==1):
##        true_class_color='firebrick' #med-high
##    elif (y[row]==2):
##        true_class_color='moccasin' #low-med
##    else: 
##        true_class_color='royalblue' #low
##    #Add the correct color code for the current sample to an array
##    true_class_list.append(true_class_color)
##    
##    
##                
###Plot PC1 vs PC3 for the entire test dataset
##plt.figure(0)
##plt.xlabel('PC-1')
##plt.ylabel('PC-2')
##plt.suptitle('PCA Analysis of Aromatic Contents') #Note: change title to match dataset
##for x,y,c in zip(pc1, pc2, true_class_list):
##    plt.plot(x, y, 'o', color=c)
###Comment out to zoom in on data, unncomment to fit to Matt's original dimensions
###plt.xlim(-1.5,2.5);
###plt.ylim(-3,4);
##plt.show()

