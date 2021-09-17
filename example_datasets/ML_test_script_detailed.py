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
train_data = pd.read_csv('train_Andrew_3.csv')
# Define y (actual class) and corresponding x (spectral data)
y = train_data.loc[:,'best1'].values
x = train_data.drop(['reference','New Class','Location/type','type','best1'],axis=1)

#Initialize arrays
pc1 = [None]*69
pc3 = [None]*69
true_class_list = list()
true_class_color = 'green' #will be reset

#Define PCA parameters
PCA = PCA(n_components=10, svd_solver='randomized', whiten=True)
#Fit training data
X = PCA.fit_transform(x)

#Define SVM parameters
ker='rbf'
gam=0.2
c=100
true_class_color='green'
true_class_list=[]

#Iterate through rows (each row is a sample)
for row in range(0,20): #Note: change this depending on data size
    #Define the test data:
    os.chdir('/Users/jtsosnowski/Desktop/UA/YoonLab/CrudeOil/MATT/AugustDeadline')
    testx = pd.read_csv('test_Andrew_3.csv', encoding='latin-1').iloc[row,5:1208]
    actual_class = pd.read_csv('test_Andrew_3.csv', encoding='latin-1').iloc[row,4]
    #Reformat the test data
    testxarray = np.asarray(testx)
    testxarray = testxarray.reshape(1, -1)
    
    #Color code data similar to Matt's SVM plot
    if (actual_class==3): 
        true_class_color='lightskyblue' #light fuel oil
    elif (actual_class==1):
        true_class='firebrick' #heavy fuel oil
    elif (actual_class==2):
        true_class_color='moccasin' #lubricant oil
    else: 
        true_class_color='royalblue' #crude oil
    
    #Run PCA on test data
    X_test = PCA.transform(testxarray)
    
    #Add the PC1 from the current sample to an array
    pc1[row] = X_test[0,0]
    #Add the PC3 value from the current sample to an array
    pc3[row] = X_test[0,2]
    #Add the correct color code for the current sample to an array
    true_class_list.append(true_class_color)
    
    #Define SVM parameters
    clf = svm.SVC(kernel=ker, gamma=gam, C=c)
    #Generate predicted class (y=label 0,1,2,or3; X=PCs) 
    y_pred = clf.fit(X, y).predict(X_test)
    print('actual class= ', actual_class, 'predicted class= ', y_pred[0])
    if actual_class==y_pred[0]:
        continue
    else:
        print("WRONG")
                
#Plot PC1 vs PC3 for the entire test dataset
plt.figure(0)
plt.xlabel('PC-1')
plt.ylabel('PC-3')
plt.suptitle('PCA Analysis of 8/4/20 Data') #Note: change title to match dataset
for x,y,c in zip(pc1, pc3, true_class_list):
    plt.plot(x, y, 'o', color=c)
#Comment out to zoom in on data, unncomment to fit to Matt's original dimensions
plt.xlim(-1.5,2.5);
plt.ylim(-3,4);
plt.show()

