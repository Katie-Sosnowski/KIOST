# AUTHORS: Alanna Zubler and Katie Sosnowski
# DATE CREATED: 01/13/2021
# LAST MODIFIED: 04/22/2021

# DESCRIPTION: This program is used to determine the ideal regression model/PCA/scaling combinations that will be
# used for independent validation in sara-regression-independent-validation.py. In this program data (located in the
# files named SARA_train_repX.csv) is randomly split into test and train data, with 30% of the data being used for
# testing.
# 100 random splits are performed and run through a combination of the desired scaling, PCA, and regression parameters.
# The program must be rerun for every combination and duplicate. Preprocessing/Scaling and regression models can be
# selected easily from the defined functions and used in the main program that appends the list of functions.
# For each run, the R-squared, MAE, and MSE metrics were appended to respective lists. At the conclusion of running
# through each of the 100 splits, an average of each of the three metrics was taken to provide general guidance on
# the performance of the preprocessing and regression combination.

# GridSearchCV was used to determine the ideal parameters for each regression method, the results of which are
# incorporated into the machine learning regression functions. A separate function for GridSearchCV can be used to
# repeat this process but the main program will need to be modified accordingly (only one split, not 100. No need
# for printing evaluation metrics as GridSearchCV already provides a score for each parameter, etc.).

# Resources:
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

# --------------------------------------------------------------------------------------------------------------------
# Import required libraries
import sys
import numpy as np
import pandas as pd

# sklearn test/train data splitting
from sklearn.model_selection import train_test_split

# sklearn preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# GridSearchCV - used to identify ideal parameters for machine learning algorithms
from sklearn.model_selection import GridSearchCV

# Machine learning regression models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Evaluation metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statistics import mean

np.set_printoptions(threshold=sys.maxsize)

# -------------------------------------- PREPROCESSING FUNCTIONS -------------------------------------------------------

# Principal Component Analysis (PCA)
def run_PCA(components, PCA_x_train, PCA_x_test):
    # Define PCA parameters
    PCA_iter = PCA(n_components=components, svd_solver='randomized', whiten=True)

    # Fit training data
    X_train = PCA_iter.fit_transform(PCA_x_train)

    # Run PCA on test data
    X_test = PCA_iter.transform(PCA_x_test)

    # Return a tuple with the train and test values for X resulting from PCA
    return (X_train, X_test)


# Standard Scaler
def run_StandardScaler(y_train, y_test):
    scaler = StandardScaler()

    # Fit the scaler on training set
    scaler.fit(y_train)

    # Transform test and train data
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Return tuple with scaled data
    return (y_train_scaled, y_test_scaled)


# MinMax Scaler
def run_MinMaxScaler(y_train, y_test):
    scaler = MinMaxScaler()

    # Fit the scaler on training set
    scaler.fit(y_train)

    # Transform test and train data
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Return tuple with scaled data
    return (y_train_scaled, y_test_scaled)


# Robust Scaler
def run_RobustScaler(y_train, y_test):
    scaler = RobustScaler()

    # Fit the scaler on training set
    scaler.fit(y_train)

    # Transform test and train data
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    # Return tuple with scaled data
    return (y_train_scaled, y_test_scaled)

# -------------------------------------- MACHINE LEARNING FUNCTIONS ----------------------------------------------------

# Random Forest Regression
def run_RandomForestRegression(RF_X_train, RF_y_train, RF_X_test):
    # Define regression method and parameters
    regressor = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_split=3, max_leaf_nodes=100)

    # Fit the model to the training data
    regressor.fit(RF_X_train, RF_y_train)

    # Predict values
    y_pred = regressor.predict(RF_X_test)

    # Return the predicted model values
    return y_pred

# K Nearest Neighbors Regression
def run_KNNRegression(KNN_X_train, KNN_y_train, KNN_X_test):
    # Define regression method and parameters
    regressor = KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm="brute")

    # Fit the model to the training data
    regressor.fit(KNN_X_train, KNN_y_train)

    # Predict values
    y_pred = regressor.predict(KNN_X_test)

    # Return the predicted model values
    return y_pred


# Support Vector Regression
def run_SVRRegression(SVR_X_train, SVR_y_train, SVR_X_test):
    # Define regression method and parameters
    regressor = SVR(kernel='rbf')

    # Fit the model to the training data
    regressor.fit(SVR_X_train, SVR_y_train)

    # Predict values
    y_pred = regressor.predict(SVR_X_test)

    # Return the predicted model values
    return y_pred

# ------------------------------------ PARAMETER TESTING (GRIDCV) FUNCTION ---------------------------------------------
# Testing Parameters for the Machine Learning Models with GridSearchCV
def test_parameters(X_train, y_train):
    # Example: RF
    # Define the parameters you wish to test
    parameters = {'kernel': ['rbf', 'sigmoid'], 'gamma': ["scale", "auto"]}

    # Apply GridSearchCV to the machine learning algorithm and paramters; fit to training data
    clf = GridSearchCV(SVR(), parameters)
    clf.fit(X_train, y_train)

    # Print the results of the best parameters and best score
    print("Random Forest")
    print(clf.best_params_)
    print(clf.best_score_)
    # Brute force algorithm, distance weight, n_neighbors between 3 and 6 seemed to work best for KNN.
    # For random forest, max leaf nodes 75-100, min_samples_split 2-5, n_estimators 100 seemed best.
    # SVR did not work well with GridSearchCV, had some errors.
    return

# --------------------------------------------- MAIN PROGRAM -----------------------------------------------------------

# Import Training Data
train_data = pd.read_csv("SARA_train_rep1.csv")

# Column titles: Saturate, Aromatic, Resin, Asphaltene
# Define y (actual class) and corresponding x (spectral data)
y = train_data.loc[:, '%Saturate'].values
x = train_data.drop(['ID', 'Name', '%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin', 'levelResin',
                     '%Asphaltene', 'levelAsphaltene',], axis=1)

# Create lists to store fit metrics to average over multiple runs
R2_list = []
MAE_list = []
MSE_list = []

# Run the process 100 times and then get the average R2, MSE, and MAE.
for i in range(0, 10):
    # Split into test vs train data (70% train, 30% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=None)

    # Run desired functions
    # Scale y
    # Data needs to be reshaped to accommodate Scaler() functions
    #y_train = y_train.reshape(-1, 1) # Comment out if not scaling
    #y_test = y_test.reshape(-1, 1) # Comment out if not scaling
    # Run scaler function
    #scaled_y = run_StandardScaler(y_train, y_test) # Comment out if not scaling
    # Turn data back into 1D arrays for use with the regression algorithm/PCA
    #y_train = scaled_y[0].ravel() # Comment out if not scaling
    #y_test = scaled_y[1].ravel() # Comment out if not scaling

    # Scale x
    #scaled_x = run_StandardScaler(x_train, x_test)
    #x_train = scaled_x[0]
    #x_test = scaled_x[1]

    # PCA
    x_PCA = run_PCA(8, x_train, x_test)
    x_train = x_PCA[0]
    x_test = x_PCA[1]

    test_parameters(x_train, y_train)

    # Regression Algorithm
    #y_pred = run_SVRRegression(x_train, y_train, x_test)

    # Get R-Squared
    #r_2 = float(format(r2_score(y_test, y_pred), '.3f'))
    #R2_list.append(r_2)

    # Get MAE (Mean Absolute Error)
    #mae = float(format(mean_absolute_error(y_test, y_pred), '.3f'))
    #MAE_list.append(mae)

    # Get MSE (Mean Squared Error)
    #mse = float(format(mean_squared_error(y_test, y_pred), '.3f'))
    #MSE_list.append(mse)

# Take an average of the statistics for all the runs
#r_2 = mean(R2_list)
#mae = mean(MAE_list)
#mse = mean(MSE_list)

# Print the statistics
#print("\nR-Squared Score: " + str(r_2))
#print("\nMean Absolute Error: " + str(mae))
#print("\nMean Squared Error: " + str(mse))
