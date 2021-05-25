# AUTHORS: Alanna Zubler and Katie Sosnowski
# DATE CREATED: 04/06/2021
# LAST MODIFIED: 04/22/2021

# DESCRIPTION: This program was written for the validation of models identified using sara-regression-testing.py.
# The selected models undergo a validation test using independent samples using this program. All of the preprocessing
# regression libraries/functions from sara-regression-testing.py are still included here, for a simpler replication
# process and for the reader to potentially use this as a resource for their own project.
# The samples used for independent validation (test data) are in files named ValidationData_Combined.csv.
# The training data is located in files named OTA2_SARA_CombinedDuplicates.csv.
# The predicted and actual values can be unscaled and plotted for a visualization of the results.

# Resources:
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

# ----------------------------------------------------------------------------------------------------------------------
# Import required libraries
import sys
import numpy as np
import pandas as pd

# sklearn preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Machine learning regression models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Evaluation metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statistics

# Plotting
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

# ------------------------------------------ PREPROCESSING FUNCTIONS ---------------------------------------------------

# Principal Component Analysis (PCA)
def run_PCA(components, PCA_x_train, PCA_x_test):
    # Define PCA parameters
    PCA_iter = PCA(n_components=components, svd_solver='randomized', whiten=True)

    # Fit training data
    X_train = PCA_iter.fit_transform(PCA_x_train)

    # Determine how much the data variance is explained by the principal components
    print(PCA_iter.explained_variance_ratio_)

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

# --------------------------------------------- MAIN PROGRAM -----------------------------------------------------------

# Import Training Data
train_data = pd.read_csv("SARA_train_combined.csv")

# Import Independent Validation Data
test_data = pd.read_csv("validation.csv")

# Column titles: Saturate, Aromatic, Resin, Asphaltene
# Define y (actual class) and corresponding x (spectral data) for training data
y_train_original = train_data.loc[:, '%Saturate'].values
x_train = train_data.drop(['ID', 'Name', '%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin',
                           'levelResin', '%Asphaltene', 'levelAsphaltene',], axis=1)

# Define x and y for validation/test data
y_test_original = test_data.loc[:, '%Saturate'].values
x_test = test_data.drop(['ID', 'Name', '%Saturate', 'levelSaturate', '%Aromatic', 'levelAromatic', '%Resin',
                          'levelResin', '%Asphaltene', 'levelAsphaltene', 'type', ], axis=1)

# Run desired functions
# Scaler
# Data needs to be reshaped to accommodate Scaler() functions
y_train = y_train_original.reshape(-1, 1)
y_test = y_test_original.reshape(-1, 1)
# Run scaler function
scaled_y = run_RobustScaler(y_train, y_test)
# Turn data back into 1D arrays for use with the regression algorithm/PCA
y_train = scaled_y[0].ravel()
y_test = scaled_y[1].ravel()

# Use these if not scaling, otherwise comment out
#y_train = y_train_original
#y_test = y_test_original

print('PCA')
# PCA
x_PCA = run_PCA(8, x_train, x_test)
x_train = x_PCA[0]
x_test = x_PCA[1]

print('Regression')
# Regression Algorithm
y_pred = run_SVRRegression(x_train, y_train, x_test)

# Get R-Squared
r_2 = float(format(r2_score(y_test, y_pred), '.3f'))
print("\nR-Squared Score: " + str(r_2))

# Get MAE (Mean Absolute Error)
mae = float(format(mean_absolute_error(y_test, y_pred), '.3f'))
print("\nMean Absolute Error: " + str(mae))

# Get MSE (Mean Squared Error)
mse = float(format(mean_squared_error(y_test, y_pred), '.3f'))
print("\nMean Squared Error: " + str(mse))

# Inverse transform the data to get actual SARA values
# The following lines are for Standard Scalar
#mean_y_train = statistics.mean(y_train_original) # get mean
#std_y_train = statistics.stdev(y_train_original) # get standard deviation

#actual_values = []
#for i in y_test:
#    inverse_transformed = i * std_y_train + mean_y_train # equation for inverse transform
#    actual_values.append(inverse_transformed)

#predicted_values = []
#for j in y_pred:
#    inverse_transformed = j * std_y_train + mean_y_train # equation for inverse transform
#    predicted_values.append(inverse_transformed)


# For Robust Scalar, MinMax Scaler
#predicted_values = []
#actual_values = []

#scaler = RobustScaler()
#y_train = y_train_original.reshape(-1, 1)
#y_pred = y_pred.reshape(-1, 1)
#scaler.fit(y_train)
#y_pred_unscaled = scaler.inverse_transform(y_pred)

#for k in y_test_original:
#    actual_values.append(k)

#for m in y_pred_unscaled:
#    predicted_values.append(m)

# Plot Results
#plt.scatter(predicted_values, actual_values)
#plt.xlabel("Predicted Asphaltene Content (%)")
#plt.ylabel("Actual Asphaltene Content (%)")
#plt.title("Actual vs. Predicted Values (Asphaltene): Robust Scalar with SVR")
#plt.plot([0, 20, 40], [0, 20, 40], color='red') # add 80 to end of lists if plotting Saturate
#plt.annotate("R^2 = {:.3f}".format(r2_score(y_test, y_pred)), xy=(10, 30))
#plt.show()