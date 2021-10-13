# Machine learning techniques for chemical and type analysis of oil spill samples via field-ready handheld spectrophotometer device

Datasets and Python scripts associated with the paper "Machine learning techniques for chemical and type analysis of oil spill samples via field-ready handheld spectrophotometer device" by Sosnowski et al. (manuscript in progress).

# SVM 
The SVM (support vector machine) folder includes two folders, "SARA" and "Type", which include the datasets and Python scripts for SARA (saturate and asphaltene) and oil type analysis, respectively.

### SARA
The "SARA" folder includes separate train files for the separate replicates of data as described in the manuscript. These are subsequently split into test and train samples during training. It also includes a composite train data file and the independent validation dataset which are both called during validation.

The file "SA_train.py" is for training an SVM model (i.e. determining the best hyperparameters using training data) for determining relative saturate and asphaltene levels. It requires that the user enter a training file name that will be split into test and train data multiple times, as well as the column of labels to use (i.e. which of the 7 models to train). Options for labels include "levelSat_4classQuartiles", "levelSat_2classMedian", "levelSat_3classRange", "levelAsp_4classQuartiles", "levelAsp_3classQuartiles", "levelAsp_3classRange1530", and "levelAsp_3classRange1020". These are described in the manuscript.

The file "SA_val.py" is for validating any of the above 7 models on the independent validation set. It requires that the user enter the column of labels as above, the display labels (for generating a confusion matrix with the appropriate class labels) and the SVM parameters chosen from training (ker, gam, c).

The "excel_files" folder includes Excel files, which are not called by the Python scripts, but are included in order to visualize how the labels were created using the "=IFS" function in Excel (as .csv files hide this information). 

The "all4" folder includes older scripts and datasets for analyzing all 4 SARA contents: saturate, aromatic, resin, and asphaltene (based on quartile analysis). 

### Type
The "Type" folder includes separate train and test files as described in the manuscript. The .csv files beginning with "types" classify oil samples based on the first level of classification (light fuel vs. lubricant vs. heavy fuel). The .csv files beginning with "lightfuels" classify oil samples based on the second level of classification (MGO vs. Bunker A). 

The file "Types_train.py" is for training (requires that the user enter a train and test file).

The file "Types_val.py" performs both levels of classification on the independent validation dataset using the parameters determined during training.

# Running the spectrophotometer device
The file "OTA_031221.py" is the code for running the custom-built Oil Type Analyzer (OTA) device and will not work without the device.

# Regression 
The "regression" folder includes datasets and code for running regression experiments on SARA data included in the supplementary information.


