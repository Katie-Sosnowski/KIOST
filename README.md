# KIOST

Scripts for applying PCA + SVM, or PCA + regression models, to data from the Korea Institute of Ocean Science and Technology (KIOST) Oil Type Analysis (OTA) project.

Note: The file OTA_031221.py is the code for running the custom-built Oil Type Analyzer (OTA) device and will not work on its own without the device.

'ML_test_script_iterable' uses a grid-search to determine the best SVM parameters for the dataset.

'ML_test_script_detailed' can be used to input specific parameters and output more detailed information about that model.

In order to run these with the datasets that are being referenced, download the .csv files and change the filepath in the script to match the destination on your machine.
