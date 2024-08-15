# Model Comparison for Telepresent Nature

## Introduction
Welcome to our repository!

This repository analyzes various machine learning models to gauge their ability to predict the position of moose and deer (quatitfied as longitude, latitude, and altitude) with the smallest mean absolute error and highest accuracy. 

The models with the best predictive ability will be used in Telepresent Nature, a tangible user interface that allows users to pinpoint the position of animals on a map using physical robots.

The models all fall into one of three categories:
1) Predict future position with past positon:
    - Autoregressive: Tensorflow neural network defined by the custom `FeedBack` class; model's lstm cell changes with every input, prepping the model for the next input
    - Single shot RNN: Tensorflow neural network defined by `Sequential` class; model's lstm components do not change with inputs
2) Predict current position given current enviornmental conditions (i.e., month, day, temperature)
    - Regressive: Tensorflow neural network defined by `Sequential` class with 4 Dense layers; final layer has a linear activation function
    - RNN Regressive: Tensorflow neural network defined by `Sequential` class with a LSTM layer
    - K-Means Regressor: Sklearn model defined by `KNeighborsRegressor` class
3) Predict current region given current enviornmental conditions (i.e., month, day, temperature)
    - Classification: Tensorflow neural network defined by `Sequential` class with 4 Dense layers; final layer has a softmax activation function
    - Random classification: Randomly guesses the region the animal belongs in

Through our analysis of the models and consideration of project needs/resources, we decided to continue on with the regressive and k-means regressor models in Telepresent Nature.

## Contents
- CSVFiles
    - CleanCSV: Contains csv cleaned files for each deer/moose in the RawCSV files
        - `[animal id]_interpolated.csv` files use interpolation to define datapoints at specific time intervals
        - `[animal id].csv` files use interpolation only when filling gaps
    - ExtendedCSV: Contains csv files for each model's predictions of positions beyond the original datasets
    - RawCSV: Contains original csv files
    - TestPerformanceCSV: Contains csv files comparing the models' testing predictions with the original testing, which are used in Tableau graphs

- ModelFiles
    - SavedModels: Contains .keras files for the Tensorflow neural networks
    - Statistics: Contains csv files with the mean absolute error and accuracy values for the models

- TableauAnalyses: Contains Tableau notebooks visualizing the csvs from CSVFiles/TestPeformanceCSV

- auxiliaries.py: Contains classes, functions, and constants associated with models' design

- cleaning_functions.py: Contains 4 classes for the creation of the csv files in CSVFiles/CleanCSV

- data_analysis.ipynb: Contains analysis of the data from the moose and deer datasets

- model_analyis.ipynb: Contains code training and testing the models