#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:17:22 2019

@author: riddaali
"""
import DataAnalysis_Functions
import Plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# =============================================================================
# Reading the BRFSS 2018 data: https://www.cdc.gov/brfss/annual_data/annual_2018.html
# PLEASE MAKE SURE THAT YOU HAVE UNZIPPED THE "LLCP2018XPT.zip" FILE BEFORE
# EXECUTING THE FOLLOWING "readFile()" FUNCTION.
# =============================================================================
reading_file = DataAnalysis_Functions.readFile('LLCP2018.XPT')

# =============================================================================
# Reading the csv file ('BRFSS2018.csv') that was created  by calling the
# "readFile()" function from the "DataAnalysis_Functions.py"
# =============================================================================
BRFSS_df = pd.read_csv('BRFSS2018.csv') 



# Selecting all the rows with NaNs (missing values)
BRFSS_df[pd.isnull(BRFSS_df).any(axis=1)]


# Calculating the total missing values for each column
print(BRFSS_df.isnull().sum())
# =============================================================================
# Overweight_Obese          0
# Education_Level           0
# Income_Categories         0
# Exercise                 25
# Age_Categories            0
# Gender                    0
# Alcohol_Consumption       0
# Arthritis              2534
# =============================================================================
# As shown above there are 2 features with missing values

# Calculating the total of all missing values
# Source used for guidance: https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/
NAs_sum = BRFSS_df.isnull().sum().sum() # 2559

# Data frame dimension before removing NAs and meaningless values 
# (e.g. Don't know, refused, etc.)
BRFSS_df.shape # [437436 rows x 8 columns]



# =============================================================================
# #-------------------------- Data Cleaning -----------------------------------
# =============================================================================
# Cleaning the data by using the "dataCleaning()" function from the 
# "DataAnalysis_Functions" python file
cleaned_df = DataAnalysis_Functions.dataCleaning(BRFSS_df)

# Ensuring that the sum of missing value is 0 after the data cleaning process
cleaned_df.isnull().sum()

# =============================================================================
# Saving the dataframe without missing data and meaningless values into a new 
# csv file called "BRFSS_cleaned.csv" by using the Pandas dataframe "to_csv()"
# function and then specifying the path to save the "cleaned" csv file.
# =============================================================================
cleaned_df.to_csv(r'./BRFSS_cleaned.csv', index = None, header=True)


# Checking if all the missing values have been removed
cleaned_df.isnull().sum().sum() # 0 

# Reading the cleaned dataframe from the 'BRFSS_cleaned.csv'
new_df = pd.read_csv('BRFSS_cleaned.csv')

# Converting the dataframe into categorical data
new_df = new_df.astype('category')

# Getting information about each feature
new_df.describe()


# =============================================================================
# #-------------------- Data Visualization: Plots ----------------------------
# =============================================================================

obesity_age_plot = Plots.plot_obesity_age(new_df)

obesity_education_plot = Plots.plot_obesity_education(new_df)

obesity_income_plot = Plots.plot_obesity_income(new_df)

obesity_exercise_plot = Plots.plot_obesity_exercise(new_df)

obesity_gender_plot = Plots.plot_obesity_gender(new_df)

obesity_alcohol_plot = Plots.plot_obesity_alcohol(new_df)

obesity_arthritis_plot = Plots.plot_obesity_arthritis(new_df)


# Using the features importance property of the Random Forest to create the plot
features_importance_plot = Plots.featureImportance()


# # Comparing the models performance by calling the "comparingModelsPerformance()"
# function included inside the "DataAnalysis_Functions.py"
models_comparison_plot = Plots.comparingModelsPerformance()

correlation_matrix_plot = Plots.correlation_matrix(new_df)


# =============================================================================
# #-------------------------- Global variables ----------------------------
# =============================================================================

# =============================================================================
# Converting the categorical labels into numerical labels by using the 
# "LabelEncoder()" function from the sklearn preprocessing library
# =============================================================================
lr_df = new_df.apply(preprocessing.LabelEncoder().fit_transform)

# =============================================================================
# Defining independent variables ("iv") by accessing the columns label through
# the "loc()" function from the Pandas dataframe library
# =============================================================================
iv = lr_df.loc[:, ['Education_Level', 'Income_Categories', 
                  'Exercise', 'Age_Categories', 'Gender', 'Alcohol_Consumption',
                  'Arthritis']]

# =============================================================================
# Defining the dependent variable ("dv") by accessing the column label through
# the "loc()" function from the Pandas dataframe library
# =============================================================================
dv = lr_df.loc[:, ['Overweight_Obese']]

# Splitting the data into training (80%) and test (20%) dataset 
iv_train, iv_test, dv_train, dv_test = model_selection.train_test_split(iv, dv, 
                                       test_size=0.2, random_state=0)


# Defining the method: "LogisticRegression()"  
log_reg = LogisticRegression()

# =============================================================================
# Splitting the dataset into 10 folds. Each fold will be used once for validation,
# while the remaining 9 (k-1 = 9) folds will be used for trainin purposes
# =============================================================================
k_fold = model_selection.KFold(n_splits=10, random_state=0)


# =============================================================================
# #-------------------------- Logistic Regression ----------------------------
# =============================================================================


def logistic_regression():
    """  
    Source used for guidance: 
        
    - https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

    ---------------------------------------------------------------------------
    Source used for guidance:
        
    Transforming the categorical data by using the LabelEncoder()
    
    - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    ---------------------------------------------------------------------------
    Source used for guidance:
        
    K-Folds cross-validation:
    
    - https://www.pluralsight.com/guides/validating-machine-learning-models-scikit-learn
    
    """    
    
    # Fitting the Logistic Regression Model on the training dataset
    log_reg.fit(iv_train, dv_train)
    
    # Getting Coefficients
    print("Coefficients of the Logistic Regression model: ",log_reg.coef_)
    # =========================================================================
    # [[-0.02446003,  0.00814376, -0.37673945,  0.01007861,  0.56478281,
    # -0.14414516, -0.47579558]]
    # =========================================================================
    
    # Caluculating the odds ratios by taking the exponential of the coefficients
    print("Odds ratios the Logistic Regression model: ",np.exp(log_reg.coef_))
    # =========================================================================
    #     [[0.9758367  1.00817701 0.68609481 1.01012957 1.75906568 0.86576207
    #   0.6213905 ]]
    #     
    # =========================================================================
    # Making predictions for the test dataset
    log_reg.predict(iv_test)
    
    # Finding out how well the model performed (accuracy measurement)
    # Accuracy rate of this logistic regression classifier on the test dataset: 0.6886
    print('Accuracy rate of this logistic regression classifier on the test dataset: {:.4f}'.format(log_reg.score(iv_test, dv_test)))
        
    # =============================================================================
    #     Calculating the mean accuracy for the Logistic regression model using 
    #     10-Fold cross-validation: 68.81%
    # =============================================================================
    outcome_cv_lr = model_selection.cross_val_score(log_reg, iv, dv, cv=k_fold)
    print("10-Folds Cross-Validation scores: %.2f%%" % (outcome_cv_lr.mean()*100.0)) 
    

logistic_regression()



# =============================================================================
# # ------------------------- Feature Selection ------------------------------
# =============================================================================
# Code adapted from the following source:
# https://towardsdatascience.com/a-look-into-feature-importance-in-logistic-regression-models-a4aa970f9b0f
# Setting the parameter "n_features_to_select" equal to 1 to get full ranking of features
independent_vars = iv_train
select_vars = RFE(log_reg, n_features_to_select = 1)
select_vars = select_vars.fit(independent_vars, dv_train)
ranks = select_vars.ranking_
print("Ranks: ", ranks)

# =============================================================================
# Creating an empty list to store the features ranking, string formatting it to
# get the rank followed by its corresponding feature
# =============================================================================
features_ranking = []
for var in ranks:
    features_ranking.append(f"{var}: {new_df.columns[var]}")
features_ranking

# Sorting the "features_ranking" in ascending order by using the "sort()" function
features_ranking.sort()
print("Sorted features ranking: ", features_ranking)
# =============================================================================
# ['1: Education_Level',
#  '2: Income_Categories',
#  '3: Exercise',
#  '4: Age_Categories',
#  '5: Gender',
#  '6: Alcohol_Consumption',
#  '7: Arthritis']
# =============================================================================


# =============================================================================
# # -------------------------- Random Forest----------------------------------
# =============================================================================

def random_forest():
    """ 
    Source used for guidance:
    - https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
    
    Plotting the confusion matrix - To see the overall misclassified data
    Code adapted from the following source: 
    - http://www.tarekatwan.com/index.php/2017/12/how-to-plot-a-confusion-matrix-in-python/

    """
    # =========================================================================
    # Creating Random Forest model with 100 decision trees, using bootstrap 
    # (random sampling with replacement) samples, the "max_features" is the 
    # square root of the number of features
    # =========================================================================
    rf = RandomForestClassifier(n_estimators = 100, bootstrap = True, 
                                max_features = 'sqrt')
    # Train the model on training data
    rf.fit(iv_train, dv_train)
    
    # Using the predict function from the Random Forest library to make predictions
    # on the test dataset
    predictions = rf.predict(iv_test)
   
    # Generating probabilities for each class
    rf_probs = rf.predict_proba(iv_test)[:, 1]
    
    # =============================================================================
    # Calculating AUC (Area Under Curve) - ROC (Receiver operating characteristic),
    # where ROC represents the probability curve and AUC measures how well the 
    # model differentiates among classes.
    # =============================================================================
    roc_auc_score(dv_test, rf_probs) # 0.6488433972934544
                              
    print(confusion_matrix(dv_test,predictions))
    # =========================================================================
    #     [[ 3919 16626]
    #     [ 3687 41747]]
    # =========================================================================
    
    # =============================================================================
    # Plotting the confusion matrix
    # Code adapted from the following source: 
    # - http://www.tarekatwan.com/index.php/2017/12/how-to-plot-a-confusion-matrix-in-python/
    # =============================================================================
    confusion_mat = confusion_matrix(dv_test,predictions)
    
    plt.clf() # Clearing the current figure
    plt.imshow(confusion_mat, cmap=ListedColormap(('lightgreen', 'red')))
    class_labels = ['No','Yes']
    plt.ylabel('True')
    plt.xlabel('Predicted')
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    neg_pos = [['TN','FP'], ['FN', 'TP']]
    # =============================================================================
    # As there are 2 classes and the "neg_pos" labels are inside a neseted list:
    # looping through a sequence of numbers from 0 upto 2, but excluding it.
    # Casting the labels to be "strings" by using the "str()" function.
    # =============================================================================
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(neg_pos[i][j])+" : "+str(confusion_mat[i][j]))
    # =============================================================================
    # Saving the plot in the specified path by using the function "savefig" 
    # from the "matplotlib.pyplot" library
    # =============================================================================
    plt.savefig('./Plots/Confusion_matrix_plot') 
    plt.show()
    
    
    
    # Creating a text report to illustrate the principal classification matrics
    print(classification_report(dv_test,predictions))
    
    # =============================================================================
    #                   precision    recall  f1-score   support
    # 
    #            0       0.52        0.19      0.28     20545
    #            1       0.72        0.92      0.80     45434
    # 
    #     accuracy                             0.69     65979
    #    macro avg       0.62        0.55      0.54     65979
    # weighted avg       0.65        0.69      0.64     65979
    # 
    # =============================================================================
    print(accuracy_score(dv_test,predictions)) #0.6921293138725958
    
    # =============================================================================
    #  Calculating the mean accuracy for the Random Forest model using 
    #  10-Fold cross-validation: 69.28%
    # =============================================================================
    outcome_cv_rf = model_selection.cross_val_score(rf, iv, dv, cv=k_fold)
    print("10-Folds Cross-Validation scores: %.2f%%" % (outcome_cv_rf.mean()*100.0)) 
    

random_forest()





# =============================================================================
# ------------------------- CODE THAT DOES NOT QUITE WORK ---------------------
# =============================================================================

# =============================================================================
# models = [LogisticRegression, RandomForestClassifier]
# 
# lr_df = new_df.apply(preprocessing.LabelEncoder().fit_transform)
# 
# # Defining independent variables ("iv")
# iv = lr_df.loc[:, ['Education_Level', 'Income_Categories', 
#                   'Exercise', 'Age_Categories', 'Gender', 'Alcohol_Consumption',
#                   'Arthritis']]
# 
# # Defining the dependent variable ("dv")
# dv = lr_df.loc[:, ['Overweight_Obese']]
# 
# # Splitting the data into training (80%) and test (20%) dataset 
# iv_train, iv_test, dv_train, dv_test = train_test_split(iv, dv, test_size=0.2, random_state=0)
# 
# def train_model(dataset, model_type):
#     model = model_type().fit(iv_train, dv_train)
#     return model
# 
#     for m in models:
#         model =train_model(new_df,m)
# 
# training = train_model(new_df, LogisticRegression)    
# predicting = training.predict(iv_test)
# 
# # Finding out how well the model performed (accuracy measurement)
# # Accuracy rate of this logistic regression classifier on the test dataset: 0.6886
# print('Accuracy rate of this logistic regression classifier on the test dataset: {:.4f}'.format(training.score(iv_test, predicting)))
# 
# 
# =============================================================================

#  ------------------------ EXAMPLE:-------------------------------------------
# =============================================================================
# models = [LogisticRegression, RandomForest]
# def train_model(dataset, model_type):
#     model = model_type().fit(dataset['X'])
#     return model
# 
# for m in models:
#     model = train_model(new_df, m)
# =============================================================================
 
# =============================================================================
# # Transforming the categorical data by using the LabelEncoder() 
# # Source used for guidance:
# # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# lr_df = new_df.apply(preprocessing.LabelEncoder().fit_transform)
# 
# # Defining independent variables ("iv")
# iv = lr_df.loc[:, ['Education_Level', 'Income_Categories', 
#                   'Exercise', 'Age_Categories', 'Gender', 'Alcohol_Consumption',
#                   'Arthritis']]
# 
# # Defining the dependent variable ("dv")
# dv = lr_df.loc[:, ['Overweight_Obese']]
# 
# # Splitting the data into training (80%) and test (20%) dataset 
# iv_train, iv_test, dv_train, dv_test = train_test_split(iv, dv, test_size=0.2, random_state=0)
# 
# # Defining the method: "LogisticRegression()"
# log_reg = LogisticRegression()
# 
# # Getting Coefficients
# log_reg.coef_
# # =============================================================================
# # array([[-0.02446003,  0.00814376, -0.37673945,  0.01007861,  0.56478281,
# #        -0.14414516, -0.47579558]])
# # =============================================================================
# 
# # Fitting the Logistic Regression Model on the training dataset
# log_reg.fit(iv_train, dv_train)
# 
# 
# # Making predictions for the test dataset
# cv_results = cross_validation.cross_val_predict(log_reg, iv_train, dv_train,cv=10)
# 
# predicted_class = log_reg(iv_test)
# # Finding out how well the model performed (accuracy measurement)
# log_reg_performance = metrics.accruacy_score(dv_test, cv_results)
# print(log_reg_performance) #0.688613043544158
# =============================================================================
 


# =============================================================================
# ---------------------------- OLD CODE ---------------------------------------
# =============================================================================

# =============================================================================
# =============================================================================
# #-------------------------- Logistic Regression ----------------------------
# =============================================================================

# # Source used for guidance: 
# # https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# 
# # Transforming the categorical data by using the LabelEncoder() 
# # Source used for guidance:
# # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# 
# lr_df = new_df.apply(preprocessing.LabelEncoder().fit_transform)
# 
# # Defining independent variables ("iv")
# iv = lr_df.loc[:, ['Education_Level', 'Income_Categories', 
#                   'Exercise', 'Age_Categories', 'Gender', 'Alcohol_Consumption',
#                   'Arthritis']]
# 
# # Defining the dependent variable ("dv")
# dv = lr_df.loc[:, ['Overweight_Obese']]
# 
# # Splitting the data into training (80%) and test (20%) dataset 
# iv_train, iv_test, dv_train, dv_test = train_test_split(iv, dv, test_size=0.2, random_state=0)
# 
# # Defining the method: "LogisticRegression()"  
# log_reg = LogisticRegression()
# 
# 
# # Fitting the Logistic Regression Model on the training dataset
# log_reg.fit(iv_train, dv_train)
# 
# # Getting Coefficients
# log_reg.coef_
# # =============================================================================
# # array([[-0.02446003,  0.00814376, -0.37673945,  0.01007861,  0.56478281,
# #        -0.14414516, -0.47579558]])
# # =============================================================================
# 
# 
# # Making predictions for the test dataset
# dv_pred = log_reg.predict(iv_test)
# 
# # Finding out how well the model performed (accuracy measurement)
# # Accuracy rate of this logistic regression classifier on the test dataset: 0.6886
# print('Accuracy rate of this logistic regression classifier on the test dataset: {:.4f}'.format(log_reg.score(iv_test, dv_test)))
# 
# =============================================================================


# =============================================================================
# =============================================================================
# #-------------------------- Random Forest----------------------------------
# =============================================================================
# # Source used for guidance:
# # https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
# 
# # Creating Random Forest model with 100 decision trees
# rf = RandomForestClassifier(n_estimators = 100, bootstrap = True, max_features = 'sqrt')
# # Train the model on training data
# rf.fit(iv_train, dv_train)
# 
# # Using the predict function from the Random Forest library to make predictions
# # on the test dataset
# predictions = rf.predict(iv_test)
# 
# # Generating probabilities for each class
# rf_probs = rf.predict_proba(iv_test)[:, 1]
# 
# # Calculating  roc auc
# roc_value = roc_auc_score(dv_test, rf_probs) #0.6487363600588887
#                           
# print(confusion_matrix(dv_test,predictions))
# # =============================================================================
# # [[ 3888 16657]
# #  [ 3656 41778]]
# # =============================================================================
# 
# print(classification_report(dv_test,predictions))
# # =============================================================================
# #               precision    recall  f1-score   support
# # 
# #            0       0.52      0.19      0.28     20545
# #            1       0.71      0.92      0.80     45434
# # 
# #     accuracy                           0.69     65979
# #    macro avg       0.62      0.55      0.54     65979
# # weighted avg       0.65      0.69      0.64     65979
# # 
# # =============================================================================
# 
# print(accuracy_score(dv_test,predictions)) #0.6921293138725958
# 
# =============================================================================







# =============================================================================
# models = [log_reg, rf]
#  
# lr_df = new_df.apply(preprocessing.LabelEncoder().fit_transform)
#  
# # Defining independent variables ("iv")
# iv = lr_df.loc[:, ['Education_Level', 'Income_Categories', 
#                    'Exercise', 'Age_Categories', 'Gender', 'Alcohol_Consumption',
#                    'Arthritis']]
#  
# # Defining the dependent variable ("dv")
# dv = lr_df.loc[:, ['Overweight_Obese']]
#  
# # Splitting the data into training (80%) and test (20%) dataset 
# iv_train, iv_test, dv_train, dv_test = train_test_split(iv, dv, test_size=0.2, random_state=0)
#  
# def train_models(dataset, model_type):
#     model = model_type().fit(iv_train, dv_train)
#     return model
# 
#     for m in models:
#         model =train_models(new_df, m)
# 
# # =============================================================================
# # #-------------------------- Logistic Regression ----------------------------
# # =============================================================================
# training_logReg = train_models(new_df, models[0])   
# log_reg.coef_ 
# predicting_logReg = training_logReg.predict(iv_test)
#  
# # Finding out how well the model performed (accuracy measurement)
# # Accuracy rate of this logistic regression classifier on the test dataset: 0.6886
# print('Accuracy rate of this logistic regression classifier on the test dataset: {:.4f}'.format(training_logReg.score(iv_test, dv_test)))
#  
# 
# # =============================================================================
# # # -------------------------- Random Forest----------------------------------
# # =============================================================================
# 
# # Train the model on training data
# training_rf = train_models(new_df, models[1])
# 
# # Using the predict function from the Random Forest library to make predictions
# # on the test dataset
# predicting_rf = training_rf.predict(iv_test)
# 
# # Generating probabilities for each class
# rf_probs = training_rf.predict_proba(iv_test)[:, 1]
# 
# # Calculating  roc auc
# roc_auc_score(dv_test, rf_probs) #0.6487363600588887
#                           
# print(confusion_matrix(dv_test,predicting_rf))
# # =========================================================================
# #     [[ 3850 16695]
# #     [ 3621 41813]]
# # =========================================================================
# 
# print(classification_report(dv_test,predicting_rf))
# # =========================================================================
# #     precision    recall  f1-score   support
# # 
# #            0       0.52      0.19      0.27     20545
# #            1       0.71      0.92      0.80     45434
# # 
# #     accuracy                           0.69     65979
# #    macro avg       0.61      0.55      0.54     65979
# #    weighted avg    0.65      0.69      0.64     65979
# #     
# # =========================================================================
# print(accuracy_score(dv_test,predicting_rf)) #0.6920838448597281
# 
# =============================================================================
