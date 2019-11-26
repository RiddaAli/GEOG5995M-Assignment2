#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:21:22 2019

@author: riddaali
"""

import xport


def readFile(file_name):
    """ 
    Creating this "readFile()" function to read the csv files. 
    It takes the file name as argument.
    
    1) Reading the BRFSS 2018 data: https://www.cdc.gov/brfss/annual_data/annual_2018.html
    Converting it from a "*.xpt" (XPORT) file to a dataframe.
    
    PLESAE NOTE: I COULD NOT SUBMIT THE XPORT ('LLCP2018.XPT') FILE DUE TO ITS
    BIG SIZE (962 MB) SO IN ORDER TO EXECUTE THE FUNCTION BELOW, PLEASE UNZIP
    THE 'LLCP2018XPT.zip' FILE.
    
    With the help of documentation available at: https://pypi.org/project/xport/
    
    2) Selecting features from the original dataframe by using the loc() function
    
    3) Renaming each column to improve its readability
    
    4) Converting each column into categorical variables
    
    5) Exporting the dataframe by saving it as a csv file
    
    :param [file_name]: [name of the file that includes the data]
    :return: [CSV file that has been created in the specified path]
    
    """
    with open(file_name, 'rb') as file:
        df = xport.to_dataframe(file)
  
    # Selecting variables (columns) from the original dataframe by using the loc() function
    selectedColsDF = df.loc[:, ['_RFBMI5', '_EDUCAG', '_INCOMG', 'EXERANY2', 
                                '_AGEG5YR', 'SEX1', 'DRNKANY5', '_DRDXAR1']]
    
    # Renaming each column to improve its readability.
    selectedColsDF.columns = ['Overweight_Obese', 'Education_Level', 
                              'Income_Categories', 'Exercise', 'Age_Categories', 
                              'Gender', 'Alcohol_Consumption', 'Arthritis']
    
    # Converting each column from object into categorical variables
    # Source used for guidance: https://pbpython.com/pandas_dtypes.html
    for eachCol in selectedColsDF:
        selectedColsDF[eachCol] = selectedColsDF[eachCol].astype('category')
    #print(selectedColsDF.dtypes)

    
    # Saving the dataframe into a csv file called "BRFSS2018.csv"
    exportDF_csv = selectedColsDF.to_csv(r'./BRFSS2018.csv', 
                                         index = None, header=True) 

    return exportDF_csv





def dataCleaning(df):
    """  
    This function takes the dataframe as an arugment.
    
    1) Filtering missing and meaningless values (e.g. Don't know, refused, etc.)
       by using the loc technique.
    2) Replacing Feature values with their actual meaning by using dictionaries.
        
    :param [df]: [dataframe that will be used to carry out the data cleaning]
    :return: [cleaned (without NAs, meaningless values, renamed values) dataframe]
    
    """
    
    df = df.loc[(df['Overweight_Obese'] != 9) & (df['Education_Level'] != 9)
                & (df['Income_Categories'] != 9) & (df['Exercise'] != 7) 
                & (df['Exercise'] != 9) & (df['Age_Categories'] != 14) 
                & (df['Gender'] != 7) & (df['Gender'] != 9) 
                & (df['Alcohol_Consumption'] != 7) 
                & (df['Alcohol_Consumption'] != 9)]
    
    # =========================================================================
    # Creating a copy of the BRFSS dataframe while removing all the missing 
    # values (NaNs) to ensure that the modifications to the data will not be
    # reflected in the original dataframe
    # =========================================================================
    df = df.dropna().copy()
    
    gender_dict = {1: 'Male', 2:'Female'}

    for k1, v1 in gender_dict.items():
        df.Gender.replace(k1, v1, inplace=True)
    
    
    overweight_dict = {1: 'No', 2: 'Yes'}
    
    for k2, v2 in overweight_dict.items():
        df.Overweight_Obese.replace(k2, v2, inplace =True)
    
    
    education_dict = {1: 'No_HighSchool_Graduate', 2: 'HighSchool_Graduate', 
                      3: 'Attended_College', 4: 'College_Graduate'}
    
    for k3, v3 in education_dict.items():
        df.Education_Level.replace(k3, v3, inplace=True)
    
    
    income_dict = {1: '<$15,000', 2: '$15,000_to_<$25,000',
                   3: '$25,000_to_<$35,000', 4: '$35,000_to_<$50,000',
                   5: '$50,000>='}
    
    for k4, v4 in income_dict.items():
        df.Income_Categories.replace(k4, v4, inplace=True)
    
    
    exercise_dict = {1: 'Yes', 2: 'No'}
    
    for k5, v5 in exercise_dict.items():
        df.Exercise.replace(k5, v5, inplace=True)
    
    
    age_dict = {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
                6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
                11: '70-74', 12: '75-79', 13: '80>='}
    
    for k6, v6 in age_dict.items():
        df.Age_Categories.replace(k6, v6, inplace=True)
    
    
    alcohol_dict = {1: 'Yes', 2: 'No'}
    
    for k7, v7 in alcohol_dict.items():
        df.Alcohol_Consumption.replace(k7, v7, inplace=True)
    
    
    arthritis_dict = {1: 'Diagnosed', 2: 'Not_Diagnosed'}
    
    for k8, v8 in arthritis_dict.items():
        df.Arthritis.replace(k8, v8, inplace=True)
    
    
    return df


    
