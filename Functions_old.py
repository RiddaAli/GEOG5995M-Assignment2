#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:19:34 2019

@author: riddaali
"""

import pandas as pd 

# Creating this "readFile()" function to read the csv files. 
# It takes the file name as argument.
def readFile(file_name):
    file_output = pd.read_csv(file_name)

    return file_output

# Dimension before removing meaningless values (e.g. Don't know, refused, etc.) 
# [434900 rows x 8 columns]
def dataCleaning(df):
    df = df[df['Overweight_Obese'] != 9]
    df = df[df['Education_Level'] != 9]
    df = df[df['Income_Categories'] != 9]
    df = df[df['Exercise'] != 9]
    df = df[df['Age_Categories'] != 14]
    df = df[df['Gender'] != 7]
    df = df[df['Gender'] != 9]
    df = df[df['Alcohol_Consumption'] != 7]
    df = df[df['Alcohol_Consumption'] != 9]
    
    return df