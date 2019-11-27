#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:05:09 2019

@author: riddaali
"""

    
import pandas as pd  
import seaborn as sns # For plots/graphs  
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

riddas_palette = ['#fe46a5', '#7f5f00','#c1f80a', '#3b638c', '#82cafc',
                  '#90b134', '#08ff08',  '#9d7651', '#C03028', '#06470c',
                  '#7038F8', '#E0C068', '#EE99AC']
												

# =============================================================================
# Placing the legend at the top right corner within the plot by setting the
# "loc" (location) = 0 and the "borderaxespad" (pad among the axes and the 
# border) = 0.
# =============================================================================						
def plot_obesity_age(new_df):
    """ 
    Plot 1: Obesity vs Age.
    
    :param [new_df]: [cleaned dataframe that will be used to generate the 
    "Obesity vs Age" plot]
    """
    fig1 = plt.figure()    
    sns.countplot(y='Overweight_Obese', hue='Age_Categories', data=new_df,
                  palette=riddas_palette)
    
    # =============================================================================
    # Saving the plot in the specified path by using the function "savefig" 
    # from the "matplotlib.pyplot" library
    # =============================================================================
    fig1.savefig('./Plots/Obesity_vs_Age') 
    plt.show()
    



def plot_obesity_education(new_df):
    """ 
    Plot 2: Obesity vs Education Level.
    
    :param [new_df]: [cleaned dataframe that will be used to generate the 
    "Obesity vs Education Level" plot]
    """
    fig2 = plt.figure()
    sns.countplot(y='Overweight_Obese', hue='Education_Level', data=new_df, 
                  palette="Set2")
    plt.legend(title="Level of Education",loc=0, borderaxespad=0.)
    fig2.savefig('./Plots/Obesity_vs_Education') 
    plt.show()
    


def plot_obesity_income(new_df):
    """ 
    Plot 3: Obesity vs Income Categories.
    
    :param [new_df]: [cleaned dataframe that will be used to generate the 
    "Obesity vs Income Categories" plot]
    """
    fig3 = plt.figure()
    sns.countplot(y='Overweight_Obese', hue='Income_Categories', data=new_df,
                  palette="Set2")
    plt.legend(title="Income Categories",loc=0, borderaxespad=0.)
    fig3.savefig('./Plots/Obesity_vs_Income') 
    plt.show()


def plot_obesity_exercise(new_df):
    """ 
    Plot 4: Obesity vs Exercise.
    
    :param [new_df]: [cleaned dataframe that will be used to generate the 
    "Obesity vs Exercise" plot]
    """
    fig4 = plt.figure()
    sns.countplot(y='Overweight_Obese', hue='Exercise', data=new_df,
                  palette="Set2")
    plt.legend(title="Exercise",loc=0, borderaxespad=0.)
    fig4.savefig('./Plots/Obesity_vs_Exercise') 
    plt.show()
    


def plot_obesity_gender(new_df):
    """ 
    Plot 5: Obesity vs Gender.
    
    :param [new_df]: [cleaned dataframe that will be used to generate the 
    "Obesity vs Gender" plot]
    """
    fig5 = plt.figure()
    sns.countplot(y='Overweight_Obese', hue='Gender', data=new_df,
                  palette="Set2")
    plt.legend(title="Gender",loc=0, borderaxespad=0.)
    fig5.savefig('./Plots/Obesity_vs_Gender') 
    plt.show()
    
     
def plot_obesity_alcohol(new_df):
    """ 
    Plot 6: Obesity vs Alcohol Consumption.
    
    :param [new_df]: [cleaned dataframe that will be used to generate the 
    "Obesity vs Alcohol Consumption" plot]
    """
    fig6 = plt.figure()    
    sns.countplot(y='Overweight_Obese', hue='Alcohol_Consumption', data=new_df,
                  palette="Set2")
    plt.legend(title="Alcohol Consumption",loc=0, borderaxespad=0.)
    fig6.savefig('./Plots/Obesity_vs_Alcohol') 
    plt.show()
    

def plot_obesity_arthritis(new_df):
    """ 
    Plot 7: Obesity vs Arthritis.
    
    :param [new_df]: [cleaned dataframe that will be used to generate the 
    "Obesity vs Arthritis" plot]
    """
    fig7 = plt.figure()
    sns.countplot(y='Overweight_Obese', hue='Arthritis', data=new_df, 
                  palette="Set2")
    plt.legend(title="Arthritis", loc=0, borderaxespad=0.)
    fig7.savefig('./Plots/Obesity_vs_Arthritis') 
    plt.show()
    
    


def featureImportance():
    """ 
    Creating the "Feature Importance" plot for the Random Forest model by using
    the built-in class "feature_importances_".
    
    Source used for guidance:
    - https://stackoverflow.com/questions/44101458/random-forest-feature-importance-chart-using-python
    """
    
    new_df = pd.read_csv('BRFSS_cleaned.csv')
    lr_df = new_df.apply(preprocessing.LabelEncoder().fit_transform)
    
    # Defining independent variables ("iv")
    iv = lr_df.loc[:, ['Education_Level', 'Income_Categories', 
                      'Exercise', 'Age_Categories', 'Gender', 'Alcohol_Consumption',
                      'Arthritis']]
    
    # Defining the dependent variable ("dv")
    dv = lr_df.loc[:, ['Overweight_Obese']]
    
    rf_model = RandomForestClassifier()
    rf_model.fit(iv, dv)
    
    # Using the built-in class "feature_importances_" of the Random Forest
    print(rf_model.feature_importances_) 
    
    # Plotting graph of feature importances for clearer visualization 
    plt.figure()
    features_importance = pd.Series(rf_model.feature_importances_, index=iv.columns) 
    features_importance.nlargest(10).plot(kind='barh')
    plt.savefig('./Plots/Feature_Importance_plot') 
    plt.show()
    
    
    
def comparingModelsPerformance():
    """
    Code adapted from the following source:
    - https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
    """
    new_df = pd.read_csv('BRFSS_cleaned.csv')
    lr_df = new_df.apply(preprocessing.LabelEncoder().fit_transform)
    
    # Defining independent variables ("iv")
    iv = lr_df.loc[:, ['Education_Level', 'Income_Categories', 
                      'Exercise', 'Age_Categories', 'Gender', 'Alcohol_Consumption',
                      'Arthritis']]
    
    # Defining the dependent variable ("dv")
    dv = lr_df.loc[:, ['Overweight_Obese']]
    
    prediction_models=[]
    prediction_models.append(('Logisitic Regression', LogisticRegression()))
    prediction_models.append(('Random Forest', RandomForestClassifier()))
    
    # Evaluating the performance of each model:
    outcomes = []
    models_names = []
    accuracy_score = 'accuracy'
    for name, model in prediction_models:
        
    	folds10 = model_selection.KFold(n_splits=10, random_state=0)
    	cv_outcomes = model_selection.cross_val_score(model, iv, dv, cv=folds10, 
                                                   scoring=accuracy_score)
    	outcomes.append(cv_outcomes)
    	models_names.append(name)
    	mean_accuracy = "%s: %.4f" % (name, cv_outcomes.mean())
        # Logisitic Regression: 0.6881 | Random Forest: 0.6926
    	print(mean_accuracy) 
   
    fig = plt.figure()
    fig.suptitle('Logistic Regression VS Random Forest')
    # 1x1 Grid 
    ax = fig.add_subplot(1, 1, 1)
    plt.boxplot(outcomes)
    ax.set_xticklabels(models_names)
    fig.savefig('./Plots/Models_comparison_plot') 
    plt.show()
    

def correlation_matrix(new_df):
    """
    Creating a correlation matrix to see the correlation between various variables
    Sources used (code adapted from the following websites): 
      - https://stackoverflow.com/questions/48035381/correlation-among-multiple-categorical-variables-pandas
      - https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
      - https://github.com/mwaskom/seaborn/issues/1773
      - https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib
      
      
      :param [new_df]: [cleaned dataframe that will be used to generate the
      correlation matrix plot]
    """
    
    # =============================================================================
    # Encoding the dataframe to convert it into an enumerated type by using 
    # the "pd.factorize()" function. Specifying the correlation method ('pearson')
    # and the minimum number of observations ("min_periods=1")
    # =============================================================================
    corr_data = new_df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', 
                            min_periods=1)
    corr = corr_data.corr()
    plt.figure()
    f,ax = plt.subplots(figsize=(15, 10))
    ax=sns.heatmap(corr, annot=True)
    bottom, top = plt.ylim() # Getting the current y-axis limits (coordinates)
    bottom += 0.5 # adding 0.5 to the bottom
    top -= 0.5 # subtracting 0.5 from the top
    plt.ylim(bottom, top) # updating the ylim for bottom & top values
    
    
    # Rotating the tick labels to ensure that they are clearly visible 
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=20,
        horizontalalignment='right'
    )
    #plt.savefig('./Plots/Correlation_matrix_plot') 
    plt.show()
    
    