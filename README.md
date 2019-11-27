## GEOG5995M-Programming for Social Scientists: Core Skills (Assignment 2):

  # Brief explanation of the files and their content:

    Python files: 
    - DataAnalysis.py: is the main file as all the functions are executed here. It comprises the Data Analysis process
                       including the prediction models.
    - DataAnalysis_Functions.py: includes 2 principal functions: 1) "readFile(file_name)" 2) "dataCleaning(df)"
    - Plots.py: includes various plots for the Exploratory Data Analysis (EDA), Features Importance plot, 
                models performance comparison plot and the correlation matrix plot.

    Files containing data:
    - LLCP2018XPT.zip: original dataset downloaded from the following website:
      "https://www.cdc.gov/brfss/annual_data/annual_2018.html"
    - BRFSS2018.csv: includes selected and renamed columns from the original dataset.
    - BRFSS_cleaned.csv: includes cleaned dataset (without missing and meaningless values such as "Don't know").      
                         Futhermore, replacing Feature values with their actual meaning.

    Plots folder: containing all the plots as images in "png" format.

    Python_assignment2_UML.png: UML diagram to show the whole process.

    docs folder: includes Sphinx documentation files (Makefile, make.bat, .rst, .py and html files). 
                 In order to view the documentation: click on the "index.html" file located inside the "_build" folder.

  # Follow the steps below in order to run the code:

    1) Unzip the 'LLCP2018XPT.zip' file.
    2) Run the "DataAnalysis.py" file as it is the main file which calls the other 2 python files: 
         "DataAnalysis_Functions.py" and "Plots.py".
    3) At this point the csv files containing specific features from the original dataset are created:
       "BRFSS2018.csv" and  "BRFSS_cleaned.csv".
    4) Plots folder is created, which contains all the plots generated for Exploratory Data Analysis purposes.
    5) Data Analysis is perfomed and the main results are printed in the console.


  # ALL THE FUNCTIONS HAVE BEEN TESTED THOROUGHLY: (TESTED: YES OR NO)
    - "readFile(file_name)": YES
    - "dataCleaning(df)": YES
    -  All the plots located inside the "Plots.py": YES
    - "logistic_regression()": YES
    - FEATURE SELECTION: YES
    - "random_forest()": YES

  # License:
    
    License is located inside the "GEOG5995M-Assignment2" folder.


     
