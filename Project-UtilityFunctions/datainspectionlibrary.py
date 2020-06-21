#Data formating library
from dataformatinglibrary import printList

#Data pre-processing library
from datapreprocessinglibrary import checkForMissingValues
from datapreprocessinglibrary import checkForDulicateRecords

#Utility functions
from defineInputs import getLabelName

#Libraries for feature selection
#SelectKBest, Chi2: Falls under filter methods (univariate selection)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier #RandomForestClassifier: Falls under wrapper methods (feature importance)
from sklearn.ensemble import ExtraTreesClassifier #ExtraTreesClassifier: Falls under wrapper methods (feature importance)

import numpy as np

#This function is used to check the statistics of a given dataSet
def getStatisticsOfData (dataSet):
    print("***** Start checking the statistics of the dataSet *****\n")
    
    labelName = getLabelName()
    #Number of rows and columns in the dataset
    print("***** Shape (number of rows and columns) in the dataset: ", dataSet.shape)
    
    #Total number of features in the dataset
    numberOfColumnsInTheDataset = len(dataSet.drop([labelName],axis=1).columns)
    #numberOfColumnsInTheDataset = len(dataSet.columns)
    print("***** Total number of features in the dataset: ",numberOfColumnsInTheDataset)
    
    #Total number of categorical featuers in the dataset
    categoricalFeaturesInTheDataset = list(set(dataSet.drop([labelName],axis=1).columns) - set(dataSet.drop([labelName],axis=1)._get_numeric_data().columns))
    #categoricalFeaturesInTheDataset = list(set(dataSet.columns) - set(dataSet._get_numeric_data().columns))
    print("***** Number of categorical features in the dataset: ",len(categoricalFeaturesInTheDataset))
  
    #Total number of numerical features in the dataset
    numericalFeaturesInTheDataset = list(dataSet.drop([labelName],axis=1)._get_numeric_data().columns)
    #numericalFeaturesInTheDataset = list(dataSet._get_numeric_data().columns)
    print("***** Number of numerical features in the dataset: ",len(numericalFeaturesInTheDataset))

    #Names of categorical features in the dataset
    print("\n***** Names of categorical features in dataset *****\n")
    printList(categoricalFeaturesInTheDataset,'Categorical features in dataset')

    #Names of numerical features in the dataset
    print("\n***** Names of numerical features in dataset *****\n")
    printList(numericalFeaturesInTheDataset,'Numerical features in the dataset')
    
    #Checking for any missing values in the data set
    anyMissingValuesInTheDataset = checkForMissingValues(dataSet)
    print("\n***** Are there any missing values in the data set: ", anyMissingValuesInTheDataset)
      
    anyDuplicateRecordsInTheDataset = checkForDulicateRecords(dataSet)
    print("\n***** Are there any duplicate records in the data set: ", anyDuplicateRecordsInTheDataset)
    #Check if there are any duplicate records in the data set
    if (anyDuplicateRecordsInTheDataset):
        dataSet = dataSet.drop_duplicates()
        print("Number of records in the dataSet after removing the duplicates: ", len(dataSet.index))

    #How many number of different values for label that are present in the dataset
    print('\n****** Number of different values for label that are present in the dataset: ',dataSet[labelName].nunique())
    #What are the different values for label in the dataset
    print('\n****** Here is the list of unique label types present in the dataset ***** \n')
    printList(list(dataSet[getLabelName()].unique()),'Unique label types in the dataset')

    #What are the different values in each of the categorical features in the dataset
    print('\n****** Here is the list of unique values present in each categorical feature in the dataset *****\n')
    categoricalFeaturesInTheDataset = list(set(dataSet.columns) - set(dataSet._get_numeric_data().columns))
    numericalFeaturesInTheDataset = list(dataSet._get_numeric_data().columns)
    for feature in categoricalFeaturesInTheDataset:
        uniq = np.unique(dataSet[feature])
        print('\n{}: {} '.format(feature,len(uniq)))
        printList(dataSet[feature].unique(),'distinct values')
        
    print('\n****** Label distribution in the dataset *****\n')
    print(dataSet[labelName].value_counts())
    print()

    print("\n***** End checking the statistics of the dataSet *****")