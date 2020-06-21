import pandas as pd
import numpy as np

#Libraries for feature encoding
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

#Utility functions
from defineInputs import getLabelName
from dataformatinglibrary import printList

#This function is used to perform one hot encoding on the categorical features in the given dataset
def featureEncodingUsingOneHotEncoder(dataSetForFeatureEncoding):
    print("****** Start one hot encoding on the categorical features in the given dataset *****")
	
    labelName = getLabelName()
    #Extract the categorical features, leave the label
    categoricalColumnsInTheDataSet = dataSetForFeatureEncoding.drop([labelName],axis=1).select_dtypes(['object'])
    #Get the names of the categorical features
    categoricalColumnNames = categoricalColumnsInTheDataSet.columns.values
    
    print("****** Number of features before one hot encoding: ",len(dataSetForFeatureEncoding.columns))
    print("****** Number of categorical features in the dataset: ",len(categoricalColumnNames))
    print("****** Categorical feature names in the dataset: ",categoricalColumnNames)
    
    print('\n****** Here is the list of unique values present in each categorical feature in the dataset *****\n')
    categoricalFeaturesInTheDataset = list(set(dataSetForFeatureEncoding.columns) - set(dataSetForFeatureEncoding._get_numeric_data().columns))
    numericalFeaturesInTheDataset = list(dataSetForFeatureEncoding._get_numeric_data().columns)
    for feature in categoricalFeaturesInTheDataset:
        uniq = np.unique(dataSetForFeatureEncoding[feature])
        print('\n{}: {} '.format(feature,len(uniq)))
        printList(dataSetForFeatureEncoding[feature].unique(),'distinct values')
        
    #Using get_dummies function to get the dummy variables for the categorical columns
    onHotEncodedDataSet=pd.get_dummies(dataSetForFeatureEncoding, columns=categoricalColumnNames, prefix=categoricalColumnNames)
    
    #Move the label column to the end
    label = onHotEncodedDataSet.pop(labelName)
    onHotEncodedDataSet[labelName] = label
    numberOfColumnsInOneHotEncodedDataset = len(onHotEncodedDataSet.columns)
    print("****** Number of features after one hot encoding: ",numberOfColumnsInOneHotEncodedDataset)

    print("****** End one hot encoding on the categorical features in the given dataset *****\n")
    return onHotEncodedDataSet
	
#This function is used to perform label encoding on the categorical features in the given dataset
def featureEncodingUsingLabelEncoder(dataSetForFeatureEncoding):
    print("****** Start label encoding on the categorical features in the given dataset *****")

    labelName = getLabelName()
    #Extract the categorical features, leave the label
    categoricalColumnsInTheDataSet = dataSetForFeatureEncoding.drop([labelName],axis=1).select_dtypes(['object'])
    #Get the names of the categorical features
    categoricalColumnNames = categoricalColumnsInTheDataSet.columns.values
 
    print("****** Number of features before label encoding: ",len(dataSetForFeatureEncoding.columns))
    print("****** Number of categorical features in the dataset: ",len(categoricalColumnNames))
    print("****** Categorical feature names in the dataset: ",categoricalColumnNames)

    print('\n****** Here is the list of unique values present in each categorical feature in the dataset *****\n')
    labelEncoder = LabelEncoder() 
    for feature in categoricalColumnNames:
        uniq = np.unique(dataSetForFeatureEncoding[feature])
        print('\n{}: {} '.format(feature,len(uniq)))
        printList(dataSetForFeatureEncoding[feature].unique(),'distinct values')
        dataSetForFeatureEncoding[feature] = labelEncoder.fit_transform(dataSetForFeatureEncoding[feature]) 
    print("****** Number of features after label encoding: ",len(dataSetForFeatureEncoding.columns))    
    
    print("****** End label encoding on the categorical features in the given dataset *****\n")
    return dataSetForFeatureEncoding
	
#This function is used to perform binary encoding on the categorical features in the given dataset
def featureEncodingUsingBinaryEncoder(dataSetForFeatureEncoding):
    print("****** Start binary encoding on the categorical features in the given dataset *****")

    labelName = getLabelName()
    #Extract the categorical features, leave the label
    categoricalColumnsInTheDataSet = dataSetForFeatureEncoding.drop([labelName],axis=1).select_dtypes(['object'])
    #Get the names of the categorical features
    categoricalColumnNames = categoricalColumnsInTheDataSet.columns.values
 
    print("****** Number of features before binary encoding: ",len(dataSetForFeatureEncoding.columns))
    print("****** Number of categorical features in the dataset: ",len(categoricalColumnNames))
    print("****** Categorical feature names in the dataset: ",categoricalColumnNames)

    print('\n****** Here is the list of unique values present in each categorical feature in the dataset *****\n')
    label = dataSetForFeatureEncoding.drop(dataSetForFeatureEncoding.loc[:, ~dataSetForFeatureEncoding.columns.isin([labelName])].columns, axis = 1)
    for feature in categoricalColumnNames:
        uniq = np.unique(dataSetForFeatureEncoding[feature])
        print('\n{}: {} '.format(feature,len(uniq)))
        printList(dataSetForFeatureEncoding[feature].unique(),'distinct values')
        featureColumns = dataSetForFeatureEncoding.drop(dataSetForFeatureEncoding.loc[:, ~dataSetForFeatureEncoding.columns.isin([feature])].columns, axis = 1)
        binaryEncoder = ce.BinaryEncoder(cols = [feature])
        binaryEncodedFeature = binaryEncoder.fit_transform(featureColumns, label)
        dataSetForFeatureEncoding = dataSetForFeatureEncoding.join(binaryEncodedFeature)
        dataSetForFeatureEncoding = dataSetForFeatureEncoding.drop(feature, axis=1)

    dataSetForFeatureEncoding = dataSetForFeatureEncoding.drop(labelName, axis=1)
    dataSetForFeatureEncoding[labelName] = label
    print("****** Number of features after binary encoding: ",len(dataSetForFeatureEncoding.columns))    
    
    print("****** End binary encoding on the categorical features in the given dataset *****\n")
    return dataSetForFeatureEncoding
	
#This function is used to perform frequency encoding on the categorical features in the given dataset
def featureEncodingUsingFrequencyEncoder(dataSetForFeatureEncoding):
    print("****** Start frequency encoding on the categorical features in the given dataset *****")

    labelName = getLabelName()
    #Extract the categorical features, leave the label
    categoricalColumnsInTheDataSet = dataSetForFeatureEncoding.drop([labelName],axis=1).select_dtypes(['object'])
    #Get the names of the categorical features
    categoricalColumnNames = categoricalColumnsInTheDataSet.columns.values
 
    print("****** Number of features before label encoding: ",len(dataSetForFeatureEncoding.columns))
    print("****** Number of categorical features in the dataset: ",len(categoricalColumnNames))
    print("****** Categorical feature names in the dataset: ",categoricalColumnNames)

    print('\n****** Here is the list of unique values present in each categorical feature in the dataset *****\n')
    label = dataSetForFeatureEncoding.drop(dataSetForFeatureEncoding.loc[:, ~dataSetForFeatureEncoding.columns.isin([labelName])].columns, axis = 1)
    for feature in categoricalColumnNames:
        uniq = np.unique(dataSetForFeatureEncoding[feature])
        print('\n{}: {} '.format(feature,len(uniq)))
        printList(dataSetForFeatureEncoding[feature].unique(),'distinct values')
        frequencyEncoder = dataSetForFeatureEncoding.groupby(feature).size()/len(dataSetForFeatureEncoding)
        dataSetForFeatureEncoding.loc[:,feature+"_Encoded"] = dataSetForFeatureEncoding[feature].map(frequencyEncoder)
        dataSetForFeatureEncoding = dataSetForFeatureEncoding.drop(feature, axis=1)

    dataSetForFeatureEncoding = dataSetForFeatureEncoding.drop(labelName, axis=1)
    dataSetForFeatureEncoding[labelName] = label
    print("****** Number of features after frequency encoding: ",len(dataSetForFeatureEncoding.columns))    
    
    print("****** End frequency encoding on the categorical features in the given dataset *****\n")
    return dataSetForFeatureEncoding