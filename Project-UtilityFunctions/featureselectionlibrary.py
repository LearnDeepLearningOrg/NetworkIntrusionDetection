#Utility functions
from defineInputs import getLabelName

from featureencodinglibrary import featureEncodingUsingLabelEncoder
from dataformatinglibrary import printList

#Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np
import pandas as pd
import math
import scipy.stats as ss
from collections import Counter
from sklearn.ensemble import RandomForestClassifier #RandomForestClassifier: Falls under wrapper methods (feature importance)
from sklearn.ensemble import ExtraTreesClassifier #ExtraTreesClassifier: Falls under wrapper methods (feature importance)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

#This function is used to calculate the conditional entropy between a given feature and the target
def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

#This function is used to perform feature selection using TheilU 
#In TheilU we calculate the uncertainty coefficient between the given feature and the target
def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    
def featureSelectionUsingTheilU(dataSetForFeatureSelection):
    print("\n****** Start performing feature selection using TheilU *****")
    print("****** Falls under the group of techniques that use correlation matrix with Heatmap *****")

    labelName = getLabelName()
    label = dataSetForFeatureSelection[labelName]

    theilu = pd.DataFrame(index=[labelName],columns=dataSetForFeatureSelection.columns)
    columns = dataSetForFeatureSelection.columns
    dataSetAfterFeatuerSelection = dataSetForFeatureSelection

    for j in range(0,len(columns)):
        u = theil_u(label.tolist(),dataSetForFeatureSelection[columns[j]].tolist())
        theilu.loc[:,columns[j]] = u
        if u < 0.50:
            dataSetAfterFeatuerSelection.pop(columns[j])

    print('***** Ploting the uncertainty coefficient between the target and each feature *****')
    theilu.fillna(value=np.nan,inplace=True)
    plt.figure(figsize=(30,1))
    sns.heatmap(theilu,annot=True,fmt='.2f')
    plt.show()

    numberOfFeaturesInTheDatasetAfterFeatureSelection = len(dataSetAfterFeatuerSelection.columns)
    print('***** Number of columns in the dataSet after feature selection: ', len(dataSetAfterFeatuerSelection.columns))
    print('***** Columns in the dataSet after feature selection: \n', dataSetAfterFeatuerSelection.columns)
    print("****** End performing feature selection using TheilU *****")
    return dataSetAfterFeatuerSelection
	
#This function is used to perform feature selection using Chi-squared test
def featureSelectionUsingChisquaredTest(dataSetForFeatureSelection):
    print("\n****** Start performing feature selection using ChisquaredTest *****")
    print("****** Falls under filter methods (univariate selection) *****")
    
    numberOfFeatureToBeSelected = 10
    labelName = getLabelName()

    #To be able to apply Chi-squared test
    dataSetForFeatureSelection = featureEncodingUsingLabelEncoder(dataSetForFeatureSelection)
    dataSetAfterFeatuerSelection = dataSetForFeatureSelection
    
    #features = dataSetForFeatureSelection.iloc[:,0:len(dataSetForFeatureSelection.columns)-1]  
    features = dataSetForFeatureSelection.drop([labelName],axis=1)
    label = dataSetForFeatureSelection[labelName]
    
    #Apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=numberOfFeatureToBeSelected)
    fitBestfeatures = bestfeatures.fit(features,label)
    columns = pd.DataFrame(features.columns)
    scores = pd.DataFrame(fitBestfeatures.scores_)
    #concat two dataframes for better visualization 
    scoresOfBestFeatures = pd.concat([columns,scores],axis=1)
    scoresOfBestFeatures.columns = ['Features','Score']
    print("\n***** Scores for each feature in the dataset are *****")
    print(scoresOfBestFeatures.nlargest(numberOfFeatureToBeSelected,'Score'))
       
    mask = fitBestfeatures.get_support() 
    for j in range(0,len(mask)):
        if (mask[j] == False):
            dataSetAfterFeatuerSelection.pop(features.columns[j])
    
    numberOfFeaturesInTheDatasetAfterFeatureSelection = len(dataSetAfterFeatuerSelection.columns)    
    print('***** Number of columns in the dataSet after feature selection: ', len(dataSetAfterFeatuerSelection.columns))
    print('***** Columns in the dataSet after feature selection: \n', dataSetAfterFeatuerSelection.columns)
    print("****** End performing feature selection using ChisquaredTest *****")
   
    return dataSetAfterFeatuerSelection

#This function is used to perform feature selection using RandomForestClassifier
def featureSelectionUsingRandomForestClassifier(dataSetForFeatureSelection):
    print("\n****** Start performing feature selection using RandomForestClassifier *****")
    print("****** Falls under wrapper methods (feature importance) *****")
	
    labelName = getLabelName()

    #Applying feature encoding before applying the RandomForestClassification
    dataSetForFeatureSelection = featureEncodingUsingLabelEncoder(dataSetForFeatureSelection)
    dataSetAfterFeatuerSelection = dataSetForFeatureSelection
    #features = dataSetForFeatureSelection.iloc[:,0:len(dataSetForFeatureSelection.columns)-1]  
    features = dataSetForFeatureSelection.drop([labelName],axis=1)
    label = dataSetForFeatureSelection[labelName]

    labelencoder = LabelEncoder()
    labelTransformed = labelencoder.fit_transform(label)

    print("****** RandomForestClassification is in progress *****")
    #Train using RamdomForestClassifier
    trainedforest = RandomForestClassifier(n_estimators=700).fit(features,labelTransformed)
    importances = trainedforest.feature_importances_ #array with importances of each feature
    idx = np.arange(0, features.shape[1]) #create an index array, with the number of features
    features_to_keep = idx[importances > np.mean(importances)] #only keep features whose importance is greater than the mean importance
    featureImportances = pd.Series(importances, index= features.columns)
    selectedFeatures = featureImportances.nlargest(len(features_to_keep))
    print("\n selectedFeatures after RandomForestClassification: ", selectedFeatures)
    print("****** Completed RandomForestClassification *****")

    #Plot the feature Importance to see which features have been considered as most important for our model to make its predictions
    #figure(num=None, figsize=(20, 22), dpi=80, facecolor='w', edgecolor='k')
    #selectedFeatures.plot(kind='barh')

    selectedFeaturesNames = selectedFeatures.keys()
    dataSetForFeatureSelection = dataSetForFeatureSelection.drop(selectedFeaturesNames,axis=1)
    dataSetAfterFeatuerSelection = dataSetAfterFeatuerSelection.drop(dataSetForFeatureSelection.columns, axis=1)
    dataSetAfterFeatuerSelection[labelName] = label
    
    numberOfFeaturesInTheDatasetAfterFeatureSelection = len(dataSetAfterFeatuerSelection.columns)    
    print('\n***** Number of columns in the dataSet after feature selection: ', len(dataSetAfterFeatuerSelection.columns))
    print('***** Columns in the dataSet after feature selection: \n', dataSetAfterFeatuerSelection.columns)
    print("****** End performing feature selection using RandomForestClassifier *****")
    return dataSetAfterFeatuerSelection
	
#This function is used to perform feature selection using ExtraTreesClassifier
def featureSelectionUsingExtraTreesClassifier(dataSetForFeatureSelection):
    print("\n****** Start performing feature selection using ExtraTreesClassifier *****")
    print("****** Falls under wrapper methods (feature importance) *****")
    
    labelName = getLabelName()

    #Applying feature encoding before applying the ExtraTreesClassification
    dataSetForFeatureSelection = featureEncodingUsingLabelEncoder(dataSetForFeatureSelection)
    dataSetAfterFeatuerSelection = dataSetForFeatureSelection
    #features = dataSetForFeatureSelection.iloc[:,0:len(dataSetForFeatureSelection.columns)-1]  
    features = dataSetForFeatureSelection.drop([labelName],axis=1)
    label = dataSetForFeatureSelection[labelName]

    labelencoder = LabelEncoder()
    labelTransformed = labelencoder.fit_transform(label)
	
    print("****** ExtraTreesClassification is in progress *****")
    #Train using ExtraTreesClassifier
    trainedforest = ExtraTreesClassifier(n_estimators=700).fit(features,labelTransformed)
    importances = trainedforest.feature_importances_ #array with importances of each feature
    idx = np.arange(0, features.shape[1]) #create an index array, with the number of features
    features_to_keep = idx[importances > np.mean(importances)] #only keep features whose importance is greater than the mean importance
    featureImportances = pd.Series(importances, index= features.columns)
    selectedFeatures = featureImportances.nlargest(len(features_to_keep))
    print("\n selectedFeatures after ExtraTreesClassification: ", selectedFeatures)
    print("****** Completed ExtraTreesClassification *****")

    #Plot the feature Importance to see which features have been considered as most important for our model to make its predictions
    #figure(num=None, figsize=(20, 22), dpi=80, facecolor='w', edgecolor='k')
    #selectedFeatures.plot(kind='barh')

    selectedFeaturesNames = selectedFeatures.keys()
    dataSetForFeatureSelection = dataSetForFeatureSelection.drop(selectedFeaturesNames,axis=1)
    dataSetAfterFeatuerSelection = dataSetAfterFeatuerSelection.drop(dataSetForFeatureSelection.columns, axis=1)
    dataSetAfterFeatuerSelection[labelName] = label
    
    numberOfFeaturesInTheDatasetAfterFeatureSelection = len(dataSetAfterFeatuerSelection.columns)    
    print('\n***** Number of columns in the dataSet after feature selection: ', len(dataSetAfterFeatuerSelection.columns))
    print('***** Columns in the dataSet after feature selection: \n', dataSetAfterFeatuerSelection.columns)
    print("****** End performing feature selection using ExtraTreesClassifier *****")
    return dataSetAfterFeatuerSelection

