import pandas as pd

#Utils
import operator

#Seaborn is an open source Python library providing high level API for visualizing the data 
import seaborn as sns
import matplotlib.pyplot as plt

#library for saving the trained models to files
import joblib

from defineInputs import getPathToTrainingAndTestingDataSets
from defineInputs import getPathToGenerateModels

#Data loading library
from dataloadinglibrary import loadCSV

from defineInputs import getLabelName

#Data pre-processing library
from datapreprocessinglibrary import splitCompleteDataSetIntoTrainingSetAndTestingSet

#Feature selection library
from featureselectionlibrary import featureSelectionUsingTheilU
from featureselectionlibrary import featureSelectionUsingChisquaredTest
from featureselectionlibrary import featureSelectionUsingRandomForestClassifier
from featureselectionlibrary import featureSelectionUsingExtraTreesClassifier

#feature encoding library
from featureencodinglibrary import featureEncodingUsingOneHotEncoder
from featureencodinglibrary import featureEncodingUsingLabelEncoder
from featureencodinglibrary import featureEncodingUsingBinaryEncoder
from featureencodinglibrary import featureEncodingUsingFrequencyEncoder

#feature scaling library
from featurescalinglibrary import featureScalingUsingMinMaxScaler
from featurescalinglibrary import featureScalingUsingStandardScalar
from featurescalinglibrary import featureScalingUsingBinarizer
from featurescalinglibrary import featureScalingUsingNormalizer

from classificationlibrary import classifyUsingDecisionTreeClassifier
from classificationlibrary import classifyUsingLogisticRegression
from classificationlibrary import classifyUsingLinearDiscriminantAnalysis
from classificationlibrary import classifyUsingGaussianNB
from classificationlibrary import classifyUsingRandomForestClassifier
from classificationlibrary import classifyUsingExtraTreesClassifier
from classificationlibrary import classifyUsingKNNClassifier
from classificationlibrary import findingOptimumNumberOfNeighboursForKNN

def compareModels(arrayOfModels):
    modelsAndAccuracies = {}
    for i in range(1,len(arrayOfModels)):
        data = arrayOfModels[i]
        modelsAndAccuracies[data[3]]=data[5]
    bestModelAndItsAccuracy = {}
    bestModelAndItsAccuracy[max(modelsAndAccuracies.items(), key=operator.itemgetter(1))[0]]=modelsAndAccuracies[max(modelsAndAccuracies.items(), key=operator.itemgetter(1))[0]]
    sns.set_style("whitegrid")
    plt.figure(figsize=(5,5))
    plt.ylabel("Algorithms",fontsize=10)
    plt.xlabel("Accuracy %",fontsize=10)
    plt.title("Comparing the models based on the accuries achieved",fontsize=15)
    sns.barplot(x=list(modelsAndAccuracies.values()), y=list(modelsAndAccuracies.keys()))
    plt.show()
    return bestModelAndItsAccuracy

### Below function is responsible for performing pre-processing, training, evaluation, persisting model
def performPreprocessingBuildModelsAndEvaluateAccuracy(trainingDataSet, testingDataSet, arrayOfModels):
    for i in range(1,len(arrayOfModels)):
        print('***************************************************************************************************************************')
        print('********************************************* Building Model-', i ,' As Below *************************************************')
        print('\t -- Feature Selection: \t ', arrayOfModels[i][0], ' \n\t -- Feature Encoding: \t ', arrayOfModels[i][1], ' \n\t -- Feature Scaling: \t ', arrayOfModels[i][2], ' \n\t -- Classification: \t ', arrayOfModels[i][3], '\n')
 
        trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath = getPathToTrainingAndTestingDataSets()
        trainingDataSet = loadCSV(trainingFileNameWithAbsolutePath)
        testingDataSet = loadCSV(testingFileNameWithAbsolutePath)

        labelName = getLabelName()
        label = trainingDataSet[labelName]

        #Combining the test and training datasets for preprocessing then together, because we observed that in sme datasets
        #the values in the categorical columns in test dataset and train dataset are being different this causes issues while
        #applying classification techniques
        completeDataSet = pd.concat(( trainingDataSet, testingDataSet ))

        #difficultyLevel = completeDataSet.pop('difficulty_level')
        
        print("completeDataSet.shape: ",completeDataSet.shape)
        print("completeDataSet.head: ",completeDataSet.head())

        #Feature Selection
        if arrayOfModels[i][0] == 'TheilsU':
            #Perform feature selection using TheilU
            completeDataSetAfterFeatuerSelection = featureSelectionUsingTheilU(completeDataSet)
        elif arrayOfModels[i][0] == 'Chi-SquaredTest':
            #Perform feature selection using Chi-squared Test
            completeDataSetAfterFeatuerSelection = featureSelectionUsingChisquaredTest(completeDataSet)
        elif arrayOfModels[i][0] == 'RandomForestClassifier':
            #Perform feature selection using RandomForestClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingRandomForestClassifier(completeDataSet)
        elif arrayOfModels[i][0] == 'ExtraTreesClassifier':
            #Perform feature selection using ExtraTreesClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingExtraTreesClassifier(completeDataSet)
        
        #Feature Encoding        
        if arrayOfModels[i][1] == 'LabelEncoder':
            #Perform lable encoding to convert categorical values into label encoded features
            completeEncodedDataSet = featureEncodingUsingLabelEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'OneHotEncoder':
            #Perform OnHot encoding to convert categorical values into one-hot encoded features
            completeEncodedDataSet = featureEncodingUsingOneHotEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'FrequencyEncoder':
            #Perform Frequency encoding to convert categorical values into frequency encoded features
            completeEncodedDataSet = featureEncodingUsingFrequencyEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'BinaryEncoder':
            #Perform Binary encoding to convert categorical values into binary encoded features
            completeEncodedDataSet = featureEncodingUsingBinaryEncoder(completeDataSetAfterFeatuerSelection)

        #Feature Scaling        
        if arrayOfModels[i][2] == 'Min-Max':
            #Perform MinMaxScaler to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingMinMaxScaler(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Binarizing':
            #Perform Binarizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingBinarizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Normalizing':
            #Perform Normalizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingNormalizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Standardization':
            #Perform Standardization to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingStandardScalar(completeEncodedDataSet)
        
        #Split the complete dataSet into training dataSet and testing dataSet
        featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet = splitCompleteDataSetIntoTrainingSetAndTestingSet(completeEncodedAndScaledDataset)
        
        trainingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTrainingDataSet, labelInPreProcessedTrainingDataSet], axis=1, sort=False)
        testingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTestingDataSet, labelInPreProcessedTestingDataSet], axis=1, sort=False)

        #Classification                
        if arrayOfModels[i][3] == 'DecisonTree':
            #Perform classification using DecisionTreeClassifier
            classifier, trainingAccuracyScore, testingAccuracyScore = classifyUsingDecisionTreeClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset)
        elif arrayOfModels[i][3] == 'RandomForestClassifier':
            classifier, trainingAccuracyScore, testingAccuracyScore = classifyUsingRandomForestClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset)
        elif arrayOfModels[i][3] == 'ExtraTreesClassifier':
            classifier, trainingAccuracyScore, testingAccuracyScore = classifyUsingExtraTreesClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset)
        elif arrayOfModels[i][3] == 'LogisticRegressionRegression':
            classifier, trainingAccuracyScore, testingAccuracyScore = classifyUsingLogisticRegression(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset)
        elif arrayOfModels[i][3] == 'LinearDiscriminantAnalysis':
            classifier, trainingAccuracyScore, testingAccuracyScore = classifyUsingLinearDiscriminantAnalysis(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset)
        elif arrayOfModels[i][3] == 'GuassianNaiveBayes':
            classifier, trainingAccuracyScore, testingAccuracyScore = classifyUsingGaussianNB(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset)
        elif arrayOfModels[i][3] == 'KNN':
            classifier, trainingAccuracyScore, testingAccuracyScore = classifyUsingKNNClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset)

        arrayOfModels[i].append(trainingAccuracyScore)
        arrayOfModels[i].append(testingAccuracyScore)
        
        modelName = arrayOfModels[i][0]+"_"+arrayOfModels[i][1]+"_"+arrayOfModels[i][2]+"_"+arrayOfModels[i][3]
        modelFileName = getPathToGenerateModels() + modelName+".pkl"
        arrayOfModels[i].append(modelName)
        arrayOfModels[i].append(modelFileName)
        #Save the model to file
        joblib.dump(classifier, modelFileName)
		
def performPreprocessing(trainingDataSet, testingDataSet, arrayOfModels):
    for i in range(0,len(arrayOfModels)):
        print('***************************************************************************************************************************')
        print('********************************************* Building Model-', i ,' As Below *************************************************')
        print('\t -- Feature Selection: \t ', arrayOfModels[i][0], ' \n\t -- Feature Encoding: \t ', arrayOfModels[i][1], ' \n\t -- Feature Scaling: \t ', arrayOfModels[i][2], '\n')
 
        trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath = getPathToTrainingAndTestingDataSets()
        trainingDataSet = loadCSV(trainingFileNameWithAbsolutePath)
        testingDataSet = loadCSV(testingFileNameWithAbsolutePath)

        labelName = getLabelName()
        label = trainingDataSet[labelName]

        #Combining the test and training datasets for preprocessing then together, because we observed that in sme datasets
        #the values in the categorical columns in test dataset and train dataset are being different this causes issues while
        #applying classification techniques
        completeDataSet = pd.concat(( trainingDataSet, testingDataSet ))

        #difficultyLevel = completeDataSet.pop('difficulty_level')
        
        print("completeDataSet.shape: ",completeDataSet.shape)
        print("completeDataSet.head: ",completeDataSet.head())

        #Feature Selection
        if arrayOfModels[i][0] == 'TheilsU':
            #Perform feature selection using TheilU
            completeDataSetAfterFeatuerSelection = featureSelectionUsingTheilU(completeDataSet)
        elif arrayOfModels[i][0] == 'Chi-SquaredTest':
            #Perform feature selection using Chi-squared Test
            completeDataSetAfterFeatuerSelection = featureSelectionUsingChisquaredTest(completeDataSet)
        elif arrayOfModels[i][0] == 'RandomForestClassifier':
            #Perform feature selection using RandomForestClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingRandomForestClassifier(completeDataSet)
        elif arrayOfModels[i][0] == 'ExtraTreesClassifier':
            #Perform feature selection using ExtraTreesClassifier
            completeDataSetAfterFeatuerSelection = featureSelectionUsingExtraTreesClassifier(completeDataSet)
        
        #Feature Encoding        
        if arrayOfModels[i][1] == 'LabelEncoder':
            #Perform lable encoding to convert categorical values into label encoded features
            completeEncodedDataSet = featureEncodingUsingLabelEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'OneHotEncoder':
            #Perform OnHot encoding to convert categorical values into one-hot encoded features
            completeEncodedDataSet = featureEncodingUsingOneHotEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'FrequencyEncoder':
            #Perform Frequency encoding to convert categorical values into frequency encoded features
            completeEncodedDataSet = featureEncodingUsingFrequencyEncoder(completeDataSetAfterFeatuerSelection)
        elif arrayOfModels[i][1] == 'BinaryEncoder':
            #Perform Binary encoding to convert categorical values into binary encoded features
            completeEncodedDataSet = featureEncodingUsingBinaryEncoder(completeDataSetAfterFeatuerSelection)

        #Feature Scaling        
        if arrayOfModels[i][2] == 'Min-Max':
            #Perform MinMaxScaler to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingMinMaxScaler(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Binarizing':
            #Perform Binarizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingBinarizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Normalizing':
            #Perform Normalizing to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingNormalizer(completeEncodedDataSet)
        elif arrayOfModels[i][2] == 'Standardization':
            #Perform Standardization to scale the features of the dataset into same range
            completeEncodedAndScaledDataset = featureScalingUsingStandardScalar(completeEncodedDataSet)
        
        #Split the complete dataSet into training dataSet and testing dataSet
        featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet = splitCompleteDataSetIntoTrainingSetAndTestingSet(completeEncodedAndScaledDataset)
        
        trainingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTrainingDataSet, labelInPreProcessedTrainingDataSet], axis=1, sort=False)
        testingEncodedAndScaledDataset = pd.concat([featuresInPreProcessedTestingDataSet, labelInPreProcessedTestingDataSet], axis=1, sort=False)
    
    return 	completeEncodedAndScaledDataset