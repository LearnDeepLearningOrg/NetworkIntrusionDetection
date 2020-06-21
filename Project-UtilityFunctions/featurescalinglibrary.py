#Utility functions
from defineInputs import getLabelName

import pandas as pd
import numpy as np

#Libraries for feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Normalizer


#This function is used to perform min-max feature scaing on the features in the given dataset
#Formula for Min-Max scalar feature scaling is (Xi-Xmin)/(Xmax-Xmin)
def featureScalingUsingMinMaxScaler(dataSetForFeatureScaling):
    print("****** Start feature scaling of the features present in the dataset using MinMaxScaler *****")

    numberOfColumnsInEncodedDataset = len(dataSetForFeatureScaling.columns)
    dataSetInArrayFormat = dataSetForFeatureScaling.values

    #Remove the label column from the dataset
    labelName = getLabelName()
    label = dataSetForFeatureScaling.pop(labelName)

    print(dataSetInArrayFormat)
    features = dataSetInArrayFormat[:,0:numberOfColumnsInEncodedDataset-1]
    print("\n****** Number of features in the dataset before performing scaling: ",np.size(features,1))
    print("\n****** Features in the dataset before performing scaling ***** \n",features)
    
    #Perform feature scaling
    scaler=MinMaxScaler(feature_range=(0,1))
    scaledFeatures=scaler.fit_transform(features)    
    print("\n****** Number of features in the dataset after performing scaling: ",np.size(scaledFeatures,1))
    print("\n****** Features in the dataset after performing scaling ***** \n",scaledFeatures)

    #Convert from array format to dataframe
    scaledFeatures = pd.DataFrame(scaledFeatures, columns=dataSetForFeatureScaling.columns)
    scaledFeatures = scaledFeatures.reset_index(drop=True)
    label = label.reset_index(drop=True)
    scaledFeatures[labelName]=label
    
    print("\n****** End of feature scaling of the features present in the dataset using MinMaxScaler *****\n")
    return scaledFeatures
	
#This function is used to perform StandardScalar feature scaing on the features in the given dataset
#This is also called as Z-score normalization
#Formula for StandardScalar scalar feature scaling is z = (x – mean) / standard-deviation. 
def featureScalingUsingStandardScalar(dataSetForFeatureScaling):
    print("****** Start feature scaling of the features present in the dataset using StandardScalar *****")

    numberOfColumnsInEncodedDataset = len(dataSetForFeatureScaling.columns)
    dataSetInArrayFormat = dataSetForFeatureScaling.values

    #Remove the label column from the dataset
    labelName = getLabelName()
    label = dataSetForFeatureScaling.pop(labelName)

    print(dataSetInArrayFormat)
    features = dataSetInArrayFormat[:,0:numberOfColumnsInEncodedDataset-1]
    print("\n****** Number of features in the dataset before performing scaling: ",np.size(features,1))
    print("\n****** Features in the dataset before performing scaling ***** \n",features)
    
    #Perform feature scaling
    scaler=StandardScaler()
    scaledFeatures=scaler.fit_transform(features)    
    print("\n****** Number of features in the dataset after performing scaling: ",np.size(scaledFeatures,1))
    print("\n****** Features in the dataset after performing scaling ***** \n",scaledFeatures)

    #Convert from array format to dataframe
    scaledFeatures = pd.DataFrame(scaledFeatures, columns=dataSetForFeatureScaling.columns)
    scaledFeatures = scaledFeatures.reset_index(drop=True)
    label = label.reset_index(drop=True)
    scaledFeatures[labelName]=label
    print("scaledFeatures.head(): ",scaledFeatures.head())
    print("scaledFeatures.shape: ",scaledFeatures.shape)
    
    print("\n****** End of feature scaling of the features present in the dataset using StandardScalar *****\n")
    return scaledFeatures
	
#This function is used to perform Binarizing feature scaing on the features in the given dataset
#It is used for binary thresholding of an array like matrix.
def featureScalingUsingBinarizer(dataSetForFeatureScaling):
    print("****** Start feature scaling of the features present in the dataset using Binarizer *****")

    numberOfColumnsInEncodedDataset = len(dataSetForFeatureScaling.columns)
    dataSetInArrayFormat = dataSetForFeatureScaling.values

    #Remove the label column from the dataset
    labelName = getLabelName()
    label = dataSetForFeatureScaling.pop(labelName)

    print(dataSetInArrayFormat)
    features = dataSetInArrayFormat[:,0:numberOfColumnsInEncodedDataset-1]
    print("\n****** Number of features in the dataset before performing scaling: ",np.size(features,1))
    print("\n****** Features in the dataset before performing scaling ***** \n",features)
    
    #Perform feature scaling
    scaledFeatures=Binarizer(0.0).fit(features).transform(features)
    print("\n****** Number of features in the dataset after performing scaling: ",np.size(scaledFeatures,1))
    print("\n****** Features in the dataset after performing scaling ***** \n",scaledFeatures)

    #Convert from array format to dataframe
    scaledFeatures = pd.DataFrame(scaledFeatures, columns=dataSetForFeatureScaling.columns)
    scaledFeatures = scaledFeatures.reset_index(drop=True)
    label = label.reset_index(drop=True)
    scaledFeatures[labelName]=label
    
    print("\n****** End of feature scaling of the features present in the dataset using Binarizer *****\n")
    return scaledFeatures
	
#This function is used to perform Normalizing feature scaing on the features in the given dataset
#It is used to rescale each sample. 
#Each sample (i.e. each row of the data matrix) with at least one non zero component 
#is rescaled independently of other samples so that its norm (l1 or l2) equals one.
def featureScalingUsingNormalizer(dataSetForFeatureScaling):
    print("****** Start feature scaling of the features present in the dataset using Normalizer *****")

    numberOfColumnsInEncodedDataset = len(dataSetForFeatureScaling.columns)
    dataSetInArrayFormat = dataSetForFeatureScaling.values

    #Remove the label column from the dataset
    labelName = getLabelName()
    label = dataSetForFeatureScaling.pop(labelName)

    print(dataSetInArrayFormat)
	
    features = dataSetInArrayFormat[:,0:numberOfColumnsInEncodedDataset-1]
    print("\n****** Number of features in the dataset before performing scaling: ",np.size(features,1))
    print("\n****** Features in the dataset before performing scaling ***** \n",features)
    
    #Perform feature scaling
    scaledFeatures=Normalizer().fit(features).transform(features)
    print("\n****** Number of features in the dataset after performing scaling: ",np.size(scaledFeatures,1))
    print("\n****** Features in the dataset after performing scaling ***** \n",scaledFeatures)

    #Convert from array format to dataframe
    scaledFeatures = pd.DataFrame(scaledFeatures, columns=dataSetForFeatureScaling.columns)
    scaledFeatures = scaledFeatures.reset_index(drop=True)
    label = label.reset_index(drop=True)
    scaledFeatures[labelName]=label
    
    print("\n****** End of feature scaling of the features present in the dataset using Normalizer *****\n")
    return scaledFeatures
	
