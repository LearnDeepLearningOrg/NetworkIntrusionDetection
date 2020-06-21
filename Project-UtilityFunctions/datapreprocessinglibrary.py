from sklearn.model_selection import train_test_split
from defineInputs import getLabelName

#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    anyMissingValuesInTheDataset = dataSet.isnull().values.any()
    return anyMissingValuesInTheDataset
	
#This function is used to check for duplicate records in a given dataSet
def checkForDulicateRecords (dataSet):
    totalRecordsInDataset = len(dataSet.index)
    numberOfUniqueRecordsInDataset = len(dataSet.drop_duplicates().index)
    anyDuplicateRecordsInTheDataset = False if totalRecordsInDataset == numberOfUniqueRecordsInDataset else True 
    print('Total number of records in the dataset: {}\nUnique records in the dataset: {}'.format(totalRecordsInDataset,numberOfUniqueRecordsInDataset))
    return anyDuplicateRecordsInTheDataset

#Split the complete dataSet into training dataSet and testing dataSet
def splitCompleteDataSetIntoTrainingSetAndTestingSet(completeDataSet):
	labelName = getLabelName()
	label = completeDataSet[labelName]
	features = completeDataSet.drop(labelName,axis=1)
	featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet=train_test_split(features,label,test_size=0.4, random_state=42)
	print("features.shape: ",features.shape)
	print("label.shape: ",label.shape)
	return featuresInPreProcessedTrainingDataSet,featuresInPreProcessedTestingDataSet,labelInPreProcessedTrainingDataSet,labelInPreProcessedTestingDataSet
