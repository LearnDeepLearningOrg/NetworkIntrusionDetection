#This function is to maintain the name of the label at a single place
def getLabelName():
	return 'attack_type'

def getPathToTrainingAndTestingDataSets():
	trainingFileNameWithAbsolutePath = "D:\\Learning\\DeepLearning\\Project-AttackDetectionSystem\\Datasets\\NSL-KDD\\KDDTrain+_20Percent.csv"
	testingFileNameWithAbsolutePath = "D:\\Learning\\DeepLearning\\Project-AttackDetectionSystem\\Datasets\\NSL-KDD\\KDDTest-21.csv"
	return trainingFileNameWithAbsolutePath, testingFileNameWithAbsolutePath
	
def modelPerformanceReport():
	modelPerformanceReport = 'D:\\Learning\\DeepLearning\\Project-AttackDetectionSystem\\ModelsAndTheirPerformanceReports\\ModelsPerformance031442020.1.xlsx'
	return modelPerformanceReport

def getPathToGenerateModels():
	generatedModelsPath = 'D:\\Learning\\DeepLearning\\Project-AttackDetectionSystem\\ModelsAndTheirPerformanceReports\\'
	return generatedModelsPath

### Models with the below configuration will be generated
def defineArrayOfModels():
	arrayOfModels = [
		[	
			"FeatureSelectionTechnique", 
			"FeatureEncodingTechnique", 
			"FeatureNormalizationTechnique", 
			"ClassificationTechnique", 
			"TrainAccuraccy", 
			"TestAccuraccy", 
			"ModelName", 
			"ModelFileName"
		],
		[
			"ExtraTreesClassifier",
			"OneHotEncoder",
			"Standardization",
			"DecisonTree"
		],
		[
			"ExtraTreesClassifier",
			"OneHotEncoder",
			"Standardization",
			"RandomForestClassifier"
		],
		[
			"ExtraTreesClassifier",
			"OneHotEncoder",
			"Standardization",
			"ExtraTreesClassifier"
		],
		[
			"ExtraTreesClassifier",
			"OneHotEncoder",
			"Standardization",
			"KNN"
		] 
	]
	print(arrayOfModels)
	return arrayOfModels

def defineArrayForPreProcessing():
	arrayOfModels = [
		[
			"ExtraTreesClassifier",
			"OneHotEncoder",
			"Standardization",
		]
	]
	print(arrayOfModels)
	return arrayOfModels
