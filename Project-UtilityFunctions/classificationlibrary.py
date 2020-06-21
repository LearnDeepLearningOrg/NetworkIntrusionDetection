#Libraries for feature encoding
from sklearn.preprocessing import LabelEncoder

#Libraries for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier #RandomForestClassifier: Falls under wrapper methods (feature importance)
from sklearn.ensemble import ExtraTreesClassifier #ExtraTreesClassifier: Falls under wrapper methods (feature importance)
from sklearn.neighbors import KNeighborsClassifier 

#Libraries to measure the accuracy
from sklearn import metrics
from sklearn.metrics import accuracy_score

#import pandas library
import pandas as pd

#This function is used to perform classification using DecisionTreeClassifier
def classifyUsingDecisionTreeClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using DecisionTreeClassifier *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = DecisionTreeClassifier()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using DecisionTreeClassifier *****\n")
    return classifier, metrics.accuracy_score(ytrain, ytrainpred), metrics.accuracy_score(ytest, ytestpred)
	
#This function is used to perform classification using LogisticRegression
def classifyUsingLogisticRegression(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using LogisticRegression *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = LogisticRegression()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using LogisticRegression *****\n")
    return classifier, metrics.accuracy_score(ytrain, ytrainpred), metrics.accuracy_score(ytest, ytestpred)
	
#This function is used to perform classification using LinearDiscriminantAnalysis
def classifyUsingLinearDiscriminantAnalysis(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using LinearDiscriminantAnalysis *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using LinearDiscriminantAnalysis *****\n")
    return classifier, metrics.accuracy_score(ytrain, ytrainpred), metrics.accuracy_score(ytest, ytestpred)
	
#This function is used to perform classification using GuassianNaiveBayes
def classifyUsingGaussianNB(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using GuassianNaiveBayes *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = GaussianNB()
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using GuassianNaiveBayes *****\n")
    return classifier, metrics.accuracy_score(ytrain, ytrainpred), metrics.accuracy_score(ytest, ytestpred)

#This function is used to perform classification using RandomForestClassifier
def classifyUsingRandomForestClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using RandomForestClassifier *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using RandomForestClassifier *****\n")
    return classifier, metrics.accuracy_score(ytrain, ytrainpred), metrics.accuracy_score(ytest, ytestpred)

#This function is used to perform classification using RandomForestClassifier
def classifyUsingExtraTreesClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using ExtraTreesClassifier *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    print("trainingEncodedAndScaledDataset.shape: ",trainingEncodedAndScaledDataset.shape)

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = ExtraTreesClassifier(n_estimators=100)
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    print("testingEncodedAndScaledDataset.shape: ",testingEncodedAndScaledDataset.shape)

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using ExtraTreesClassifier *****\n")
    return classifier, metrics.accuracy_score(ytrain, ytrainpred), metrics.accuracy_score(ytest, ytestpred)

def classifyUsingKNNClassifier(trainingEncodedAndScaledDataset, testingEncodedAndScaledDataset):
    print("****** Start classification training using KNeighborsClassifier *****")
    xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
    ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

    labelencoder_ytrain = LabelEncoder()
    ytrain = labelencoder_ytrain.fit_transform(ytrain)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(xtrain,ytrain)

    ytrainpred = classifier.predict(xtrain)
    print("\n*** Classification accuracy score during model training: ", metrics.accuracy_score(ytrain, ytrainpred))

    xtest = testingEncodedAndScaledDataset.iloc[:, :-1].values
    ytest = testingEncodedAndScaledDataset.iloc[:, len(testingEncodedAndScaledDataset.columns)-1].values

    print("testingEncodedAndScaledDataset.shape: ",testingEncodedAndScaledDataset.shape)

    labelencoder_ytest = LabelEncoder()
    ytest = labelencoder_ytest.fit_transform(ytest)

    # Predicting the Test set results
    ytestpred = classifier.predict(xtest)
    print("*** Classification accuracy score during model testing: ", metrics.accuracy_score(ytest, ytestpred))
    print("\n****** End classification training using KNeighborsClassifier *****\n")
    return classifier, metrics.accuracy_score(ytrain, ytrainpred), metrics.accuracy_score(ytest, ytestpred)

def findingOptimumNumberOfNeighboursForKNN(trainingEncodedAndScaledDataset):
	print("****** Start finding optimum number of neighbours for KNN *****")
	xtrain = trainingEncodedAndScaledDataset.iloc[:, :-1].values
	ytrain = trainingEncodedAndScaledDataset.iloc[:, len(trainingEncodedAndScaledDataset.columns)-1].values

	labelencoder_ytrain = LabelEncoder()
	ytrain = labelencoder_ytrain.fit_transform(ytrain)

	# creating odd list of K for KNN
	neighbors = list(range(1, 150, 2))

	# empty list that will hold cv scores
	cv_scores = []

	# perform 10-fold cross validation
	for k in neighbors:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, xtrain, ytrain, cv=10, scoring='accuracy')
		cv_scores.append(scores.mean())
		print("With number of neighbours as {}, average score is {}".format(k,scores.mean()))

	# changing to misclassification error
	mse = [1 - x for x in cv_scores]

	# determining best k
	optimal_k = neighbors[mse.index(min(mse))]
	print("The optimal number of neighbors is {}".format(optimal_k))

	# plot misclassification error vs k
	plt.plot(neighbors, mse)
	plt.xlabel("Number of Neighbors K")
	plt.ylabel("Misclassification Error")
	plt.show()

	print("****** End finding optimum number of neighbours for KNN *****")