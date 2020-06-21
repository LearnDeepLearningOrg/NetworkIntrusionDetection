import pandas as pd

#This function is used to load CSV file from the 'data' directory 
#in the present working directly 
def loadCSV (fileNameWithAbsolutePath):
    dataSet = pd.read_csv(fileNameWithAbsolutePath)
    return dataSet
