# Python3 program to find combinations from n 
# arrays such that one element from each 
# array is present 

# function to prcombinations that contain 
# one element from each of the given arrays 
def print1(arr): 
	
	# number of arrays 
	n = len(arr) 

	# to keep track of next element 
	# in each of the n arrays 
	indices = [0 for i in range(n)] 

	while (1): 
		print("[")
        
		# prcurrent combination 
		for i in range(n): 
			print("'"+arr[i][indices[i]], end = "',") 
		print() 

		# find the rightmost array that has more 
		# elements left after the current element 
		# in that array 
		next = n - 1
		while (next >= 0 and
			(indices[next] + 1 >= len(arr[next]))): 
			next-=1

		# no such array is found so no more 
		# combinations left 
		if (next < 0): 
			return

		# if found move to next element in that 
		# array 
		indices[next] += 1

		# for all arrays to the right of this 
		# array current index again points to 
		# first element 
		for i in range(next + 1, n): 
			indices[i] = 0
		print("],")


# Driver Code 

# initializing a vector with 3 empty vectors 
arr = [[] for i in range(4)] 

# now entering data 
# [[1, 2, 3], [4], [5, 6]] 
arr[0].append('TheilsU') 
arr[0].append('Chi-SquaredTest') 
arr[0].append('RandomForestClassifier') 
arr[0].append('ExtraTreesClassifier') 

arr[1].append('OneHotEncoder') 
arr[1].append('LabelEncoder') 
arr[1].append('BinaryEncoder') 
arr[1].append('FrequencyEncoder') 

arr[2].append('Min-Max') 
arr[2].append('Standardization') 
arr[2].append('Binarizing') 
arr[2].append('Normalizing') 

arr[3].append('DecisonTree') 
arr[3].append('RandomForestClassifier') 
arr[3].append('ExtraTreesClassifier') 
arr[3].append('LogisticRegressionRegression') 
arr[3].append('LinearDiscriminantAnalysis') 
arr[3].append('GuassianNaiveBayes') 

print1(arr) 

# This code is contributed by mohit kumar 
